"""Expand the fixed-representation audit with local matched reference sets.

Each retained candidate has the same number of distinct unlabelled references. All
choices use catalogue availability, position, flux and area only. Released
representation values and expert grades never enter the matching cost.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from . import data, sampling

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260913
REFERENCES_PER_CANDIDATE = 1
COLUMNS = sampling.COLUMNS + ['vis_det', 'ellipticity', 'det_quality_flag',
    'flux_vis_1fwhm_aper', 'fluxerr_vis_1fwhm_aper',
    'flux_y_1fwhm_aper', 'fluxerr_y_1fwhm_aper']
_LOCK = threading.Lock()


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def record(root, path, *, query=None, url=None, fresh=False):
    """Keep expanded provenance separate from the original experiment."""
    root, path = Path(root), Path(path)
    manifest_path = root/'data/expanded_manifest.json'
    with _LOCK:
        m = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            'project': 'LensLab expanded matched-set audit', 'sources': {}}
        key = path.relative_to(root).as_posix()
        old = m['sources'].get(key, {})
        m['sources'][key] = {'url': url, 'query': query,
            'retrieved_utc': datetime.now(timezone.utc).isoformat() if fresh else old.get('retrieved_utc', datetime.now(timezone.utc).isoformat()),
            'bytes': path.stat().st_size, 'sha256': _sha(path)}
        tmp = manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(m, indent=2)+'\n')
        tmp.replace(manifest_path)


def query_csv(root, query):
    """Three bounded retries for transient failures; stop on access denial."""
    root = Path(root)
    key = hashlib.sha256(query.encode()).hexdigest()[:16]
    path = root/f'data/raw/expanded_pool_{key}.csv'
    url = data.TAP + '?' + urllib.parse.urlencode({'REQUEST':'doQuery', 'LANG':'ADQL',
        'FORMAT':'csv', 'QUERY':query, 'MAXREC':10000})
    if path.exists():
        mpath = root/'data/expanded_manifest.json'
        if mpath.exists():
            with _LOCK:
                entry = json.loads(mpath.read_text()).get('sources', {}).get(path.relative_to(root).as_posix())
            if entry and (entry['query'] != query or entry['sha256'] != _sha(path)):
                raise ValueError('Expanded cache provenance/checksum mismatch.')
        fresh = False
    else:
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent':'LensLab public-data exploration'})
                with urllib.request.urlopen(request, timeout=50) as response:
                    body = response.read()
                if not body:
                    raise ValueError('Empty TAP response')
                tmp = path.with_suffix('.part')
                tmp.write_bytes(body)
                tmp.replace(path)
                break
            except urllib.error.HTTPError as error:
                if error.code in (401, 403) or attempt == 2:
                    raise
                time.sleep(attempt + 1)
            except (urllib.error.URLError, TimeoutError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        fresh = True
    frame = pd.read_csv(path, dtype={'object_id':str})
    frame.columns = frame.columns.str.strip()
    frame['object_id'] = frame.object_id.str.strip()
    if len(frame) >= 10000:
        raise ValueError('Expanded pool reached MAXREC; split before matching.')
    record(root, path, query=query, url=url, fresh=fresh)
    return frame


def query_pools(root, selected):
    groups = [selected.iloc[i:i+10] for i in range(0, len(selected), 10)]
    stop_requests = threading.Event()
    def batch(item):
        if stop_requests.is_set():
            raise RuntimeError('Acquisition stopped after an access denial.')
        i, group = item
        circles = ' OR '.join(f'DISTANCE({r.right_ascension},{r.declination},right_ascension,declination)<0.03'
            for _, r in group.iterrows())
        query = (f"SELECT {','.join(COLUMNS)} FROM catalogue.mer_catalogue WHERE ({circles}) "
            'AND vis_det=1 AND segmentation_area>=300 AND flux_detection_total>3.6307805477')
        try:
            frame = query_csv(root, query)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                stop_requests.set()
            raise
        print(f'Expanded local pools {i+1}/{len(groups)}: {len(frame)} rows', flush=True)
        return frame
    with ThreadPoolExecutor(max_workers=3) as executor:
        frames = list(executor.map(batch, enumerate(groups)))
    joined = pd.concat(frames, ignore_index=True)
    return joined.drop_duplicates('object_id').sort_values('object_id').reset_index(drop=True)


def candidate_options(candidates, pool):
    """All possible edges satisfying the original, fixed matching calipers."""
    options = {}
    for _, row in candidates.iterrows():
        distance = sampling.separation_deg(pool.right_ascension.to_numpy(), pool.declination.to_numpy(), row.right_ascension, row.declination)
        dm = pool.mag_total.to_numpy() - row.mag_total
        da = pool.log_area.to_numpy() - row.log_area
        good = ((distance <= sampling.RADIUS_DEG) & (np.abs(dm) <= sampling.MAG_CALIPER)
                & (np.abs(da) <= sampling.AREA_CALIPER))
        ix = np.flatnonzero(good)
        cost = (dm[ix]/sampling.MAG_CALIPER)**2 + (da[ix]/sampling.AREA_CALIPER)**2 + .01*(distance[ix]/sampling.RADIUS_DEG)**2
        order = np.lexsort((pool.object_id.to_numpy()[ix], cost))
        options[row.object_id] = [(int(j), float(c)) for j, c in zip(ix[order], cost[order])]
    return options


def assign(candidates, options, ratio):
    """Seeded greedy complete sets: failed candidates consume no references."""
    used, assigned = set(), {}
    for identifier in candidates.sort_values('object_id').sample(frac=1, random_state=SEED).object_id:
        possible = [(ix, cost) for ix, cost in options[identifier] if ix not in used]
        if len(possible) >= ratio:
            chosen = possible[:ratio]
            assigned[identifier] = chosen
            used.update(ix for ix, _ in chosen)
    return assigned


def acquire(root=ROOT):
    root = Path(root)
    ratio = REFERENCES_PER_CANDIDATE
    candidates_path = root/'data/processed/candidates.csv'
    candidates = pd.read_csv(candidates_path, dtype={'object_id':str, 'tile_id':str})
    published = pd.read_csv(root/'data/raw/q1_discovery_engine_lens_catalog.csv', dtype={'object_id':str})
    available = set(pd.read_csv(root/'data/raw/zoobot/available_ids.csv', dtype={'object_id':str}).object_id)
    has_rep = candidates.object_id.isin(available)
    quality = sampling.eligible(candidates)
    selected = sampling.add_matching_columns(candidates.loc[has_rep & quality].sort_values('object_id')).reset_index(drop=True)
    pool_all = query_pools(root, selected)
    known = pool_all.object_id.isin(published.object_id)
    pool_unknown = pool_all.loc[~known].copy()
    no_rep = ~pool_unknown.object_id.isin(available)
    pool = sampling.add_matching_columns(pool_unknown.loc[~no_rep].copy())
    known_positions = published.loc[np.isfinite(published.right_ascension) & np.isfinite(published.declination)]
    separation = SkyCoord(pool.right_ascension.to_numpy()*u.deg, pool.declination.to_numpy()*u.deg).match_to_catalog_sky(
        SkyCoord(known_positions.right_ascension.to_numpy()*u.deg, known_positions.declination.to_numpy()*u.deg))[1].arcsec
    pool['nearest_published_candidate_arcsec'] = separation
    too_close = separation < sampling.MIN_KNOWN_SEPARATION_ARCSEC
    pool = pool.loc[~too_close].sort_values('object_id').reset_index(drop=True)
    options = candidate_options(selected, pool)
    support_reference_indices = set(ix for edges in options.values() for ix, _ in edges)
    comparisons = {}
    for trial_ratio in (1,2,3,4):
        assignments = assign(selected, options, trial_ratio)
        kept = selected[selected.object_id.isin(assignments)]
        comparisons[str(trial_ratio)] = {'matched_candidates':len(kept), 'reference_objects':len(kept)*trial_ratio,
            'total_objects':len(kept)*(1+trial_ratio), 'matched_by_grade':kept.grade.value_counts().sort_index().to_dict()}
    fixed = assign(selected, options, ratio)
    option_counts = {identifier:len(value) for identifier,value in options.items()}
    n_before = pd.Series(option_counts)
    summary = {
        'seed':SEED, 'references_per_candidate':ratio,
        'design':f'All eligible published candidates; exactly {ratio} local unlabelled reference(s) per retained candidate, without replacement.',
        'candidate_flow':{'published_unique_candidates':len(candidates), 'with_published_representation':int(has_rep.sum()),
            'missing_representation':int((~has_rep).sum()), 'excluded_common_cuts_after_representation':int((has_rep & ~quality).sum()),
            'eligible_candidates':len(selected), 'retained_candidates':len(fixed),
            'at_least_required_options_before_competition':int(n_before.ge(ratio).sum()),
            'fewer_than_required_options_before_competition':int(n_before.lt(ratio).sum()),
            'dropped_after_competition':int(n_before.ge(ratio).sum()-len(fixed)),
            'retained_references':len(fixed)*ratio, 'final_objects':len(fixed)*(ratio+1)},
        'by_grade':{'published':candidates.grade.value_counts().sort_index().to_dict(),
            'with_representation':candidates[has_rep].grade.value_counts().sort_index().to_dict(),
            'eligible':selected.grade.value_counts().sort_index().to_dict(),
            'retained':selected[selected.object_id.isin(fixed)].grade.value_counts().sort_index().to_dict()},
        'reference_pool_flow':{'published_representations_total':len(available), 'archive_unique_objects_with_common_cuts_near_candidates':len(pool_all),
            'query_batches_of_at_most_ten_circles':int(np.ceil(len(selected)/10)),
            'published_candidate_positions_used_for_exclusion':len(known_positions),
            'excluded_published_candidate_ids':int(known.sum()), 'excluded_missing_representation':int(no_rep.sum()),
            'excluded_within_20arcsec_of_any_published_position':int(too_close.sum()), 'eligible_reference_pool':len(pool),
            'eligible_references_matching_at_least_one_candidate':len(support_reference_indices),
            'selected_references':len(fixed)*ratio},
        'candidate_options_histogram':n_before.value_counts().sort_index().to_dict(),
        'candidate_options_quantiles':n_before.quantile([0,.1,.25,.5,.75,.9,1]).to_dict(),
        'ratio_feasibility_only':comparisons,
        'ratio_choice':'The proposed 1:4 extension retained only 438 of 1415 eligible candidates. The 1:1 design was therefore selected using catalogue feasibility alone, before computing any extension clusters or outcome statistics. It preserves 1126 candidates versus 844 at 1:2, avoids new ratio weighting, and keeps the original calipers. The earlier 1:1 pilot was already known.',
        'calipers':{'radius_arcsec':108.0, 'absolute_magnitude_difference':.3, 'absolute_log10_area_difference':.2,
            'minimum_reference_distance_from_any_published_candidate_arcsec':20.0},
        'common_cuts':'VIS detection=1; segmentation_area>=300 pixels; detection total flux>3.6307805477 microJy (AB<22.5); finite coordinates.',
        'matching_order':'Sort candidate object_id as strings, then pandas sample(frac=1, random_state=20260913).',
        'matching_cost':'(delta_mag/0.3)^2 + (delta_log10_area/0.2)^2 + 0.01*(separation_deg/0.03)^2; ties by reference object_id.',
        'incomplete_sets':f'If fewer than {ratio} unused admissible reference(s) remain, exclude the candidate and reserve no reference. No calipers are widened; no resampling of candidates.',
        'not_used_for_matching':['expert grade or score', 'colour', 'ellipticity', 'image pixels', 'representation values'],
        'interpretation':f'Reference objects are unlabelled, not verified non-lenses. The {100/(ratio+1):g} percent candidate fraction is an analysis design, not an estimated survey prevalence.',
        'spatial_scope':'Local circles around all 1415 eligible candidates across the Q1 deep fields; not limited to the initial three EDA patches.',
        'representation_source':'All retained objects use the unchanged 40-D file released at Zenodo 15106473. No new encoder is run to fill missing candidate IDs.',
        'input_candidates_sha256':_sha(candidates_path),
        'input_published_catalogue_sha256':_sha(root/'data/raw/q1_discovery_engine_lens_catalog.csv'),
        'input_available_ids_sha256':_sha(root/'data/raw/zoobot/available_ids.csv'),
        'input_representation_sha256':_sha(root/'data/raw/zoobot/representations_pca_40.parquet')}
    candidates['has_published_representation'] = has_rep
    candidates['passes_common_cuts'] = quality
    candidates['admissible_references_before_competition'] = candidates.object_id.map(option_counts).astype('Int64')
    candidates['selection_status'] = np.select([~has_rep, has_rep & ~quality,
        candidates.object_id.isin(fixed), candidates.object_id.map(option_counts).fillna(0).lt(ratio)],
        ['missing_published_representation','fails_common_cuts','retained','fewer_than_required_admissible_references'],
        default='insufficient_unused_references_after_competition')
    output = []
    selected_index = selected.set_index('object_id', drop=False)
    for identifier, matches in fixed.items():
        candidate = selected_index.loc[identifier]
        set_id = 'set_'+identifier
        base = {name:candidate[name] for name in COLUMNS}
        base.update(set_id=set_id, role='candidate', grade=candidate.grade, field=candidate.field,
            tile_id=candidate.tile_id, mag_total=float(candidate.mag_total), log_area=float(candidate.log_area),
            reference_rank=0, match_distance_arcsec=0.0, match_delta_mag=0.0, match_delta_log_area=0.0,
            match_cost=0.0, nearest_published_candidate_arcsec=0.0)
        output.append(base)
        for rank,(ix,cost) in enumerate(matches,1):
            ref = pool.iloc[ix]
            item = {name:ref[name] for name in COLUMNS}
            item.update(set_id=set_id, role='reference', grade='unlabelled', field=candidate.field,
                tile_id='', mag_total=float(ref.mag_total), log_area=float(ref.log_area),
                reference_rank=rank, match_distance_arcsec=float(sampling.separation_deg(ref.right_ascension, ref.declination,candidate.right_ascension,candidate.declination)*3600),
                match_delta_mag=float(ref.mag_total-candidate.mag_total), match_delta_log_area=float(ref.log_area-candidate.log_area),
                match_cost=cost, nearest_published_candidate_arcsec=float(ref.nearest_published_candidate_arcsec))
            output.append(item)
    sample = pd.DataFrame(output).sort_values(['set_id','reference_rank']).reset_index(drop=True)
    sample['snr_total'] = sample.flux_detection_total / sample.fluxerr_detection_total.where(sample.fluxerr_detection_total>0)
    sample['color_vis_y'] = -2.5*np.log10(sample.flux_vis_1fwhm_aper.where(sample.flux_vis_1fwhm_aper>0) / sample.flux_y_1fwhm_aper.where(sample.flux_y_1fwhm_aper>0))
    sample['path_vis'] = sample.object_id.map(lambda v:f'data/raw/cutouts/{v}_VIS.fits')
    assert sample.object_id.is_unique
    assert sample.object_id.isin(available).all()
    assert sample.groupby('set_id').size().eq(ratio+1).all()
    assert sample.query("role=='candidate'").groupby('set_id').size().eq(1).all()
    references = sample.query("role=='reference'")
    assert references.match_delta_mag.abs().le(.3).all()
    assert references.match_delta_log_area.abs().le(.2).all()
    assert references.match_distance_arcsec.le(108).all()
    assert references.nearest_published_candidate_arcsec.ge(20).all()
    assert not references.object_id.isin(published.object_id).any()
    summary['candidate_selection_status_by_grade'] = candidates.groupby(['selection_status','grade']).size().unstack(fill_value=0).to_dict(orient='index')
    summary['retained_candidate_counts_by_field'] = sample.query("role=='candidate'").field.value_counts().sort_index().to_dict()
    summary['balance'] = {}
    for variable in ['mag_total','log_area','snr_total']:
        cv = sample.loc[sample.role.eq('candidate'),variable].dropna()
        rv = sample.loc[sample.role.eq('reference'),variable].dropna()
        smd = (cv.mean()-rv.mean())/np.sqrt((cv.var(ddof=1)+rv.var(ddof=1))/2)
        summary['balance'][variable] = {'candidate_median':float(cv.median()),'reference_median':float(rv.median()),
            'standardized_mean_difference':float(smd)}
    for frame, name in [(candidates,'expanded_candidates.csv'), (pool,'expanded_reference_pool.csv'), (sample,'expanded_sample.csv')]:
        path = root/'data/processed'/name
        frame.to_csv(path, index=False)
        record(root,path)
    summary['sample_sha256'] = _sha(root/'data/processed/expanded_sample.csv')
    summary_path = root/'results/expanded_sampling_summary.json'
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['candidate_flow'],indent=2),flush=True)
    print(json.dumps(comparisons,indent=2),flush=True)
    return sample, summary


if __name__ == '__main__':
    acquire()
