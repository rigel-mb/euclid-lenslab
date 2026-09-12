"""Download the public Q1 data used by the LensLab notebooks.

Run ``python run.py --download`` from the project root. Downloads are cached.
The reference sample is unlabelled: it is not a catalogue of confirmed non-lenses.
"""

from __future__ import annotations

import argparse
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
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TAP = "https://eas.esac.esa.int/tap-server/tap/sync"
ZENODO = "https://zenodo.org/records/15025832/files/"
SEED = 20260912
# Three explicitly delimited patches within the Q1 deep-field footprint.
REGIONS = {"north": (269.73, 66.03, 0.6),
           "south": (61.20, -48.40, 0.6),
           "fornax": (53.50, -28.10, 0.6)}
MER = ["object_id", "right_ascension", "declination", "segmentation_area",
       "ellipticity", "kron_radius", "fwhm", "vis_det", "det_quality_flag",
       "point_like_prob", "extended_prob", "spurious_prob", "blended_prob",
       "flux_detection_total", "fluxerr_detection_total"]
MER += [f"{prefix}_{band}_{kind}" for band in ("vis", "y", "j", "h")
        for kind in ("1fwhm_aper", "2fwhm_aper", "sersic")
        for prefix in ("flux", "fluxerr")]
MORPH = ["concentration", "asymmetry", "smoothness", "gini", "moment_20",
         "sersic_visnir_flags", "sersic_visnir_reduced_chi2"]
MORPH += [f"sersic_sersic_{band}_{parameter}" for band in ("vis", "nir")
          for parameter in ("index", "radius", "axis_ratio")]
PHZ = ["phz_median", "phz_mode_1", "phz_mode_1_area", "phz_flags",
       "phz_classification", "phz_weight"]
_LOCK = threading.Lock()


def _save_manifest(manifest: dict) -> None:
    """An atomic replacement lets concurrent cache readers see complete JSON."""
    temporary = DATA / f"manifest-{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(DATA / "manifest.json")


def _record(path: Path, *, url: str | None = None, query: str | None = None,
            parameters: dict | None = None, fresh: bool = False) -> None:
    """Preserve cached retrieval dates; record a new date for a new download."""
    manifest_path = DATA / "manifest.json"
    with _LOCK:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            "project": "LensLab", "seed": SEED, "sources": {}, "notes": []}
        key = path.relative_to(ROOT).as_posix()
        previous = manifest["sources"].get(key, {})
        manifest["sources"][key] = {
            "url": url, "query": query, "parameters": parameters,
            "retrieved_utc": (datetime.now(timezone.utc).isoformat() if fresh else
                              previous.get("retrieved_utc", datetime.now(timezone.utc).isoformat())),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        _save_manifest(manifest)


def download(url: str, destination: Path, *, query: str | None = None,
             parameters: dict | None = None, timeout: int = 50) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        manifest_path = DATA / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text()).get("sources", {}).get(
                destination.relative_to(ROOT).as_posix(), {})
            if previous and (previous.get("url") != url or previous.get("query") != query
                             or previous.get("parameters") != parameters):
                raise ValueError(f"Cached request differs for {destination}; use a fresh data directory.")
            if previous.get("sha256") and hashlib.sha256(destination.read_bytes()).hexdigest() != previous["sha256"]:
                raise ValueError(f"Cached bytes changed for {destination}; restore or remove this file.")
        _record(destination, url=url, query=query, parameters=parameters)
        return destination
    payload = urllib.parse.urlencode(parameters).encode() if parameters else None
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, data=payload,
                                             headers={"User-Agent": "LensLab public-data exploration"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()
            if not content:
                raise ValueError(f"Empty response: {url}")
            destination.write_bytes(content)
            _record(destination, url=url, query=query, parameters=parameters, fresh=True)
            return destination
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Download failed: {url}") from last_error


def query_csv(query: str, name: str, *, max_rows: int = 200000) -> pd.DataFrame:
    params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
              "QUERY": query, "MAXREC": max_rows}
    url = TAP + "?" + urllib.parse.urlencode(params)
    path = download(url, DATA / "raw" / f"{name}.csv", query=query)
    # Always preserve 64-bit object identifiers, including negative values.
    result = pd.read_csv(path, dtype={"object_id": "string"})
    result.columns = result.columns.str.strip()
    if "object_id" in result:
        result["object_id"] = result["object_id"].str.strip()
    return result


def fetch_features(ids: list[str], prefix: str, *, enrich: bool = True) -> pd.DataFrame:
    """Small ID batches keep anonymous synchronous requests bounded."""
    selected = [f"m.{column}" for column in MER]
    joins = ""
    if enrich:
        selected += [f"s.{column}" for column in MORPH]
        selected += [f"p.{column}" for column in PHZ]
        joins = (" LEFT OUTER JOIN catalogue.mer_morphology AS s ON m.object_id=s.object_id"
                 " LEFT OUTER JOIN catalogue.phz_photo_z AS p ON m.object_id=p.object_id")
    chunks = [ids[index:index + 250] for index in range(0, len(ids), 250)]

    def batch(item):
        index, group = item
        query = (f"SELECT {','.join(selected)} FROM catalogue.mer_catalogue AS m{joins} "
                 f"WHERE m.object_id IN ({','.join(str(int(value)) for value in group)})")
        result = query_csv(query, f"{prefix}_{index:03d}")
        print(f"{prefix}: batch {index + 1}/{len(chunks)}, {len(result)} rows", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        tables = list(pool.map(batch, enumerate(chunks)))
    result = pd.concat(tables, ignore_index=True)
    if result.object_id.duplicated().any():
        raise ValueError(f"Non-unique object_id after ESA joins: {prefix}")
    return result


def field_name(ra: pd.Series, dec: pd.Series) -> np.ndarray:
    return np.where(dec > 0, "north", np.where(dec > -38, "fornax", "south"))


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    # Preserve negative fluxes. Missing values arrive as NaN from the archive CSV.
    for column in frame.select_dtypes(include="number"):
        values = frame[column]
        frame[column] = values.mask(~np.isfinite(values))
    for band in ("vis", "y", "j", "h"):
        flux = frame[f"flux_{band}_1fwhm_aper"]
        error = frame[f"fluxerr_{band}_1fwhm_aper"]
        frame[f"mag_{band}"] = 23.9 - 2.5 * np.log10(flux.where(flux > 0))
        frame[f"snr_{band}"] = flux / error.where(error > 0)
    for first, second in (("vis", "y"), ("y", "j"), ("j", "h")):
        frame[f"color_{first}_{second}"] = frame[f"mag_{first}"] - frame[f"mag_{second}"]
    frame["field"] = field_name(frame.right_ascension, frame.declination)
    # The Q1 DPDD corrects the older (inverted) description in TAP_SCHEMA.
    galaxy = (frame.phz_classification.fillna(0).astype(int) & 2) == 2
    frame["phz_valid"] = frame.phz_median.ge(0) & frame.phz_flags.eq(0) & galaxy
    return frame


def build_candidates() -> pd.DataFrame:
    path = download(ZENODO + "q1_discovery_engine_lens_catalog.csv?download=1",
                    DATA / "raw" / "q1_discovery_engine_lens_catalog.csv")
    catalogue = pd.read_csv(path, dtype={"object_id": "string", "tile_index": "string"})
    candidates = catalogue.loc[catalogue["subset"].eq("discovery_engine")].copy()
    if len(candidates) != 2415 or candidates.object_id.duplicated().any():
        raise ValueError("The published catalogue changed: review its selection before rerunning.")
    features = fetch_features(candidates.object_id.tolist(), "candidates_features")
    # Existing catalogue columns remain the record of labels; ESA provides measurements.
    duplicate = [c for c in features if c in candidates and c != "object_id"]
    features = features.rename(columns={c: f"esa_{c}" for c in duplicate})
    candidates = candidates.merge(features, on="object_id", how="left", validate="one_to_one")
    for column in duplicate:
        if column in MER:
            candidates[column] = candidates[f"esa_{column}"].where(
                candidates[f"esa_{column}"].notna(), candidates[column])
    candidates["tile_id"] = candidates["tile_index"]
    candidates["has_esa_photometry"] = candidates.flux_detection_total.notna()
    candidates = add_derived(candidates)
    output = DATA / "processed" / "candidates.csv"
    candidates.to_csv(output, index=False)
    _record(output)
    print(f"Candidates ready: {len(candidates)}", flush=True)
    return candidates


def build_reference(candidates: pd.DataFrame, size: int = 10000) -> pd.DataFrame:
    frames = []
    for label, (ra, dec, radius) in REGIONS.items():
        where = (f"DISTANCE({ra},{dec},right_ascension,declination)<{radius} "
                 "AND vis_det=1 AND segmentation_area>=300 AND flux_detection_total>3.6307805477")
        count = query_csv(f"SELECT COUNT(*) AS n FROM catalogue.mer_catalogue WHERE {where}",
                          f"reference_count_{label}").iloc[0, 0]
        query = ("SELECT object_id,right_ascension,declination,segmentation_area,flux_detection_total "
                 f"FROM catalogue.mer_catalogue WHERE {where}")
        frame = query_csv(query, f"reference_frame_{label}")
        if len(frame) != int(count):
            raise ValueError(f"Incomplete reference frame {label}: {len(frame)} versus {count}")
        frame["region"] = label
        frames.append(frame)
    parent = pd.concat(frames, ignore_index=True).drop_duplicates("object_id")
    # Remove all labelled candidates, not just A/B, to keep the reference disjoint.
    parent = parent.loc[~parent.object_id.isin(candidates.object_id)].copy()
    sample = parent.sort_values("object_id").sample(n=min(size, len(parent)), random_state=SEED)
    features = fetch_features(sample.object_id.tolist(), "reference_features")
    reference = sample[["object_id", "region"]].merge(features, how="left", on="object_id", validate="one_to_one")
    reference = add_derived(reference)
    reference["label_status"] = "unlabelled"
    reference.to_csv(DATA / "processed" / "reference.csv", index=False)
    _record(DATA / "processed" / "reference.csv")
    with _LOCK:
        path = DATA / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["reference_sampling"] = {
            "regions_ra_dec_radius_deg": REGIONS,
            "selection": "vis_det=1; segmentation_area>=300 pixels; flux_detection_total>3.6307805477 microJy",
            "magnitude_note": "The flux threshold corresponds to AB 22.5 for detection total flux; this is our own cut, not the SLDE selection.",
            "frame_rows_after_excluding_candidates": len(parent), "sample_rows": len(reference),
            "method": "Simple random sample without replacement after sorting object_id, pandas random_state=20260912",
            "scope": "Only the three specified patches and the explicit selection; not all Q1. Unlabelled, not confirmed non-lenses."}
        _save_manifest(manifest)
    return reference


def download_image(row: pd.Series | dict, band: str = "VIS", radius_arcsec: float = 8.0) -> Path:
    """Fetch a small MER cutout in a given band, without downloading its mosaic."""
    object_id = str(row["object_id"])
    ra, dec = float(row["right_ascension"]), float(row["declination"])
    band = band.upper()
    if band not in {"VIS", "Y", "J", "H"}:
        raise ValueError("band must be VIS, Y, J, or H")
    filter_clause = "instrument_name='VIS'" if band == "VIS" else f"filter_name='NIR_{band}'"
    query = ("SELECT TOP 1 file_name,file_path,filter_name FROM q1.mosaic_product WHERE "
             f"{filter_clause} AND INTERSECTS(CIRCLE({ra},{dec},{radius_arcsec / 3600}),fov)=1 "
             "ORDER BY file_name")
    products = query_csv(query, f"image_product_{object_id}_{band}")
    if products.empty:
        raise ValueError(f"No {band} mosaic covers {object_id}")
    product = products.iloc[0]
    parameters = {"TAPCLIENT": "ASTROQUERY", "FILEPATH": f"{product.file_path}/{product.file_name}",
                  "POS": f"CIRCLE,{ra},{dec},{radius_arcsec / 3600}"}
    path = DATA / "raw" / "cutouts" / f"{object_id}_{band}.fits"
    return download("https://eas.esac.esa.int/sas-cutout/cutout", path, parameters=parameters)


def build_gallery(candidates: pd.DataFrame, per_grade: int = 6) -> pd.DataFrame:
    sample = (candidates.sort_values("object_id").groupby("grade", group_keys=False)
              .sample(n=per_grade, random_state=SEED))
    def fetch(row):
        record = row[["object_id", "id_str", "grade", "right_ascension", "declination"]].to_dict()
        for band in ("VIS", "Y", "J", "H"):
            try:
                path = download_image(row, band)
                record[f"path_{band.lower()}"] = path.relative_to(ROOT).as_posix()
            except (RuntimeError, ValueError) as error:
                record[f"error_{band.lower()}"] = str(error)
        record["radius_arcsec"] = 8.0
        return record

    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(fetch, [row for _, row in sample.iterrows()]):
            rows.append(record)
            pd.DataFrame(rows).to_csv(DATA / "processed" / "gallery.csv", index=False)
            print(f"Gallery: {len(rows)}/{len(sample)}", flush=True)
    _record(DATA / "processed" / "gallery.csv")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-size", type=int, default=10000)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    acquire(images=not args.skip_images, reference_size=args.reference_size)


def acquire(root: Path = ROOT, images: bool = True, reference_size: int = 10000):
    """Acquire or reuse the exact cached inputs; return the two processed tables."""
    global ROOT, DATA
    ROOT = Path(root).resolve()
    DATA = ROOT / "data"
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    download("https://zenodo.org/api/records/15025832", DATA / "raw" / "zenodo_record.json")
    query_csv("SELECT table_name,column_name,datatype,description,unit FROM TAP_SCHEMA.columns "
              "WHERE table_name='catalogue.mer_catalogue' OR table_name='catalogue.mer_morphology' "
              "OR table_name LIKE 'catalogue.phz%'", "schema")
    candidates = build_candidates()
    for name in ("modeling_lens_mass.csv", "modeling_mge_magnitude.csv"):
        download(ZENODO + name + "?download=1", DATA / "raw" / name)
    reference = build_reference(candidates, reference_size)
    if images:
        # Use the serialized coordinates so direct and offline runs make identical requests.
        gallery_candidates = pd.read_csv(DATA / "processed" / "candidates.csv", dtype={"object_id": "string"})
        build_gallery(gallery_candidates)
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data_notes"] = {
        "zenodo_license": "CC-BY-4.0 (API metadata.license.id)",
        "credits": "Euclid/ESA/Euclid Consortium; SLDE catalogue and modelling: Walmsley et al. and collaborators.",
        "q1_archive_scope": "catalogue.mer_catalogue TAP description links to /dr/q1/dpdd; image products explicitly queried from q1.mosaic_product.",
        "q1_phz_definition_url": "https://euclid.esac.esa.int/dr/q1/dpdd/phzdpd/dpcards/phz_phzpfoutputcatalog.html",
        "phz_quality": "DPDD Q1: classification bit value 2 is galaxy; flags=0 is OK. Older TAP column descriptions invert star/galaxy and describe old flags.",
        "phz_valid": "finite phz_median>=0, phz_flags==0, and (phz_classification & 2)==2; a catalogue quality cut, not a guarantee of redshift accuracy.",
        "cleaning": "Non-finite numerical values become NaN. No arbitrary sentinels removed; negative fluxes retained. Magnitudes only where flux>0, SNR only where error>0.",
        "photometry": "mag_band uses PSF-matched 1FWHM aperture flux in microJy: 23.9-2.5log10(flux). It is an aperture magnitude, not total luminosity.",
        "morphology": "Direct MER descriptors and Sersic measurements; no Galaxy Zoo/Zoobot morphology probabilities downloaded.",
        "candidate_labels": "Published grade used verbatim, not recomputed from expert_score. A/B are high-confidence candidates; C is uncertain, not a confirmed negative.",
        "modeling": "335 rows in the downloaded mass table despite 336 in the record description; 334 id_str match the candidate catalogue. Unmatched model retained as published.",
        "gallery": "Simple random sample of six objects per published grade, seed20260912, sorted object_id. Four bands; requested radius8arcsec; plots should read the WCS scale."}
    _save_manifest(manifest)
    return candidates, reference


if __name__ == "__main__":
    main()
