"""Show fixed candidate/reference pairs and their actual ten image neighbours.

Pairs are sampled before inspecting their neighbours or fetching any images.
Missing previews stay in their original rank: no objects are substituted.
The published Zoobot coordinates, not these VIS previews, define proximity.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from . import encoder
from .sampling import reproject_stamp

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260918
K = 10


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def select_pairs(root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    """Freeze one pair per grade without consulting the neighbour graph."""
    root = Path(root)
    sample_path = root / "data/processed/expanded_sample.csv"
    sample = pd.read_csv(sample_path, dtype={"object_id": str, "set_id": str})
    candidates = sample.loc[sample.role.eq("candidate")].sort_values("object_id")
    pairs = []
    for grade in ("A", "B", "C"):
        candidate = candidates.loc[candidates.grade.eq(grade)].sample(n=1, random_state=SEED).iloc[0]
        reference = sample.loc[sample.set_id.eq(candidate.set_id) & sample.role.eq("reference")]
        if len(reference) != 1:
            raise ValueError("Each gallery candidate needs exactly one matched reference.")
        pairs.append({"grade": grade, "set_id": candidate.set_id,
                      "candidate_id": candidate.object_id,
                      "reference_id": reference.iloc[0].object_id})
    selection = {
        "seed": SEED,
        "selection_rule": "For A, B and C independently: sort candidate object IDs, then pandas.sample(n=1, random_state=20260918). Keep its matched reference. Freeze before reading neighbour results or image availability.",
        "sample_sha256": encoder.sha256(sample_path),
        "pairs": pairs,
    }
    path = root / "results/neighbour_gallery_selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("The fixed gallery selection changed; do not replace examples after seeing results.")
    _write_json(path, selection)
    return sample, selection


def _acquire(root: Path, selected: pd.DataFrame) -> tuple[dict, dict]:
    folder = root / "data/raw/gallery_cutouts"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data/gallery_image_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    coverage_path = root / "data/processed/gallery_image_coverage.csv"
    previous = (pd.read_csv(coverage_path, dtype={"object_id": str}).fillna("")
                .set_index("object_id").to_dict("index")) if coverage_path.exists() else {}
    available, records, pending = {}, {}, []
    for row in selected.itertuples(index=False):
        paths = [root / f"data/raw/cutouts/{row.object_id}_VIS.fits"]
        for cache in (root / "data/raw/cutouts", root / "data/raw/second_cutouts", folder):
            paths.extend(sorted(cache.glob(f"{row.object_id}_VIS_*.fits")))
        for path in dict.fromkeys(paths):
            if not path.exists():
                continue
            try:
                stamp = reproject_stamp(path, row.right_ascension, row.declination)
                encoder.stretch_vis(stamp[None])
                available[row.object_id] = stamp
                records[row.object_id] = {"object_id": row.object_id, "status": "valid",
                                          "path": path.relative_to(root).as_posix(),
                                          "sha256": encoder.sha256(path)}
                break
            except (ValueError, OSError):
                continue
        else:
            old = previous.get(row.object_id, {})
            old_path = root / old.get("path", "")
            invalid_cached = (str(old.get("status", "")).startswith("unavailable: ValueError:")
                              and old_path.is_file() and encoder.sha256(old_path) == old.get("sha256"))
            denied = (old.get("status") in ("metadata_http_403", "not_requested_after_403")
                      or str(old.get("status", "")).startswith("unavailable: HTTPError: HTTP Error 403"))
            if invalid_cached or denied:
                records[row.object_id] = {"object_id": row.object_id, **old}
            else:
                pending.append(row)
    print(f"Gallery VIS previews: {len(available)} cached; {len(pending)} to acquire.", flush=True)
    products = []
    try:
        for start in range(0, len(pending), 50):
            ids = ",".join(str(int(row.object_id)) for row in pending[start:start + 50])
            query = ("SELECT m.object_id,p.file_name,p.file_path FROM catalogue.mer_catalogue AS m "
                     "JOIN q1.mosaic_product AS p ON "
                     "CONTAINS(CIRCLE(m.right_ascension,m.declination,0.0019444444444444444),p.fov)=1 "
                     f"WHERE m.object_id IN ({ids}) AND p.instrument_name='VIS'")
            params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query, "MAXREC": 10000}
            url = "https://eas.esac.esa.int/tap-server/tap/sync?" + urllib.parse.urlencode(params)
            path = folder / ("products_" + hashlib.sha256(query.encode()).hexdigest()[:12] + ".csv")
            if not path.exists():
                encoder._fetch(url, path)
            frame = pd.read_csv(path, dtype={"object_id": str})
            if len(frame) >= 10000:
                raise ValueError("Gallery metadata query reached MAXREC.")
            products.append(frame)
            manifest["files"][path.relative_to(root).as_posix()] = {
                "url": url, "query": query, "sha256": encoder.sha256(path), "bytes": path.stat().st_size}
    except urllib.error.HTTPError as error:
        # An access denial ends acquisition. Keep selected examples as placeholders.
        for row in pending:
            records[row.object_id] = {"object_id": row.object_id, "status": f"metadata_http_{error.code}"}
        _write_json(manifest_path, manifest)
        return available, records
    product_by_id = (pd.concat(products).sort_values(["object_id", "file_name"])
                     .drop_duplicates("object_id").set_index("object_id")) if products else pd.DataFrame()
    stop = threading.Event()

    def one(row):
        if stop.is_set():
            return row.object_id, None, {"object_id": row.object_id, "status": "not_requested_after_403"}, None
        if row.object_id not in product_by_id.index:
            return row.object_id, None, {"object_id": row.object_id, "status": "no_full_coverage_product"}, None
        product = product_by_id.loc[row.object_id]
        token = hashlib.sha256(product.file_name.encode()).hexdigest()[:10]
        path = folder / f"{row.object_id}_VIS_{token}.fits"
        params = {"TAPCLIENT": "ASTROQUERY", "FILEPATH": f"{product.file_path}/{product.file_name}",
                  "POS": f"CIRCLE,{float(row.right_ascension)},{float(row.declination)},{8 / 3600}"}
        url = "https://eas.esac.esa.int/sas-cutout/cutout"
        last = None
        for attempt in range(3):
            if stop.is_set():
                break
            try:
                if path.exists() and path.stat().st_size == 0:
                    path.unlink()
                if not path.exists():
                    request = urllib.request.Request(url, data=urllib.parse.urlencode(params).encode(),
                                                     headers={"User-Agent": "LensLab public-data exploration"})
                    with urllib.request.urlopen(request, timeout=50) as response:
                        content = response.read()
                    if not content:
                        raise OSError("Empty response from ESA cutout service.")
                    temporary = path.with_suffix(".part")
                    temporary.write_bytes(content)
                    temporary.replace(path)
                stamp = reproject_stamp(path, row.right_ascension, row.declination)
                encoder.stretch_vis(stamp[None])
                record = {"object_id": row.object_id, "status": "valid",
                          "path": path.relative_to(root).as_posix(), "sha256": encoder.sha256(path)}
                source = {"url": url, "parameters": params, "sha256": encoder.sha256(path),
                          "bytes": path.stat().st_size}
                return row.object_id, stamp, record, source
            except urllib.error.HTTPError as error:
                last = error
                if error.code == 403:
                    stop.set()
                    break
            except (ValueError, OSError) as error:
                last = error
                # A received, scientifically invalid image cannot be repaired by retrying.
                if path.exists() and path.stat().st_size:
                    break
            if attempt < 2:
                time.sleep(attempt + 1)
        record = {"object_id": row.object_id,
                  "status": f"unavailable: {type(last).__name__}: {last}" if last else "not_requested_after_403"}
        source = None
        if path.exists():
            record.update(path=path.relative_to(root).as_posix(), sha256=encoder.sha256(path))
            source = {"url": url, "parameters": params, "sha256": encoder.sha256(path), "bytes": path.stat().st_size}
        return row.object_id, None, record, source

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, row) for row in pending]
        for index, future in enumerate(as_completed(futures), 1):
            object_id, stamp, record, source = future.result()
            records[object_id] = record
            if stamp is not None:
                available[object_id] = stamp
            if source is not None:
                manifest["files"][record["path"]] = source
            _write_json(manifest_path, manifest)
            if index % 10 == 0 or index == len(pending):
                print(f"Gallery VIS downloads: {index}/{len(pending)}; {len(available)} valid total.", flush=True)
    _write_json(manifest_path, manifest)
    return available, records


def _thumbnail(stamp: np.ndarray) -> str:
    stretched = encoder.stretch_vis(stamp[None])[0]
    preview = Image.fromarray(np.uint8(np.flipud(stretched) * 255), mode="L")
    stream = io.BytesIO()
    preview.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def _font(size: int, bold: bool = False):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def render_static(root: Path, gallery: dict) -> Path:
    """Readable README preview: six anchors, all ten neighbours per anchor."""
    width, row_height, top, footer = 2400, 251, 180, 124
    canvas = Image.new("RGB", (width, top + 6 * row_height + footer), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    ink, muted, teal, gold = "#193341", "#5b6b73", "#087f8c", "#b77d16"
    draw.text((38, 27), "Which galaxies sit next to a candidate in Zoobot space?", font=_font(38, True), fill=ink)
    draw.text((38, 83), "One fixed pair per grade. Each row shows its anchor and the exact ten nearest neighbours.", font=_font(25), fill=muted)
    draw.text((38, 124), "A / B / C = published candidate grades     |     U = unlabelled reference, not a confirmed non-lens", font=_font(23), fill=muted)
    side, size, gap = 255, 178, 14
    for pair_index, pair in enumerate(gallery["pairs"]):
        for role_index, role in enumerate(("candidate", "reference")):
            row_index = pair_index * 2 + role_index
            y = top + row_index * row_height
            draw.rectangle((0, y, width, y + row_height), fill="#f0f6f7" if pair_index % 2 == 0 else "#ffffff")
            if role_index == 0:
                draw.line((35, y, width - 35, y), fill="#ccdadf", width=2)
            color = teal if role == "candidate" else muted
            count = pair[f"{role}_labelled_count"]
            draw.text((38, y + 34), f"PAIR {pair['grade']}", font=_font(21, True), fill=muted)
            draw.text((38, y + 76), "Candidate " + pair["grade"] if role == "candidate" else "Reference U", font=_font(24, True), fill=color)
            draw.text((38, y + 124), f"{count}/10", font=_font(41, True), fill=teal)
            draw.text((38, y + 179), "candidate neighbours", font=_font(18), fill=muted)
            objects = [{"object_id": pair[f"{role}_id"], "rank": 0}] + pair[f"{role}_neighbours"]
            for column, neighbour in enumerate(objects):
                item = gallery["items"][neighbour["object_id"]]
                x = side + column * (size + gap)
                label = "U" if item["label"] == "Unlabelled" else item["label"]
                text = f"ANCHOR · {label}" if column == 0 else f"#{neighbour['rank']} · {label}"
                badge = teal if item["role"] == "candidate" else muted
                draw.text((x, y + 12), text, font=_font(20, True), fill=badge)
                image_y = y + 43
                if item["image_available"]:
                    image = Image.open(io.BytesIO(base64.b64decode(item["image"].split(",", 1)[1]))).convert("RGB")
                    canvas.paste(image.resize((size, size), Image.Resampling.BILINEAR), (x, image_y))
                else:
                    draw.rectangle((x, image_y, x + size, image_y + size), fill="#dfe6e9")
                    draw.text((x + 15, image_y + 64), "No preview", font=_font(20), fill=muted)
                draw.rectangle((x, image_y, x + size, image_y + size), outline=gold if column == 0 else badge, width=3)
                # Identical angular field and orientation on every preview.
                if item["image_available"]:
                    bar_x, bar_y = x + 12, image_y + size - 18
                    draw.line((bar_x, bar_y, bar_x + int(size * 2 / 9.6), bar_y), fill="white", width=3)
            draw.line((side + size + gap // 2, y + 36, side + size + gap // 2, y + 230), fill="#a9b9c1", width=2)
    y = top + 6 * row_height + 17
    draw.text((38, y), "Examples were fixed with seed 20260918 before reading neighbour results; individual rows need not follow the overall average.", font=_font(20), fill=muted)
    draw.text((38, y + 32), "Search: all 2,252 objects, published 40D Zoobot coordinates; the anchor and its paired object are excluded. No rank is skipped.", font=_font(20), fill=muted)
    draw.text((38, y + 64), "Previews: Euclid Q1 VIS only, 9.6″ square; north up, east left; white bar = 2″. Authors’ Zoobot inputs used VIS + Y.", font=_font(20), fill=muted)
    destination = Path(root) / "figures/neighbour_gallery.png"
    canvas.save(destination)
    return destination


def run(root: Path = ROOT) -> dict:
    """Regenerate a fixed, auditable gallery without changing the analysis."""
    root = Path(root)
    sample, selection = select_pairs(root)
    graph_path = root / "results/neighbours.csv"
    graph = pd.read_csv(graph_path, dtype={"object_id": str, "neighbour_object_id": str})
    sample_index = sample.set_index("object_id")
    pairs, requested = [], set()
    for fixed in selection["pairs"]:
        pair = dict(fixed)
        for role in ("candidate", "reference"):
            anchor_id = pair[f"{role}_id"]
            rows = graph.loc[graph.object_id.eq(anchor_id)].sort_values("rank")
            if rows["rank"].tolist() != list(range(1, K + 1)) or rows.neighbour_object_id.duplicated().any():
                raise ValueError("Gallery needs the exact ten unique ranked neighbours of every anchor.")
            if sample_index.loc[rows.neighbour_object_id, "set_id"].eq(pair["set_id"]).any():
                raise ValueError("An anchor or its matched partner entered its neighbourhood.")
            pair[f"{role}_neighbours"] = [
                {"object_id": row.neighbour_object_id, "rank": int(row.rank), "distance": float(row.distance_40d)}
                for row in rows.itertuples(index=False)]
            pair[f"{role}_labelled_count"] = int(sample_index.loc[rows.neighbour_object_id, "role"].eq("candidate").sum())
            requested.add(anchor_id)
            requested.update(rows.neighbour_object_id)
        pairs.append(pair)
    selected = sample.loc[sample.object_id.isin(requested)].sort_values("object_id")
    available, records = _acquire(root, selected)
    coverage_path = root / "data/processed/gallery_image_coverage.csv"
    pd.DataFrame([records[oid] for oid in sorted(records)]).to_csv(coverage_path, index=False)
    ids = sorted(available)
    if ids:
        np.savez_compressed(root / "data/processed/gallery_stamps.npz", object_id=np.array(ids),
                            images=np.stack([available[oid] for oid in ids]), pixel_arcsec=0.1)
    items = {}
    for row in selected.itertuples(index=False):
        record = records[row.object_id]
        items[row.object_id] = {
            "label": row.grade if row.role == "candidate" else "Unlabelled", "role": row.role,
            "image": _thumbnail(available[row.object_id]) if row.object_id in available else None,
            "image_available": row.object_id in available, "ra": float(row.right_ascension), "dec": float(row.declination),
            "image_status": record["status"], "image_path": record.get("path"), "image_sha256": record.get("sha256"),
        }
    gallery = {"method": {
        "seed": SEED, "selection_rule": selection["selection_rule"],
        "sample_sha256": selection["sample_sha256"], "graph_sha256": encoder.sha256(graph_path),
        "k": K, "corpus_objects": len(sample), "preview_objects": len(items), "available_previews": len(available),
        "pixel_arcsec": 0.1, "stamp_size": 96, "field_arcsec": 9.6,
        "orientation": "North up; east left", "scale_bar_arcsec": 2,
        "neighbour_rule": "Exact ranks 1–10 in the final graph; Euclidean distance in the published 40D Zoobot coordinates; exclude the whole matched pair.",
        "missing_image_rule": "Keep the selected object and its true rank; show a placeholder. No replacement or dropped rank.",
        "preview_processing": "VIS FITS reprojected to a 96×96 TAN grid at 0.1 arcsec/pixel; border background, 99.5th-percentile scaling and asinh stretch. Preview only, not the authors' VIS+Y Zoobot input.",
        "provenance": {"coverage": "data/processed/gallery_image_coverage.csv", "new_images": "data/gallery_image_manifest.json",
                       "cached_images": ["data/manifest.json", "data/expanded_manifest.json", "data/second_image_manifest.json"]},
    }, "pairs": pairs, "items": items}
    _write_json(root / "results/neighbour_gallery.json", gallery)
    render_static(root, gallery)
    print(f"Gallery: {len(pairs)} fixed pairs, {len(items)} unique objects, {len(available)} VIS previews.", flush=True)
    for pair in pairs:
        print(f"  Grade {pair['grade']}: candidate {pair['candidate_labelled_count']}/10; reference {pair['reference_labelled_count']}/10", flush=True)
    return gallery


if __name__ == "__main__":
    run()
