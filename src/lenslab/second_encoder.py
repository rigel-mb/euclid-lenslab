"""A frozen DINOv2 comparison on a fixed, modest image subset.

The released Zoobot coordinates and these new DINOv2 features are separate
spaces. DINOv2 sees our VIS-only stamps, not the authors' VIS+Y inputs. This is
a sensitivity study of the representation pipeline, not a lens classifier or
a controlled comparison of network architectures.
"""
from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from . import encoder

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
WEIGHTS_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
MODEL_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
MODEL_SHA256 = {
    "model.safetensors": WEIGHTS_SHA256,
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
}
SUBSET_SEED = 20260915
REDISTRIBUTION_SOURCES = {
    "license_url": "https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE",
    "license_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    "license_sha256": "600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7",
    "model_card_url": f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/README.md",
    "model_card_sha256": "4c20dca454a8e5c670e8de5c7e6040f512aeca5438516f7623eedc4e3b00599c",
}


def select_subset(root: Path = ROOT, *, sets_per_grade: int = 20) -> pd.DataFrame:
    """Freeze a seeded set sample before images or DINOv2 outputs are examined."""
    root = Path(root)
    expanded_path = root / "data/processed/expanded_sample.csv"
    expanded = pd.read_csv(expanded_path, dtype={"object_id": str, "set_id": str})
    candidates = expanded.loc[expanded.role.eq("candidate")].sort_values("object_id")
    selected_sets = []
    for grade in ["A", "B", "C"]:
        eligible = candidates.loc[candidates.grade.eq(grade)]
        selected_sets.extend(eligible.sample(n=min(sets_per_grade, len(eligible)),
                                             random_state=SUBSET_SEED).set_id)
    subset = expanded.loc[expanded.set_id.isin(selected_sets)].copy()
    subset = subset.sort_values("object_id").reset_index(drop=True)
    destination = root / "data/processed/second_sample.csv"
    if destination.exists():
        previous = pd.read_csv(destination, dtype={"object_id": str, "set_id": str})
        if previous.object_id.tolist() != subset.object_id.tolist():
            raise ValueError("The frozen second-encoder IDs changed; do not select by availability.")
    else:
        subset.to_csv(destination, index=False)
    selection = {
        "seed": SUBSET_SEED, "requested_sets_per_grade": sets_per_grade,
        "selected_sets": subset.set_id.nunique(), "selected_objects": len(subset),
        "candidate_grades": subset.loc[subset.role.eq("candidate"), "grade"].value_counts().to_dict(),
        "references_per_candidate": int(len(subset) / subset.set_id.nunique()) - 1,
        "rule": "Sort candidate object IDs, then take up to 20 sets per A/B/C grade using seed 20260915; retain every matched reference from the primary cohort. Freeze before image acquisition and second-encoder outputs.",
        "image_failure_rule": "Exclude the entire incomplete set from the comparison; no replacement set or relaxed image-quality criterion.",
        "expanded_sample_sha256": _hash(expanded_path), "subset_sha256": _hash(destination),
    }
    (root / "results/second_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return subset


def acquire_images(root: Path = ROOT) -> dict:
    """Acquire only the fixed comparison subset, with no replacement IDs.

    Existing stamps are reused. New metadata require full geometric coverage
    of the 9.6-arcsec square; products are chosen by filename, never appearance.
    """
    from .sampling import reproject_stamp

    root = Path(root)
    selected = pd.read_csv(root / "data/processed/second_sample.csv",
                           dtype={"object_id": str, "set_id": str})
    all_rows = selected.copy()
    all_rows = all_rows.drop_duplicates("object_id").sort_values("object_id")
    folder = root / "data/raw/second_cutouts"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data/second_image_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    available, records, pending = {}, [], []
    for _, row in all_rows.iterrows():
        candidates = [root / f"data/raw/cutouts/{row.object_id}_VIS.fits"]
        candidates += sorted((root / "data/raw/cutouts").glob(f"{row.object_id}_VIS_*.fits"))
        candidates += sorted(folder.glob(f"{row.object_id}_VIS_*.fits"))
        for path in candidates:
            if path.exists():
                try:
                    image = reproject_stamp(path, row.right_ascension, row.declination)
                    encoder.stretch_vis(image[None])
                    available[row.object_id] = image
                    records.append({"object_id": row.object_id, "status": "valid",
                                    "path": path.relative_to(root).as_posix(), "sha256": _hash(path)})
                    break
                except (ValueError, OSError):
                    pass
        else:
            pending.append(row)
    products = []
    for start in range(0, len(pending), 50):
        ids = ",".join(str(int(row.object_id)) for row in pending[start:start + 50])
        query = ("SELECT m.object_id,p.file_name,p.file_path FROM catalogue.mer_catalogue AS m "
                 "JOIN q1.mosaic_product AS p ON "
                 "CONTAINS(CIRCLE(m.right_ascension,m.declination,0.0019444444444444444),p.fov)=1 "
                 f"WHERE m.object_id IN ({ids}) AND p.instrument_name='VIS'")
        params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query, "MAXREC": 10000}
        url = "https://eas.esac.esa.int/tap-server/tap/sync?" + urllib.parse.urlencode(params)
        name = "products_" + hashlib.sha256(query.encode()).hexdigest()[:12] + ".csv"
        path = folder / name
        if not path.exists():
            encoder._fetch(url, path)
        frame = pd.read_csv(path, dtype={"object_id": str})
        if len(frame) >= 10000:
            raise ValueError("Second-encoder image product query reached MAXREC.")
        products.append(frame)
        manifest["files"][path.relative_to(root).as_posix()] = {
            "url": url, "query": query, "sha256": _hash(path), "bytes": path.stat().st_size}
        print(f"Second-encoder image metadata: {min(start + 50, len(pending))}/{len(pending)}", flush=True)
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
            try:
                if path.exists() and path.stat().st_size == 0:
                    path.unlink()  # Retry an empty HTTP body, not an astronomical image.
                if not path.exists():
                    request = urllib.request.Request(url, data=urllib.parse.urlencode(params).encode(),
                                                     headers={"User-Agent": "LensLab public-data exploration"})
                    with urllib.request.urlopen(request, timeout=50) as response:
                        content = response.read()
                    if not content:
                        raise ValueError("Empty response from ESA cutout service.")
                    temporary = path.with_suffix(".part")
                    temporary.write_bytes(content)
                    temporary.replace(path)
                image = reproject_stamp(path, row.right_ascension, row.declination)
                encoder.stretch_vis(image[None])
                record = {"object_id": row.object_id, "status": "valid",
                          "path": path.relative_to(root).as_posix(), "sha256": _hash(path)}
                source = {"url": url, "parameters": params, "sha256": _hash(path),
                          "bytes": path.stat().st_size}
                return row.object_id, image, record, source
            except Exception as error:
                last = error
                if attempt < 2:
                    time.sleep(attempt + 1)
        if "403" in str(last):
            stop.set()
        record = {"object_id": row.object_id, "status": f"unavailable: {type(last).__name__}: {last}"}
        source = None
        if path.exists():
            record.update(path=path.relative_to(root).as_posix(), sha256=_hash(path))
            source = {"url": url, "parameters": params, "sha256": _hash(path), "bytes": path.stat().st_size}
        return row.object_id, None, record, source

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, row) for row in pending]
        for index, future in enumerate(as_completed(futures)):
            identifier, image, record, source = future.result()
            records.append(record)
            if image is not None:
                available[identifier] = image
            if source is not None:
                manifest["files"][record["path"]] = source
            if (index + 1) % 20 == 0:
                print(f"Second-encoder VIS downloads: {index + 1}/{len(pending)}; valid total={len(available)}", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    selected_ids = [identifier for identifier in selected.object_id if identifier in available]
    if not selected_ids:
        raise ValueError("No valid VIS stamps were acquired for the frozen comparison subset.")
    np.savez_compressed(root / "data/processed/second_stamps.npz", object_id=np.array(selected_ids),
                        images=np.stack([available[identifier] for identifier in selected_ids]), pixel_arcsec=0.1)
    pd.DataFrame(records).to_csv(root / "data/processed/second_image_coverage.csv", index=False)
    summary = {"requested_objects": len(all_rows), "comparison_requested_objects": len(selected),
               "available_objects": len(available), "comparison_available_objects": len(selected_ids),
               "new_requested_objects": len(pending), "cached_objects": len(all_rows) - len(pending)}
    (root / "results/second_image_acquisition.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def acquire_model(root: Path = ROOT) -> Path:
    """Download the pinned official release; never execute remote model code."""
    folder = Path(root) / "data/raw/second_encoder"
    folder.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        path = folder / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                urllib.request.urlretrieve(
                    f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}",
                    temporary,
                )
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    if any(_hash(folder / name) != MODEL_SHA256[name] for name in MODEL_FILES):
        raise ValueError("DINOv2 weights/configuration differ from the pinned official release.")
    for prefix, name in [("license", "LICENSE"), ("model_card", "MODEL_CARD.txt")]:
        path = folder / name
        if not path.exists():
            encoder._fetch(REDISTRIBUTION_SOURCES[prefix + "_url"], path)
        if _hash(path) != REDISTRIBUTION_SOURCES[prefix + "_sha256"]:
            raise ValueError(f"DINOv2 {prefix} differs from the pinned source.")
    (folder / "source.json").write_text(json.dumps(REDISTRIBUTION_SOURCES, indent=2) + "\n")
    return folder


def _load_model(root: Path, device: str, threads: int):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    if device == "cpu":
        torch.set_num_threads(threads)
    folder = acquire_model(root)
    processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True,
                                                  use_fast=False)
    model = AutoModel.from_pretrained(folder, local_files_only=True,
                                     use_safetensors=True, trust_remote_code=False)
    model = model.eval().requires_grad_(False).to(device)
    return processor, model


def _rgb_images(images: np.ndarray):
    """North-up, deterministic VIS display stretch; identical grey RGB channels."""
    from PIL import Image

    return [Image.fromarray(np.uint8(np.rint(np.flipud(image) * 255))).convert("RGB")
            for image in encoder.stretch_vis(images)]


def _cached_features(root: Path) -> dict | None:
    """Accept existing outputs only when their inputs, identity and bytes agree."""
    path = root / "results/second_encoder_summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    paths = {
        "sample_sha256": root / "data/processed/second_sample.csv",
        "stamp_sha256": root / "data/processed/second_stamps.npz",
        "output_sha256": root / "data/processed/second_embeddings.npz",
    }
    complete_path = root / "data/processed/second_complete_sample.csv"
    if not complete_path.exists() or not all(p.exists() for p in paths.values()):
        return None
    if summary.get("revision") != REVISION or summary.get("weights_sha256") != WEIGHTS_SHA256:
        return None
    if any(_hash(p) != summary.get(key) for key, p in paths.items()):
        return None
    selected = pd.read_csv(paths["sample_sha256"], dtype={"object_id": str, "set_id": str})
    complete = pd.read_csv(complete_path, dtype={"object_id": str, "set_id": str})
    expected = selected.loc[~selected.set_id.isin(summary["excluded_set_ids"])].reset_index(drop=True)
    identity = ["object_id", "set_id", "role", "grade"]
    if not expected[identity].equals(complete[identity].reset_index(drop=True)):
        return None
    if len(complete) != summary["complete_objects"] or complete.set_id.nunique() != summary["complete_sets"]:
        return None
    with np.load(paths["output_sha256"], allow_pickle=False) as values:
        ids = values["object_id"].astype(str)
        if ids.tolist() != complete.object_id.tolist() or ids.tolist() != sorted(ids):
            return None
        raw, unit, zoobot = values["dino_raw"], values["dino_l2"], values["zoobot"]
        if raw.shape != (len(ids), 384) or not np.isfinite(raw).all():
            return None
        if not np.allclose(unit, raw / np.linalg.norm(raw, axis=1, keepdims=True), rtol=1e-6, atol=1e-7):
            return None
        published = encoder.load_published(root).loc[ids].to_numpy(copy=True)
        if not np.array_equal(zoobot, published):
            return None
    return summary


def run(root: Path = ROOT, *, device: str = "auto", batch_size: int = 4,
        threads: int = 2, reuse_cache: bool = True) -> dict:
    """Extract CLS features for the frozen IDs; retain complete sets only.

    second_sample.csv is fixed before image acquisition. Its set_id groups
    contain one candidate and a fixed number of reference objects. Missing or invalid VIS
    stamps remove a complete set from this image comparison only, never from
    the primary published-representation analysis.
    """
    root = Path(root)
    if reuse_cache:
        cached = _cached_features(root)
        if cached is not None:
            print(f"Verified cached DINOv2 features for {cached['complete_objects']} objects; "
                  f"original {cached['execution']['device']} inference: "
                  f"{cached['execution']['inference_seconds']:.1f}s.")
            return cached
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose device='auto', 'cpu' or 'cuda'.")
    sample_path = root / "data/processed/second_sample.csv"
    stamps_path = root / "data/processed/second_stamps.npz"
    selected = pd.read_csv(sample_path, dtype={"object_id": str, "set_id": str})
    if selected.object_id.duplicated().any():
        raise ValueError("The second-encoder sample must not reuse objects.")
    group_sizes = selected.groupby("set_id").size()
    if group_sizes.nunique() != 1 or group_sizes.min() < 2:
        raise ValueError("Expected equally sized candidate-reference sets.")
    for _, group in selected.groupby("set_id"):
        if group.role.eq("candidate").sum() != 1 or not group.role.isin(["candidate", "reference"]).all():
            raise ValueError("Expected one candidate and the remaining reference objects per set.")
    with np.load(stamps_path, allow_pickle=False) as stamps:
        ids = stamps["object_id"].astype(str)
        images = stamps["images"]
    original_stamp_ids = set(ids)
    if len(ids) != len(images) or len(set(ids)) != len(ids):
        raise ValueError("Image identifiers must be unique and aligned.")
    by_id, invalid = {}, []
    for identifier, image in zip(ids, images):
        try:
            encoder.stretch_vis(image[None])
            by_id[identifier] = image
        except ValueError:
            invalid.append(identifier)
    complete = [set_id for set_id, group in selected.groupby("set_id")
                if group.object_id.isin(by_id).all()]
    sample = selected.loc[selected.set_id.isin(complete)].copy().reset_index(drop=True)
    if sample.set_id.nunique() < 10:
        raise ValueError("Too few complete image sets for a useful comparison.")
    ids = sample.object_id.to_numpy(dtype=str)
    images = np.stack([by_id[identifier] for identifier in ids])
    processor, model = _load_model(root, device, threads)
    rgb = _rgb_images(images)
    outputs = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(rgb), batch_size):
            inputs = processor(images=rgb[i:i + batch_size], return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            result = model(**inputs).last_hidden_state[:, 0]
            outputs.append(result.cpu().numpy())
            if (i // batch_size + 1) % 20 == 0:
                print(f"DINOv2 frozen inference: {min(i + batch_size, len(ids))}/{len(ids)}", flush=True)
    elapsed = time.perf_counter() - start
    raw = np.concatenate(outputs).astype(np.float32)
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    published = encoder.load_published(root).loc[ids].to_numpy(copy=True)
    if raw.shape != (len(ids), 384) or not np.isfinite(raw).all():
        raise ValueError("Unexpected DINOv2 output shape or non-finite values.")
    output_path = root / "data/processed/second_embeddings.npz"
    np.savez_compressed(output_path, object_id=ids, dino_raw=raw, dino_l2=unit,
                        zoobot=published)
    sample.to_csv(root / "data/processed/second_complete_sample.csv", index=False)
    model_folder = root / "data/raw/second_encoder"
    image_coverage_path = root / "data/processed/second_image_coverage.csv"
    unavailable_reasons = {}
    if image_coverage_path.exists():
        coverage = pd.read_csv(image_coverage_path, dtype={"object_id": str})
        unavailable_rows = coverage.loc[coverage.object_id.isin(selected.object_id) & coverage.status.ne("valid")]
        unavailable_reasons = dict(zip(unavailable_rows.object_id, unavailable_rows.status))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "revision": REVISION,
        "checkpoint_license": "Apache-2.0 (official Hugging Face repository)",
        "redistribution_sources": REDISTRIBUTION_SOURCES,
        "model_url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "paper_url": "https://arxiv.org/abs/2304.07193",
        "architecture": "DINOv2 ViT-S/14: 12 blocks, 6 heads, hidden width 384; no task head.",
        "upstream_learning": "Self-supervised pretraining/distillation on general images, by the model authors.",
        "local_learning": "Frozen feature extraction only: eval mode, no gradient, no fine-tuning, no grades supplied to the encoder.",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "weights_sha256": WEIGHTS_SHA256,
        "model_file_sha256": {name: _hash(model_folder / name) for name in MODEL_FILES},
        "selected_objects": len(selected), "selected_sets": selected.set_id.nunique(),
        "references_per_candidate": int(group_sizes.iloc[0]) - 1,
        "complete_objects": len(sample), "complete_sets": len(complete),
        "selected_candidate_grades": selected.loc[selected.role.eq("candidate"), "grade"].value_counts().to_dict(),
        "complete_candidate_grades": sample.loc[sample.role.eq("candidate"), "grade"].value_counts().to_dict(),
        "missing_stamp_ids": sorted(set(selected.object_id) - original_stamp_ids),
        "invalid_stamp_ids": sorted(invalid),
        "unavailable_image_reasons": unavailable_reasons,
        "excluded_set_ids": sorted(set(selected.set_id) - set(complete)),
        "input": {
            "stamp": "96x96 VIS, 0.1 arcsec/pixel, north up and east left.",
            "stretch": "Subtract the median outer 12-pixel border; positive residuals; clip at the per-image 99.5th percentile; arcsinh(10*x)/arcsinh(10).",
            "rgb": "The stretched VIS values are rounded to uint8 and replicated into all three channels; no colour information.",
            "official_processor": "Slow BitImageProcessor from the pinned release: bicubic resize shortest edge to 256; centre crop 224x224; rescale 1/255; ImageNet mean/std.",
            "effective_angular_crop_arcsec": 8.4,
            "augmentation": "None; no random crops, rotations or test-time averaging.",
        },
        "features": {
            "dino": "384-dimensional final layer-normalized CLS token; L2-normalized per object for Euclidean neighbours (equivalent ordering to cosine similarity). Raw features are retained too.",
            "zoobot": "Original 40 published PCA coordinates, unchanged and joined by exact MER object_id.",
            "space_handling": "Neighbours are recomputed independently inside each space on exactly the same objects. Coordinates from different encoders are never concatenated or mixed.",
        },
        "limitations": [
            "This small subset checks representation sensitivity, not population prevalence or generalization.",
            "DINOv2 sees VIS-only fixed-centre crops while the published Zoobot pipeline uses different VIS+Y inputs and upstream preprocessing.",
            "Distance normalization and dimensionality also differ, so any change is a pipeline comparison, not an isolated architecture effect.",
            "Candidate grades are incomplete, selected annotations; reference objects are unlabelled, not confirmed non-lenses.",
        ],
        "execution": {"device": device, "cpu_threads": threads if device == "cpu" else None,
                      "batch_size": batch_size, "inference_seconds": elapsed,
                      "versions": {name: importlib.metadata.version(name)
                                   for name in ["torch", "transformers", "Pillow", "numpy"]}},
        "sample_sha256": _hash(sample_path), "stamp_sha256": _hash(stamps_path),
        "output_sha256": _hash(output_path),
    }
    path = root / "results/second_encoder_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def compare(root: Path = ROOT) -> dict:
    """Use identical objects and the same neighbour audit for both encoders."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from threadpoolctl import threadpool_limits
    from .neighbours import neighbour_graph, audit_neighbours
    from .plots import style, NAVY, TEAL, MUTED, PALE

    root = Path(root)
    sample = pd.read_csv(root / "data/processed/second_complete_sample.csv",
                         dtype={"object_id": str, "set_id": str}).reset_index(drop=True)
    with np.load(root / "data/processed/second_embeddings.npz", allow_pickle=False) as values:
        ids = values["object_id"].astype(str)
        spaces = {"zoobot": values["zoobot"], "dino": values["dino_l2"]}
    if ids.tolist() != sample.object_id.tolist() or ids.tolist() != sorted(ids):
        raise ValueError("Comparison features and stable sorted object IDs are misaligned.")
    audits, table, neighbour_rows = {}, sample.copy(), []
    with threadpool_limits(limits=2):
        for name, features in spaces.items():
            graph = neighbour_graph(features, sample.set_id.to_numpy(), k=10)
            audit = audit_neighbours(sample, graph, permutations=1000)
            audits[name] = {key: value for key, value in audit.items() if key not in {"fraction", "null"}}
            table[f"{name}_candidate_neighbour_fraction"] = audit["fraction"]
            for i, neighbours in enumerate(graph):
                neighbour_rows.extend({"representation": name, "object_id": ids[i],
                                       "neighbour_rank": rank + 1, "neighbour_id": ids[j]}
                                      for rank, j in enumerate(neighbours))
    table.to_csv(root / "results/second_embedding_table.csv", index=False)
    pd.DataFrame(neighbour_rows).to_csv(root / "results/second_neighbours.csv", index=False)
    summary = {
        "objects": len(sample), "matched_sets": int(sample.set_id.nunique()),
        "candidate_grades": sample.loc[sample.role.eq("candidate"), "grade"].value_counts().to_dict(),
        "candidate_fraction": float(sample.role.eq("candidate").mean()),
        "zoobot": audits["zoobot"], "dino": audits["dino"],
        "design": "Exactly the same complete pairs and audit as the main analysis; ten neighbours excluding the query and its partner; 1000 within-pair role permutations.",
        "interpretation": "Zoobot shows a stronger candidate/reference neighbourhood contrast than DINOv2 on this fixed subset. Its representation pipeline is more aligned with the existing candidate labels in this protocol, consistent with the potential value of galaxy-morphology pretraining; this is not a lens-detection accuracy comparison.",
        "comparison_limits": "DINOv2 VIS-only fixed crops/L2-normalized 384D differ from the published Zoobot VIS+Y/preprocessing/PCA40/raw Euclidean space. Network, input and geometry effects cannot be disentangled here. Prior candidate selection used related image models, so this is not independent validation.",
        "subset_limit": "The secondary subset deliberately samples A/B/C equally, unlike the C-dominated main cohort. Compare encoders on this same subset, not against main-cohort percentages.",
        "embeddings_sha256": _hash(root / "data/processed/second_embeddings.npz"),
        "complete_sample_sha256": _hash(root / "data/processed/second_complete_sample.csv"),
    }
    (root / "results/second_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    style()
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    x = np.arange(2)
    names = ["zoobot", "dino"]
    candidates = [audits[name]["candidate_neighbour_fraction"] for name in names]
    references = [audits[name]["reference_neighbour_fraction"] for name in names]
    ax.bar(x - .19, candidates, width=.34, color=TEAL, label="Around candidates", zorder=3)
    ax.bar(x + .19, references, width=.34, color="#A8B5C3", label="Around references", zorder=3)
    for i, (a, b) in enumerate(zip(candidates, references)):
        ax.text(i-.19, a+.025, f"{a:.1%}", ha="center", fontsize=16, color=NAVY, weight="bold")
        ax.text(i+.19, b+.025, f"{b:.1%}", ha="center", fontsize=16, color=NAVY, weight="bold")
    ax.axhline(summary["candidate_fraction"], color=MUTED, linestyle=":", linewidth=1.4,
               label="50% in the matched subset", zorder=2)
    ax.set_xticks(x, [f"Zoobot\nContrast: {100*audits['zoobot']['difference']:.1f} pp",
                     f"DINOv2\nContrast: {100*audits['dino']['difference']:.1f} pp"])
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Candidates among the 10 image neighbours")
    ax.grid(axis="y", color=PALE, alpha=.5, zorder=0)
    fig.suptitle("Zoobot gives a stronger neighbourhood contrast", x=.08, y=.98,
                 ha="left", color=NAVY, fontsize=18, weight="bold")
    fig.text(.08, .895, f"{len(sample)} identical objects · {sample.set_id.nunique()} matched pairs · no fine-tuning",
             fontsize=12, color=NAVY)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(.52, .12), fontsize=9)
    fig.text(.08, .04, "The pipelines differ in bands, crops and preprocessing. This supports a descriptive comparison, not a causal claim\nabout specialization or a measurement of lens-detection accuracy.", fontsize=9, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=.1, right=.975, top=.79, bottom=.27)
    fig.savefig(root / "figures/second_comparison.png", dpi=190, facecolor="white")
    plt.close(fig)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--recompute", action="store_true", help="Re-extract frozen DINOv2 features instead of reusing verified outputs.")
    args = parser.parse_args()
    result = run(device=args.device, batch_size=args.batch_size, reuse_cache=not args.recompute)
    compare()
    print(json.dumps(result["execution"], indent=2))
