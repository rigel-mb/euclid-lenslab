"""Reuse the published Euclid Zoobot image representations.

The 40 PCA coordinates were released by the morphology-catalogue authors.
This module downloads and cross-matches them; it does not train a network,
recreate their image processing, or read the lens grades. Our FITS stamps are
used for visual inspection and the separate frozen DINOv2 comparison.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RECORD = "15106473"
RECORD_URL = f"https://zenodo.org/records/{RECORD}"
FILENAME = "representations_pca_40.parquet"
FILE_URL = f"https://zenodo.org/api/records/{RECORD}/files/{FILENAME}/content"
SOURCE_SHA256 = "e3c43d3e2796a0913bacca6ff8b0f7b0be70bceeccc3098f3a5ffde16732508e"
SOURCE_MD5 = "a43e3edfe6df3ef0ad9bd10da67d3dae"
SOURCE_BYTES = 138553084
FEATURE_COLUMNS = [f"feat_pca_{i}" for i in range(40)]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _fetch(url: str, destination: Path) -> None:
    """Stream to a temporary file so an interrupted download is never cached."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "LensLab public-data exploration"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _object_ids(frame: pd.DataFrame) -> pd.Series:
    # Official identifiers are Q1_R1_<tile>_<signed MER object_id>.
    valid = frame["id_str"].str.match(r"^Q1_R1_\d+_-?\d+$")
    if not valid.all():
        raise ValueError("Unexpected published identifier format.")
    ids = frame["id_str"].str.rsplit("_", n=1).str[-1].astype(str)
    if ids.duplicated().any():
        raise ValueError("Published MER identifiers are not unique.")
    return ids


def acquire(root: Path = ROOT) -> Path:
    """Cache the fixed public release and expose its eligible MER identifiers."""
    folder = Path(root) / "data/raw/zoobot"
    source = folder / FILENAME
    if not source.exists():
        _fetch(FILE_URL, source)
    if source.stat().st_size != SOURCE_BYTES or sha256(source) != SOURCE_SHA256:
        raise ValueError("Published representation checksum mismatch; restore the fixed source file.")
    record = folder / f"zenodo_record_{RECORD}.json"
    if not record.exists():
        _fetch(f"https://zenodo.org/api/records/{RECORD}", record)
    metadata = json.loads(record.read_text())
    remote_file = next(item for item in metadata["files"] if item["key"] == FILENAME)
    if remote_file["checksum"] != "md5:" + SOURCE_MD5:
        raise ValueError("Zenodo record checksum differs from the pinned source.")
    identifiers = pd.read_parquet(source, columns=["id_str"])
    pd.DataFrame({"object_id": _object_ids(identifiers)}).to_csv(folder / "available_ids.csv", index=False)
    return source


def load_published(root: Path = ROOT) -> pd.DataFrame:
    """Read only image-derived features, indexed by exact signed MER identifier."""
    source = acquire(root)
    frame = pd.read_parquet(source, columns=["id_str"] + FEATURE_COLUMNS)
    ids = _object_ids(frame)
    features = frame[FEATURE_COLUMNS].copy()
    features.index = pd.Index(ids, name="object_id")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Non-finite published representation.")
    return features


def stretch_vis(images: np.ndarray) -> np.ndarray:
    """Specified VIS stretch for image inspection and DINOv2 inputs."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (96, 96):
        raise ValueError("Expected images with shape (N, 96, 96).")
    if not np.isfinite(images).all():
        raise ValueError("Stamp quality filtering must remove non-finite images first.")
    border = np.ones((96, 96), dtype=bool)
    border[12:-12, 12:-12] = False
    background = np.median(images[:, border], axis=1)
    positive = np.maximum(images - background[:, None, None], 0)
    scale = np.percentile(positive, 99.5, axis=(1, 2))
    if np.any(scale <= 0):
        raise ValueError("A stamp has no positive signal after background subtraction.")
    clipped = np.minimum(positive / scale[:, None, None], 1)
    return (np.arcsinh(10 * clipped) / np.arcsinh(10)).astype(np.float32)
