from __future__ import annotations

from pathlib import Path

import h5py

from xcps_ai.config import RunConfig


def data_file(data_dir: Path, run: RunConfig) -> Path:
    return data_dir / run.filename


def parse_roi_key(key: str) -> int:
    prefix, value = key.split()
    if prefix != "ROI":
        raise ValueError(f"Expected ROI key, got {key!r}")
    return int(value)


def roi_keys(handle: h5py.File) -> list[str]:
    keys = [key for key in handle.keys() if key.startswith("ROI ")]
    return sorted(keys, key=parse_roi_key)


def available_rois(path: Path) -> list[int]:
    with h5py.File(path, "r") as handle:
        return [parse_roi_key(key) for key in roi_keys(handle)]
