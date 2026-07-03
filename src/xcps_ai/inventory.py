from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from xcps_ai.config import RUNS, RunConfig
from xcps_ai.hdf5 import data_file, parse_roi_key, roi_keys


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(item) for item in shape)


def _dataset_shape(group: h5py.Group, name: str) -> str | None:
    return _shape_text(group[name].shape) if name in group else None


def _dataset_dtype(group: h5py.Group, name: str) -> str | None:
    return str(group[name].dtype) if name in group else None


def _delay_summary(group: h5py.Group) -> dict[str, float | int | None]:
    if "delay" not in group:
        return {
            "delay_len": None,
            "delay_start_s": None,
            "delay_stop_s": None,
            "delay_step_median_s": None,
        }
    delay = np.asarray(group["delay"][()])
    summary: dict[str, float | int | None] = {
        "delay_len": int(delay.size),
        "delay_start_s": float(delay[0]) if delay.size else None,
        "delay_stop_s": float(delay[-1]) if delay.size else None,
        "delay_step_median_s": None,
    }
    if delay.size > 1:
        summary["delay_step_median_s"] = float(np.nanmedian(np.diff(delay)))
    return summary


def _params_summary(group: h5py.Group) -> dict[str, float]:
    if "params" not in group:
        return {}
    params = np.asarray(group["params"][()])
    return {f"param_{i}_{j}": float(value) for (i, j), value in np.ndenumerate(params)}


def _roi_pixels(group: h5py.Group) -> int | None:
    if "roi_" not in group:
        return None
    return int(np.asarray(group["roi_"][()]).sum())


def inventory_rows(data_dir: Path, run: RunConfig) -> list[dict[str, Any]]:
    path = data_file(data_dir, run)
    if not path.exists():
        return [
            {
                "uid": run.uid,
                "temperature_k": run.temperature_k,
                "wait_time": run.wait_time,
                "file": str(path),
                "file_exists": False,
            }
        ]

    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for roi_key in roi_keys(handle):
            group = handle[roi_key]
            row: dict[str, Any] = {
                "uid": run.uid,
                "temperature_k": run.temperature_k,
                "wait_time": run.wait_time,
                "file": str(path),
                "file_exists": True,
                "file_size_mb": path.stat().st_size / 1024 / 1024,
                "roi": parse_roi_key(roi_key),
                "roi_key": roi_key,
                "q_index": parse_roi_key(roi_key),
                "roi_pixels": _roi_pixels(group),
                "ttcf_shape": _dataset_shape(group, "ttcf"),
                "ttcf_dtype": _dataset_dtype(group, "ttcf"),
                "g2_shape": _dataset_shape(group, "g2"),
                "g2e_shape": _dataset_shape(group, "g2e"),
                "roi_mask_shape": _dataset_shape(group, "roi_"),
                "bp_shape": _dataset_shape(group, "bp"),
            }
            row.update(_delay_summary(group))
            row.update(_params_summary(group))
            rows.append(row)
    return rows


def build_inventory(data_dir: Path, runs: tuple[RunConfig, ...] = RUNS) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.extend(inventory_rows(data_dir, run))
    return pd.DataFrame(rows)
