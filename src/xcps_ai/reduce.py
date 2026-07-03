from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from xcps_ai.config import DEFAULT_CUTOFF, RunConfig
from xcps_ai.hdf5 import data_file


@dataclass(frozen=True)
class ReductionOptions:
    cutoff: int = DEFAULT_CUTOFF
    max_lag: int | None = None


def diagonal_average(ttcf: np.ndarray, max_lag: int | None = None) -> pd.DataFrame:
    """Average upper TTCF diagonals to compute one-time g2(tau)."""
    matrix = np.asarray(ttcf)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"TTCF must be square, got {matrix.shape}")

    n_frames = matrix.shape[0]
    if max_lag is None:
        max_lag = n_frames - 1
    max_lag = min(max(0, int(max_lag)), n_frames - 1)

    rows = []
    for lag in range(max_lag + 1):
        values = np.diagonal(matrix, offset=lag)
        values = values[np.isfinite(values)]
        if values.size:
            mean = float(values.mean())
            std = float(values.std())
            sem = float(std / np.sqrt(values.size))
        else:
            mean = np.nan
            std = np.nan
            sem = np.nan
        rows.append(
            {
                "lag": lag,
                "g2": mean,
                "g2e": sem,
                "diagonal_std": std,
                "n_pairs": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def default_max_lag(n_frames_after_crop: int, cutoff: int) -> int:
    if cutoff <= 0:
        return n_frames_after_crop - 1
    return max(0, n_frames_after_crop - cutoff - 1)


def reduce_roi(
    data_dir: Path,
    run: RunConfig,
    roi: int,
    options: ReductionOptions,
) -> pd.DataFrame:
    path = data_file(data_dir, run)
    roi_key = f"ROI {roi:d}"
    with h5py.File(path, "r") as handle:
        group = handle[roi_key]
        dataset = group["ttcf"]
        if dataset.shape[0] != dataset.shape[1]:
            raise ValueError(f"{path}:{roi_key}/ttcf is not square: {dataset.shape}")
        if run.wait_time >= dataset.shape[0]:
            raise ValueError(f"wait_time {run.wait_time} exceeds TTCF length {dataset.shape[0]}")

        cropped_n = dataset.shape[0] - run.wait_time
        max_lag = options.max_lag
        if max_lag is None:
            max_lag = default_max_lag(cropped_n, options.cutoff)
        else:
            max_lag = min(max_lag, cropped_n - 1)

        ttcf = np.asarray(dataset[run.wait_time :, run.wait_time :])
        frame = diagonal_average(ttcf, max_lag=max_lag)
        if "delay" in group:
            delay = np.asarray(group["delay"][: max_lag + 1], dtype=float)
        else:
            delay = frame["lag"].to_numpy(dtype=float)

    frame.insert(0, "delay_s", delay)
    frame.insert(0, "source_file", path.name)
    frame.insert(0, "wait_time", run.wait_time)
    frame.insert(0, "q_index", roi)
    frame.insert(0, "roi", roi)
    frame.insert(0, "temperature_k", run.temperature_k)
    frame.insert(0, "uid", run.uid)
    return frame


def reduced_path(output_dir: Path, run: RunConfig, roi: int) -> Path:
    return output_dir / f"uid{run.uid:d}_T{run.temperature_k:d}K_roi{roi:02d}_g2.csv"


def write_reduction(frame: pd.DataFrame, output_dir: Path, run: RunConfig, roi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = reduced_path(output_dir, run, roi)
    frame.to_csv(path, index=False)
    return path
