from __future__ import annotations

from dataclasses import dataclass

DEFAULT_DATA_DIR = "data"
DEFAULT_ANALYSIS_DIR = "analysis"
DEFAULT_CUTOFF = 599


@dataclass(frozen=True)
class RunConfig:
    uid: int
    temperature_k: int
    wait_time: int

    @property
    def filename(self) -> str:
        return f"processed_data_uid{self.uid:d}_equal_hkl_box_roi_12_twotime.h5"


RUNS: tuple[RunConfig, ...] = (
    RunConfig(uid=165012, temperature_k=10, wait_time=0),
    RunConfig(uid=165028, temperature_k=39, wait_time=0),
    RunConfig(uid=165052, temperature_k=50, wait_time=0),
    RunConfig(uid=165069, temperature_k=100, wait_time=0),
    RunConfig(uid=165079, temperature_k=150, wait_time=1800),
    RunConfig(uid=165089, temperature_k=200, wait_time=1800),
    RunConfig(uid=164970, temperature_k=250, wait_time=0),
)

RUNS_BY_UID = {run.uid: run for run in RUNS}
