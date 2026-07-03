from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from xcps_ai.config import DEFAULT_ANALYSIS_DIR, DEFAULT_CUTOFF, DEFAULT_DATA_DIR, RUNS, RUNS_BY_UID
from xcps_ai.hdf5 import available_rois, data_file
from xcps_ai.inventory import build_inventory
from xcps_ai.models import (
    FitOptions,
    fit_reduced_file,
    fit_shared_beta_reduced_files,
    fit_shared_tau_reduced_files,
    fit_tau_q_power_law_by_temperature,
    fit_tau_q_power_law_common_n,
    summarize_by_temperature,
)
from xcps_ai.plotting import (
    filter_grids_by_roi,
    load_isf_grids,
    plot_contrast_signature_figures,
    plot_isf_grid,
    plot_isf_grids,
    plot_spin_signature_figures,
    plot_tau_q_common_n_prefactor,
    plot_tau_q_power_law_parameters,
)
from xcps_ai.reduce import ReductionOptions, reduce_roi, reduced_path, write_reduction

app = typer.Typer(no_args_is_help=True)


def _runs(uids: list[int] | None):
    if not uids:
        return list(RUNS)
    unknown = sorted(set(uids) - set(RUNS_BY_UID))
    if unknown:
        raise typer.BadParameter(f"Unknown UID(s): {unknown}")
    return [RUNS_BY_UID[uid] for uid in uids]


def _reduced_files(path: Path) -> list[Path]:
    return sorted(file for file in path.glob("*_g2.csv") if file.name != "reduction_manifest.csv")


@app.command()
def inventory(
    data_dir: Path = typer.Option(Path(DEFAULT_DATA_DIR), "--data-dir"),
    output: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "inventory.csv", "--output", "-o"),
) -> None:
    """Create a metadata inventory for the local TTCF HDF5 files."""
    frame = build_inventory(data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame)} rows to {output}")


@app.command()
def reduce(
    data_dir: Path = typer.Option(Path(DEFAULT_DATA_DIR), "--data-dir"),
    output_dir: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "g2", "--output-dir", "-o"),
    uid: list[int] | None = typer.Option(None, "--uid", help="UID to process; repeatable."),
    roi: list[int] | None = typer.Option(None, "--roi", help="ROI/q-index to process; repeatable."),
    cutoff: int = typer.Option(DEFAULT_CUTOFF, "--cutoff"),
    max_lag: int | None = typer.Option(None, "--max-lag"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Compute one-time g2(tau) curves by TTCF diagonal averaging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_rois = set(roi) if roi else None
    rows = []
    options = ReductionOptions(cutoff=cutoff, max_lag=max_lag)

    for run in _runs(uid):
        rois = available_rois(data_file(data_dir, run))
        if requested_rois is not None:
            rois = [value for value in rois if value in requested_rois]
        for roi_value in rois:
            output = reduced_path(output_dir, run, roi_value)
            if output.exists() and not force:
                typer.echo(f"Skipping existing {output}")
                rows.append(
                    {
                        "uid": run.uid,
                        "temperature_k": run.temperature_k,
                        "roi": roi_value,
                        "output": str(output),
                        "status": "skipped",
                    }
                )
                continue

            typer.echo(f"Reducing UID {run.uid}, T={run.temperature_k} K, ROI/q-index {roi_value}")
            frame = reduce_roi(data_dir, run, roi_value, options)
            written = write_reduction(frame, output_dir, run, roi_value)
            rows.append(
                {
                    "uid": run.uid,
                    "temperature_k": run.temperature_k,
                    "roi": roi_value,
                    "output": str(written),
                    "status": "written",
                    "n_points": len(frame),
                    "max_delay_s": float(frame["delay_s"].max()),
                }
            )

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "reduction_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    typer.echo(f"Wrote reduction manifest to {manifest_path}")


@app.command()
def fit(
    reduced_dir: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "g2", "--reduced-dir"),
    output: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "fits.csv", "--output", "-o"),
    model: str = typer.Option("kww", "--model"),
    min_delay: float = typer.Option(1.0, "--min-delay"),
    max_delay: float | None = typer.Option(None, "--max-delay"),
    baseline: float = typer.Option(1.0, "--baseline"),
) -> None:
    """Fit reduced g2 curves."""
    if model not in {"exp", "kww"}:
        raise typer.BadParameter("model must be exp or kww")
    files = _reduced_files(reduced_dir)
    if not files:
        raise typer.BadParameter(f"No reduced g2 files found in {reduced_dir}")

    options = FitOptions(
        model=model,
        min_delay=min_delay,
        max_delay=max_delay,
        baseline=baseline,
    )  # type: ignore[arg-type]
    frame = pd.DataFrame([fit_reduced_file(path, options) for path in files])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame)} fit rows to {output}")


@app.command("fit-shared-tau")
def fit_shared_tau(
    reduced_dir: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "g2", "--reduced-dir"),
    output: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "fits_shared_tau_by_temperature.csv",
        "--output",
        "-o",
    ),
    model: str = typer.Option("kww", "--model"),
    min_delay: float = typer.Option(1.0, "--min-delay"),
    max_delay: float | None = typer.Option(None, "--max-delay"),
    baseline: float = typer.Option(1.0, "--baseline"),
) -> None:
    """Fit one shared relaxation time across all ROI curves at each temperature."""
    if model not in {"exp", "kww"}:
        raise typer.BadParameter("model must be exp or kww")
    files = _reduced_files(reduced_dir)
    if not files:
        raise typer.BadParameter(f"No reduced g2 files found in {reduced_dir}")

    options = FitOptions(
        model=model,
        min_delay=min_delay,
        max_delay=max_delay,
        baseline=baseline,
    )  # type: ignore[arg-type]
    frame = pd.DataFrame(fit_shared_tau_reduced_files(files, options))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame)} shared-tau fit rows to {output}")


@app.command("fit-shared-beta")
def fit_shared_beta(
    reduced_dir: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "g2", "--reduced-dir"),
    output: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "fits_shared_beta_by_temperature.csv",
        "--output",
        "-o",
    ),
    min_delay: float = typer.Option(1.0, "--min-delay"),
    max_delay: float | None = typer.Option(None, "--max-delay"),
    baseline: float = typer.Option(1.0, "--baseline"),
) -> None:
    """Fit one shared KWW beta across all ROI curves at each temperature."""
    files = _reduced_files(reduced_dir)
    if not files:
        raise typer.BadParameter(f"No reduced g2 files found in {reduced_dir}")

    options = FitOptions(
        model="kww",
        min_delay=min_delay,
        max_delay=max_delay,
        baseline=baseline,
    )
    frame = pd.DataFrame(fit_shared_beta_reduced_files(files, options))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame)} shared-beta fit rows to {output}")


@app.command()
def summarize(
    fits: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "fits.csv", "--fits"),
    output: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "temperature_summary.csv",
        "--output",
        "-o",
    ),
) -> None:
    """Summarize fit parameters by temperature."""
    frame = pd.read_csv(fits)
    summary = summarize_by_temperature(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    typer.echo(f"Wrote {len(summary)} temperature rows to {output}")


@app.command()
def plot_isf_contours(
    fits: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "fits_all_temperatures.csv", "--fits"),
    reduced_dir: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "g2", "--reduced-dir"),
    output_dir: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "plots" / "isf",
        "--output-dir",
        "-o",
    ),
    combined_name: str = typer.Option("isf_contours_all_temperatures.png", "--combined-name"),
    yscale: str = typer.Option("log", "--yscale", help="Y-axis scale: log or linear."),
    x_axis: str = typer.Option("roi", "--x-axis", help="X-axis: roi or q."),
    y_axis: str = typer.Option("tau", "--y-axis", help="Y-axis: tau or energy."),
    q_origin_rlu: float = typer.Option(0.0, "--q-origin-rlu"),
    q_step_rlu: float = typer.Option(0.0051, "--q-step-rlu"),
    energy_unit: str = typer.Option("feV", "--energy-unit"),
    roi_min: int | None = typer.Option(None, "--roi-min"),
    roi_max: int | None = typer.Option(None, "--roi-max"),
    exclude_roi: list[int] | None = typer.Option(
        None,
        "--exclude-roi",
        help="ROI index to omit; repeatable.",
    ),
    plot_points: bool = typer.Option(False, "--plot-points", help="Overlay every data point."),
    point_size: float = typer.Option(1.0, "--point-size", help="Marker size for data points."),
    point_alpha: float = typer.Option(0.25, "--point-alpha", help="Marker opacity."),
    y_min: float | None = typer.Option(None, "--y-min", help="Lower y-axis display limit."),
    y_max: float | None = typer.Option(None, "--y-max", help="Upper y-axis display limit."),
    vmin: float = typer.Option(0.0, "--vmin", help="Lower color-scale limit."),
    vmax: float = typer.Option(1.05, "--vmax", help="Upper color-scale limit."),
) -> None:
    """Plot filled |F(tau)| contours by ROI for every temperature."""
    if yscale not in {"linear", "log"}:
        raise typer.BadParameter("yscale must be log or linear")
    if x_axis not in {"roi", "q"}:
        raise typer.BadParameter("x-axis must be roi or q")
    if y_axis not in {"tau", "energy"}:
        raise typer.BadParameter("y-axis must be tau or energy")
    if vmin >= vmax:
        raise typer.BadParameter("vmin must be smaller than vmax")
    if point_size <= 0:
        raise typer.BadParameter("point-size must be positive")
    if not 0 < point_alpha <= 1:
        raise typer.BadParameter("point-alpha must be in (0, 1]")
    if y_min is not None and y_max is not None and y_min >= y_max:
        raise typer.BadParameter("y-min must be smaller than y-max")
    if roi_min is not None and roi_max is not None and roi_min > roi_max:
        raise typer.BadParameter("roi-min must be less than or equal to roi-max")

    grids = load_isf_grids(fits, reduced_dir)
    grids = filter_grids_by_roi(
        grids,
        roi_min=roi_min,
        roi_max=roi_max,
        exclude_roi=set(exclude_roi or []),
    )
    if not grids:
        raise typer.BadParameter("ROI filters removed all data")
    output_dir.mkdir(parents=True, exist_ok=True)

    for grid in grids:
        temperature = (
            int(grid.temperature_k)
            if float(grid.temperature_k).is_integer()
            else grid.temperature_k
        )
        output = output_dir / f"isf_contour_T{temperature}K.png"
        plot_isf_grid(
            grid,
            output,
            yscale=yscale,
            x_axis=x_axis,
            y_axis=y_axis,
            q_origin_rlu=q_origin_rlu,
            q_step_rlu=q_step_rlu,
            energy_unit=energy_unit,
            plot_points=plot_points,
            point_size=point_size,
            point_alpha=point_alpha,
            y_min=y_min,
            y_max=y_max,
            vmin=vmin,
            vmax=vmax,
        )
        typer.echo(f"Wrote {output}")

    combined = output_dir / combined_name
    plot_isf_grids(
        grids,
        combined,
        yscale=yscale,
        x_axis=x_axis,
        y_axis=y_axis,
        q_origin_rlu=q_origin_rlu,
        q_step_rlu=q_step_rlu,
        energy_unit=energy_unit,
        plot_points=plot_points,
        point_size=point_size,
        point_alpha=point_alpha,
        y_min=y_min,
        y_max=y_max,
        vmin=vmin,
        vmax=vmax,
    )
    typer.echo(f"Wrote {combined}")


@app.command()
def plot_spin_signature(
    fits: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "fits_all_temperatures.csv", "--fits"),
    output_dir: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "plots" / "spin_signature",
        "--output-dir",
        "-o",
    ),
    roi_min: int = typer.Option(0, "--roi-min"),
    roi_max: int = typer.Option(8, "--roi-max"),
    q_step_rlu: float = typer.Option(0.0051, "--q-step-rlu"),
) -> None:
    """Plot fit-parameter views supporting the below-50 K spin-order interpretation."""
    if roi_min > roi_max:
        raise typer.BadParameter("roi-min must be less than or equal to roi-max")
    if q_step_rlu <= 0:
        raise typer.BadParameter("q-step-rlu must be positive")

    paths = plot_spin_signature_figures(
        fits,
        output_dir,
        roi_min=roi_min,
        roi_max=roi_max,
        q_step_rlu=q_step_rlu,
    )
    for name, path in paths.items():
        typer.echo(f"Wrote {name}: {path}")


@app.command()
def plot_contrast_signature(
    fits: Path = typer.Option(Path(DEFAULT_ANALYSIS_DIR) / "fits_all_temperatures.csv", "--fits"),
    output_dir: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "plots" / "contrast_signature",
        "--output-dir",
        "-o",
    ),
    roi_min: int = typer.Option(0, "--roi-min"),
    roi_max: int = typer.Option(8, "--roi-max"),
    q_step_rlu: float = typer.Option(0.0051, "--q-step-rlu"),
) -> None:
    """Plot fitted-contrast views for interpreting amplitude trends."""
    if roi_min > roi_max:
        raise typer.BadParameter("roi-min must be less than or equal to roi-max")
    if q_step_rlu <= 0:
        raise typer.BadParameter("q-step-rlu must be positive")

    paths = plot_contrast_signature_figures(
        fits,
        output_dir,
        roi_min=roi_min,
        roi_max=roi_max,
        q_step_rlu=q_step_rlu,
    )
    for name, path in paths.items():
        typer.echo(f"Wrote {name}: {path}")


@app.command("fit-sub-ballistic")
def fit_sub_ballistic(
    fits: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "fits_shared_beta_by_temperature.csv",
        "--fits",
    ),
    output_dir: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "sub_ballistic_tau_q",
        "--output-dir",
        "-o",
    ),
    roi_min: int = typer.Option(1, "--roi-min"),
    roi_max: int | None = typer.Option(8, "--roi-max"),
    q_origin_rlu: float = typer.Option(0.0, "--q-origin-rlu"),
    q_step_rlu: float = typer.Option(0.0051, "--q-step-rlu"),
) -> None:
    """Fit tau = k * Q^-n at each temperature and plot k(T), n(T)."""
    if roi_min < 0:
        raise typer.BadParameter("roi-min must be nonnegative")
    if roi_max is not None and roi_min > roi_max:
        raise typer.BadParameter("roi-min must be less than or equal to roi-max")
    if q_step_rlu <= 0:
        raise typer.BadParameter("q-step-rlu must be positive")

    frame = pd.read_csv(fits)
    parameters = fit_tau_q_power_law_by_temperature(
        frame,
        roi_min=roi_min,
        roi_max=roi_max,
        q_origin_rlu=q_origin_rlu,
        q_step_rlu=q_step_rlu,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = output_dir / "tau_q_power_law_parameters.csv"
    parameters.to_csv(parameters_path, index=False)
    typer.echo(f"Wrote parameters: {parameters_path}")

    paths = plot_tau_q_power_law_parameters(parameters, output_dir)
    for name, path in paths.items():
        typer.echo(f"Wrote {name}: {path}")


@app.command("fit-sub-ballistic-common-n")
def fit_sub_ballistic_common_n(
    fits: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "fits_shared_beta_by_temperature.csv",
        "--fits",
    ),
    output_dir: Path = typer.Option(
        Path(DEFAULT_ANALYSIS_DIR) / "sub_ballistic_tau_q_common_n",
        "--output-dir",
        "-o",
    ),
    roi_min: int = typer.Option(1, "--roi-min"),
    roi_max: int | None = typer.Option(8, "--roi-max"),
    q_origin_rlu: float = typer.Option(0.0, "--q-origin-rlu"),
    q_step_rlu: float = typer.Option(0.0051, "--q-step-rlu"),
    exclude_temperature: list[float] | None = typer.Option(
        None,
        "--exclude-temperature",
        help="Temperature to omit from the shared-n fit; repeatable. Defaults to 250 K.",
    ),
) -> None:
    """Fit tau = k(T) * Q^-n with one common n across temperatures."""
    if roi_min < 0:
        raise typer.BadParameter("roi-min must be nonnegative")
    if roi_max is not None and roi_min > roi_max:
        raise typer.BadParameter("roi-min must be less than or equal to roi-max")
    if q_step_rlu <= 0:
        raise typer.BadParameter("q-step-rlu must be positive")

    frame = pd.read_csv(fits)
    excluded_temperatures = {250.0} if exclude_temperature is None else set(exclude_temperature)
    parameters, summary = fit_tau_q_power_law_common_n(
        frame,
        roi_min=roi_min,
        roi_max=roi_max,
        q_origin_rlu=q_origin_rlu,
        q_step_rlu=q_step_rlu,
        exclude_temperatures=excluded_temperatures,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = output_dir / "tau_q_common_n_prefactors.csv"
    summary_path = output_dir / "tau_q_common_n_summary.csv"
    parameters.to_csv(parameters_path, index=False)
    summary.to_csv(summary_path, index=False)
    typer.echo(f"Wrote prefactors: {parameters_path}")
    typer.echo(f"Wrote summary: {summary_path}")

    paths = plot_tau_q_common_n_prefactor(parameters, summary, output_dir)
    for name, path in paths.items():
        typer.echo(f"Wrote {name}: {path}")
