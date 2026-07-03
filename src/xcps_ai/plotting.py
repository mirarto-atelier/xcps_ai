from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HBAR_EV_S = 6.582119569e-16
ENERGY_UNITS_EV = {
    "eV": 1.0,
    "meV": 1e-3,
    "ueV": 1e-6,
    "neV": 1e-9,
    "peV": 1e-12,
    "feV": 1e-15,
}


@dataclass(frozen=True)
class IsfGrid:
    temperature_k: float
    roi: np.ndarray
    delay_s: np.ndarray
    isf: np.ndarray


def intermediate_scattering_function(
    g2: np.ndarray,
    contrast: float,
    baseline: float = 1.0,
) -> np.ndarray:
    """Compute |F(tau)| from g2(tau) using the Siegert relation."""
    if contrast <= 0 or not np.isfinite(contrast):
        return np.full_like(g2, np.nan, dtype=float)
    return np.sqrt(np.clip((np.asarray(g2, dtype=float) - baseline) / contrast, 0.0, None))


def roi_to_q_rlu(
    roi: np.ndarray,
    q_origin_rlu: float = 0.0,
    q_step_rlu: float = 0.0051,
) -> np.ndarray:
    return q_origin_rlu + np.asarray(roi, dtype=float) * q_step_rlu


def tau_to_energy(
    delay_s: np.ndarray,
    unit: str = "feV",
) -> np.ndarray:
    if unit not in ENERGY_UNITS_EV:
        allowed = ", ".join(sorted(ENERGY_UNITS_EV))
        raise ValueError(f"energy unit must be one of: {allowed}")
    delay = np.asarray(delay_s, dtype=float)
    energy = np.full(delay.shape, np.nan, dtype=float)
    valid = delay > 0
    energy[valid] = (HBAR_EV_S / delay[valid]) / ENERGY_UNITS_EV[unit]
    return energy


def _resolved_reduced_file(row: pd.Series, reduced_dir: Path | None) -> Path:
    if "reduced_file" in row and pd.notna(row["reduced_file"]):
        return Path(str(row["reduced_file"]))
    if reduced_dir is None:
        raise ValueError("fits table has no reduced_file column; pass reduced_dir")
    roi = int(row["roi"])
    uid = int(row["uid"])
    temperature = int(row["temperature_k"])
    return reduced_dir / f"uid{uid}_T{temperature}K_roi{roi:02d}_g2.csv"


def load_isf_grids(
    fits_file: Path,
    reduced_dir: Path | None = None,
) -> list[IsfGrid]:
    fits = pd.read_csv(fits_file)
    if "success" in fits:
        fits = fits[fits["success"] == True].copy()  # noqa: E712
    fits = fits[np.isfinite(fits["contrast"]) & (fits["contrast"] > 0)].copy()
    if fits.empty:
        raise ValueError(f"No successful fits with positive contrast in {fits_file}")

    grids: list[IsfGrid] = []
    for temperature, group in fits.groupby("temperature_k", sort=True):
        roi_values: list[int] = []
        columns: list[np.ndarray] = []
        common_delay: np.ndarray | None = None

        for _, row in group.sort_values("roi").iterrows():
            reduced_file = _resolved_reduced_file(row, reduced_dir)
            reduced = pd.read_csv(reduced_file)
            delay = reduced["delay_s"].to_numpy(dtype=float)
            g2 = reduced["g2"].to_numpy(dtype=float)
            isf = intermediate_scattering_function(
                g2,
                contrast=float(row["contrast"]),
                baseline=float(row.get("baseline", 1.0)),
            )
            if common_delay is None:
                common_delay = delay
                column = isf
            elif delay.shape == common_delay.shape and np.allclose(delay, common_delay):
                column = isf
            else:
                column = np.interp(common_delay, delay, isf, left=np.nan, right=np.nan)

            roi_values.append(int(row["roi"]))
            columns.append(column)

        if common_delay is None:
            continue
        grids.append(
            IsfGrid(
                temperature_k=float(temperature),
                roi=np.asarray(roi_values, dtype=int),
                delay_s=common_delay,
                isf=np.column_stack(columns),
            )
        )

    return grids


def filter_grids_by_roi(
    grids: list[IsfGrid],
    *,
    roi_min: int | None = None,
    roi_max: int | None = None,
    exclude_roi: set[int] | None = None,
) -> list[IsfGrid]:
    filtered: list[IsfGrid] = []
    excluded = exclude_roi or set()
    for grid in grids:
        mask = np.ones(grid.roi.shape, dtype=bool)
        if roi_min is not None:
            mask &= grid.roi >= roi_min
        if roi_max is not None:
            mask &= grid.roi <= roi_max
        if excluded:
            mask &= ~np.isin(grid.roi, list(excluded))
        if not np.any(mask):
            continue
        filtered.append(
            IsfGrid(
                temperature_k=grid.temperature_k,
                roi=grid.roi[mask],
                delay_s=grid.delay_s,
                isf=grid.isf[:, mask],
            )
        )
    return filtered


def _format_temperature(temperature_k: float) -> str:
    if float(temperature_k).is_integer():
        return f"{int(temperature_k)} K"
    return f"{temperature_k:g} K"


def _axis_values(
    grid: IsfGrid,
    *,
    x_axis: str,
    y_axis: str,
    q_origin_rlu: float,
    q_step_rlu: float,
    energy_unit: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    if x_axis == "roi":
        x = grid.roi
        xlabel = "ROI"
    elif x_axis == "q":
        x = roi_to_q_rlu(grid.roi, q_origin_rlu=q_origin_rlu, q_step_rlu=q_step_rlu)
        xlabel = "q (r.l.u.)"
    else:
        raise ValueError("x_axis must be 'roi' or 'q'")

    if y_axis == "tau":
        y = grid.delay_s
        z = grid.isf
        ylabel = "tau (s)"
    elif y_axis == "energy":
        y = tau_to_energy(grid.delay_s, unit=energy_unit)
        finite = np.isfinite(y)
        y = y[finite]
        z = grid.isf[finite, :]
        order = np.argsort(y)
        y = y[order]
        z = z[order, :]
        ylabel = f"energy hbar/tau ({energy_unit})"
    else:
        raise ValueError("y_axis must be 'tau' or 'energy'")

    return x, y, z, xlabel, ylabel


def _set_y_axis(
    ax: plt.Axes,
    y: np.ndarray,
    yscale: str,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        raise ValueError("No finite y-axis values")
    if yscale == "log":
        finite = finite[finite > 0]
    if finite.size == 0:
        raise ValueError(f"No valid y-axis values for {yscale} scale")
    visible = finite
    if y_min is not None:
        visible = visible[visible >= y_min]
    if y_max is not None:
        visible = visible[visible <= y_max]
    if visible.size == 0:
        raise ValueError("No y-axis values inside requested y limits")

    lower = float(np.nanmin(visible)) if y_min is None else y_min
    upper = float(np.nanmax(visible)) if y_max is None else y_max
    if yscale == "log" and lower <= 0:
        lower = float(np.nanmin(visible[visible > 0]))
    ax.set_yscale(yscale)
    ax.set_ylim(lower, upper)


def _plot_data_points(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    point_size: float,
    point_alpha: float,
) -> None:
    x_grid, y_grid = np.meshgrid(x, y)
    finite = np.isfinite(x_grid) & np.isfinite(y_grid) & np.isfinite(z)
    ax.scatter(
        x_grid[finite],
        y_grid[finite],
        color="black",
        s=point_size,
        linewidths=0,
        alpha=point_alpha,
        rasterized=True,
    )


def plot_isf_grid(
    grid: IsfGrid,
    output: Path,
    *,
    vmin: float = 0.0,
    vmax: float = 1.05,
    levels: int = 22,
    cmap: str = "magma",
    yscale: str = "log",
    x_axis: str = "roi",
    y_axis: str = "tau",
    q_origin_rlu: float = 0.0,
    q_step_rlu: float = 0.0051,
    energy_unit: str = "feV",
    plot_points: bool = False,
    point_size: float = 1.0,
    point_alpha: float = 0.25,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'")
    x, y, z, xlabel, ylabel = _axis_values(
        grid,
        x_axis=x_axis,
        y_axis=y_axis,
        q_origin_rlu=q_origin_rlu,
        q_step_rlu=q_step_rlu,
        energy_unit=energy_unit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    contour_levels = np.linspace(vmin, vmax, levels)
    image = ax.contourf(
        x,
        y,
        z,
        levels=contour_levels,
        cmap=cmap,
        extend="both",
    )
    if plot_points:
        _plot_data_points(
            ax,
            x,
            y,
            z,
            point_size=point_size,
            point_alpha=point_alpha,
        )
    ax.set_title(_format_temperature(grid.temperature_k))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _set_y_axis(ax, y, yscale, y_min=y_min, y_max=y_max)
    fig.colorbar(image, ax=ax, label="|F| from measured g2")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_isf_grids(
    grids: list[IsfGrid],
    output: Path,
    *,
    vmin: float = 0.0,
    vmax: float = 1.05,
    levels: int = 22,
    cmap: str = "magma",
    yscale: str = "log",
    x_axis: str = "roi",
    y_axis: str = "tau",
    q_origin_rlu: float = 0.0,
    q_step_rlu: float = 0.0051,
    energy_unit: str = "feV",
    plot_points: bool = False,
    point_size: float = 1.0,
    point_alpha: float = 0.25,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    if not grids:
        raise ValueError("No ISF grids to plot")
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'")
    output.parent.mkdir(parents=True, exist_ok=True)
    ncols = min(4, len(grids))
    nrows = ceil(len(grids) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.1 * ncols, 3.8 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    contour_levels = np.linspace(vmin, vmax, levels)
    image = None

    for ax, grid in zip(axes.ravel(), grids, strict=False):
        x, y, z, xlabel, ylabel = _axis_values(
            grid,
            x_axis=x_axis,
            y_axis=y_axis,
            q_origin_rlu=q_origin_rlu,
            q_step_rlu=q_step_rlu,
            energy_unit=energy_unit,
        )
        image = ax.contourf(
            x,
            y,
            z,
            levels=contour_levels,
            cmap=cmap,
            extend="both",
        )
        if plot_points:
            _plot_data_points(
                ax,
                x,
                y,
                z,
                point_size=point_size,
                point_alpha=point_alpha,
            )
        ax.set_title(_format_temperature(grid.temperature_k))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _set_y_axis(ax, y, yscale, y_min=y_min, y_max=y_max)

    for ax in axes.ravel()[len(grids) :]:
        ax.axis("off")

    fig.suptitle("KCuF3 XPCS intermediate scattering function |F| by temperature")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="|F| from measured g2")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def load_fit_parameter_frame(
    fits_file: Path,
    *,
    roi_min: int = 0,
    roi_max: int = 8,
    q_step_rlu: float = 0.0051,
) -> pd.DataFrame:
    frame = pd.read_csv(fits_file)
    if "success" in frame:
        if frame["success"].dtype == bool:
            frame = frame[frame["success"]].copy()
        else:
            frame = frame[frame["success"].astype(str).str.lower().isin({"true", "1"})].copy()
    frame = frame[(frame["roi"] >= roi_min) & (frame["roi"] <= roi_max)].copy()
    if frame.empty:
        raise ValueError("No successful fit rows remain after ROI filtering")
    frame["q_rlu"] = frame["roi"].astype(float) * q_step_rlu
    return frame.sort_values(["temperature_k", "roi"]).reset_index(drop=True)


def spin_signature_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_temperatures = {10, 39, 50}
    temperatures = set(frame["temperature_k"].astype(int))
    missing = sorted(required_temperatures - temperatures)
    if missing:
        raise ValueError(f"Missing required below-50 K temperatures: {missing}")

    temperature_summary = (
        frame.groupby("temperature_k", sort=True)
        .agg(
            n=("roi", "count"),
            tau_median_s=("tau_s", "median"),
            tau_q25_s=("tau_s", lambda values: values.quantile(0.25)),
            tau_q75_s=("tau_s", lambda values: values.quantile(0.75)),
            beta_median=("beta", "median"),
            beta_q25=("beta", lambda values: values.quantile(0.25)),
            beta_q75=("beta", lambda values: values.quantile(0.75)),
            contrast_median=("contrast", "median"),
            r_squared_median=("r_squared", "median"),
        )
        .reset_index()
    )

    below = frame[frame["temperature_k"].isin([10, 39, 50])].copy()
    tau = below.pivot(index=["roi", "q_rlu"], columns="temperature_k", values="tau_s")
    beta = below.pivot(index=["roi", "q_rlu"], columns="temperature_k", values="beta")
    contrast = below.pivot(index=["roi", "q_rlu"], columns="temperature_k", values="contrast")
    ratio = pd.DataFrame(index=tau.index)
    ratio["tau_10_s"] = tau[10]
    ratio["tau_39_s"] = tau[39]
    ratio["tau_50_s"] = tau[50]
    ratio["tau39_over_tau50"] = tau[39] / tau[50]
    ratio["tau10_over_tau39"] = tau[10] / tau[39]
    ratio["beta_10"] = beta[10]
    ratio["beta_39"] = beta[39]
    ratio["beta_50"] = beta[50]
    ratio["contrast_10"] = contrast[10]
    ratio["contrast_39"] = contrast[39]
    ratio["contrast_50"] = contrast[50]
    return temperature_summary, ratio.reset_index()


def _mark_reference_temperatures(ax: plt.Axes) -> None:
    ax.axvline(39.0, color="#4257a8", linestyle="--", linewidth=1.2)
    ax.axvline(50.0, color="#9f4a34", linestyle="--", linewidth=1.2)
    ax.text(39.0, 0.98, "AFM", transform=ax.get_xaxis_transform(), ha="right", va="top")
    ax.text(50.0, 0.98, "50 K", transform=ax.get_xaxis_transform(), ha="left", va="top")


def _temperature_trend_plot(frame: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    for _roi, group in frame.groupby("roi"):
        group = group.sort_values("temperature_k")
        axes[0].plot(group["temperature_k"], group["tau_s"], color="0.82", linewidth=0.8)
        axes[1].plot(group["temperature_k"], group["beta"], color="0.82", linewidth=0.8)

    axes[0].fill_between(
        summary["temperature_k"],
        summary["tau_q25_s"],
        summary["tau_q75_s"],
        color="#5875a4",
        alpha=0.18,
        label="ROI IQR",
    )
    axes[0].plot(
        summary["temperature_k"],
        summary["tau_median_s"],
        marker="o",
        color="#1f3b73",
        label="median ROI 0-8",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("temperature (K)")
    axes[0].set_ylabel("fit tau_s (s)")
    axes[0].set_title("Relaxation time has a 50 K maximum")
    axes[0].legend(loc="best")

    axes[1].fill_between(
        summary["temperature_k"],
        summary["beta_q25"],
        summary["beta_q75"],
        color="#8c6bb1",
        alpha=0.18,
        label="ROI IQR",
    )
    axes[1].plot(
        summary["temperature_k"],
        summary["beta_median"],
        marker="o",
        color="#5c2a7d",
        label="median ROI 0-8",
    )
    axes[1].axhline(1.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("temperature (K)")
    axes[1].set_ylabel("KWW beta")
    axes[1].set_title("Compressed KWW beta near 39-50 K")
    axes[1].legend(loc="best")

    for ax in axes:
        _mark_reference_temperatures(ax)
        ax.grid(True, alpha=0.25)

    fig.savefig(output, dpi=180)
    plt.close(fig)


def _below_50_q_plot(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = {50: "#9f4a34", 39: "#4257a8", 10: "#1b7f5a"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)

    below = frame[frame["temperature_k"].isin([10, 39, 50])].copy()
    for temperature, group in below.groupby("temperature_k", sort=False):
        label = f"{int(temperature)} K"
        group = group.sort_values("q_rlu")
        axes[0].plot(
            group["q_rlu"],
            group["tau_s"],
            marker="o",
            color=colors[int(temperature)],
            label=label,
        )
        axes[1].plot(
            group["q_rlu"],
            group["beta"],
            marker="o",
            color=colors[int(temperature)],
            label=label,
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("q (r.l.u.)")
    axes[0].set_ylabel("fit tau_s (s)")
    axes[0].set_title("39 K is faster than 50 K across q")
    axes[0].legend(loc="best")

    axes[1].axhline(1.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("q (r.l.u.)")
    axes[1].set_ylabel("KWW beta")
    axes[1].set_title("Below-50 K beta versus q")
    axes[1].legend(loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.savefig(output, dpi=180)
    plt.close(fig)


def _ratio_plot(ratio: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.3, 4.6), constrained_layout=True)
    ax.axhline(1.0, color="0.25", linestyle=":", linewidth=1.0)
    ax.plot(
        ratio["q_rlu"],
        ratio["tau39_over_tau50"],
        marker="o",
        color="#4257a8",
        label="tau(39 K) / tau(50 K)",
    )
    ax.plot(
        ratio["q_rlu"],
        ratio["tau10_over_tau39"],
        marker="o",
        color="#1b7f5a",
        label="tau(10 K) / tau(39 K)",
    )
    ax.set_xlabel("q (r.l.u.)")
    ax.set_ylabel("tau ratio")
    ax.set_title("Speed-up at AFM scale and refreezing at 10 K")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _spin_signature_summary_plot(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    ratio: pd.DataFrame,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = {50: "#9f4a34", 39: "#4257a8", 10: "#1b7f5a"}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)

    axes[0, 0].fill_between(
        summary["temperature_k"],
        summary["tau_q25_s"],
        summary["tau_q75_s"],
        color="#5875a4",
        alpha=0.18,
    )
    axes[0, 0].plot(summary["temperature_k"], summary["tau_median_s"], marker="o", color="#1f3b73")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("temperature (K)")
    axes[0, 0].set_ylabel("median tau_s (s)")
    axes[0, 0].set_title("Non-monotonic slowing near 50 K")
    _mark_reference_temperatures(axes[0, 0])

    axes[0, 1].fill_between(
        summary["temperature_k"],
        summary["beta_q25"],
        summary["beta_q75"],
        color="#8c6bb1",
        alpha=0.18,
    )
    axes[0, 1].plot(summary["temperature_k"], summary["beta_median"], marker="o", color="#5c2a7d")
    axes[0, 1].axhline(1.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[0, 1].set_xlabel("temperature (K)")
    axes[0, 1].set_ylabel("median beta")
    axes[0, 1].set_title("Compressed KWW beta near 39-50 K")
    _mark_reference_temperatures(axes[0, 1])

    below = frame[frame["temperature_k"].isin([10, 39, 50])].copy()
    for temperature, group in below.groupby("temperature_k", sort=False):
        group = group.sort_values("q_rlu")
        axes[1, 0].plot(
            group["q_rlu"],
            group["tau_s"],
            marker="o",
            color=colors[int(temperature)],
            label=f"{int(temperature)} K",
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("q (r.l.u.)")
    axes[1, 0].set_ylabel("tau_s (s)")
    axes[1, 0].set_title("Below-50 K q dependence")
    axes[1, 0].legend(loc="best")

    axes[1, 1].axhline(1.0, color="0.25", linestyle=":", linewidth=1.0)
    axes[1, 1].plot(
        ratio["q_rlu"],
        ratio["tau39_over_tau50"],
        marker="o",
        color="#4257a8",
        label="39 K / 50 K",
    )
    axes[1, 1].plot(
        ratio["q_rlu"],
        ratio["tau10_over_tau39"],
        marker="o",
        color="#1b7f5a",
        label="10 K / 39 K",
    )
    axes[1, 1].set_xlabel("q (r.l.u.)")
    axes[1, 1].set_ylabel("tau ratio")
    axes[1, 1].set_title("AFM-scale speed-up then low-T refreezing")
    axes[1, 1].legend(loc="best")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)

    fig.suptitle("KCuF3 XPCS signatures below 50 K")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_spin_signature_figures(
    fits_file: Path,
    output_dir: Path,
    *,
    roi_min: int = 0,
    roi_max: int = 8,
    q_step_rlu: float = 0.0051,
) -> dict[str, Path]:
    frame = load_fit_parameter_frame(
        fits_file,
        roi_min=roi_min,
        roi_max=roi_max,
        q_step_rlu=q_step_rlu,
    )
    summary, ratio = spin_signature_tables(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "temperature_summary_roi_filtered.csv"
    ratio_path = output_dir / "below_50_tau_ratios_by_q.csv"
    summary.to_csv(summary_path, index=False)
    ratio.to_csv(ratio_path, index=False)

    paths = {
        "temperature_trends": output_dir / "temperature_trends_tau_beta.png",
        "below_50_q": output_dir / "below_50_tau_beta_vs_q.png",
        "ratios": output_dir / "below_50_tau_ratios_vs_q.png",
        "summary": output_dir / "spin_signature_summary.png",
        "temperature_summary_csv": summary_path,
        "ratio_csv": ratio_path,
    }
    _temperature_trend_plot(frame, summary, paths["temperature_trends"])
    _below_50_q_plot(frame, paths["below_50_q"])
    _ratio_plot(ratio, paths["ratios"])
    _spin_signature_summary_plot(frame, summary, ratio, paths["summary"])
    return paths


def contrast_signature_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    temperature_summary = (
        frame.groupby("temperature_k", sort=True)
        .agg(
            n=("roi", "count"),
            contrast_median=("contrast", "median"),
            contrast_q25=("contrast", lambda values: values.quantile(0.25)),
            contrast_q75=("contrast", lambda values: values.quantile(0.75)),
            tau_median_s=("tau_s", "median"),
            beta_median=("beta", "median"),
            r_squared_median=("r_squared", "median"),
        )
        .reset_index()
    )

    below = frame[frame["temperature_k"].isin([10, 39, 50])].copy()
    required_temperatures = {10, 39, 50}
    temperatures = set(below["temperature_k"].astype(int))
    missing = sorted(required_temperatures - temperatures)
    if missing:
        raise ValueError(f"Missing required below-50 K temperatures: {missing}")

    contrast = below.pivot(index=["roi", "q_rlu"], columns="temperature_k", values="contrast")
    tau = below.pivot(index=["roi", "q_rlu"], columns="temperature_k", values="tau_s")
    contrast_by_q = pd.DataFrame(index=contrast.index)
    contrast_by_q["contrast_10"] = contrast[10]
    contrast_by_q["contrast_39"] = contrast[39]
    contrast_by_q["contrast_50"] = contrast[50]
    contrast_by_q["contrast39_over_contrast50"] = contrast[39] / contrast[50]
    contrast_by_q["contrast10_over_contrast39"] = contrast[10] / contrast[39]
    contrast_by_q["tau39_over_tau50"] = tau[39] / tau[50]
    return temperature_summary, contrast_by_q.reset_index()


def _contrast_signature_summary_plot(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    contrast_by_q: pd.DataFrame,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)

    axes[0, 0].fill_between(
        summary["temperature_k"],
        summary["contrast_q25"],
        summary["contrast_q75"],
        color="#d48b4c",
        alpha=0.22,
        label="ROI IQR",
    )
    axes[0, 0].plot(
        summary["temperature_k"],
        summary["contrast_median"],
        marker="o",
        color="#a65628",
        label="median ROI 0-8",
    )
    axes[0, 0].set_xlabel("temperature (K)")
    axes[0, 0].set_ylabel("fitted contrast A")
    axes[0, 0].set_title("Contrast is suppressed at 10 K")
    axes[0, 0].legend(loc="best")
    _mark_reference_temperatures(axes[0, 0])

    normalized = summary.copy()
    normalized["contrast_norm"] = (
        normalized["contrast_median"] / normalized["contrast_median"].max()
    )
    normalized["tau_norm"] = normalized["tau_median_s"] / normalized["tau_median_s"].max()
    axes[0, 1].plot(
        normalized["temperature_k"],
        normalized["contrast_norm"],
        marker="o",
        color="#a65628",
        label="median contrast / max",
    )
    axes[0, 1].plot(
        normalized["temperature_k"],
        normalized["tau_norm"],
        marker="o",
        color="#1f3b73",
        label="median tau / max",
    )
    axes[0, 1].set_xlabel("temperature (K)")
    axes[0, 1].set_ylabel("normalized value")
    axes[0, 1].set_title("39 K tau change is not an amplitude artifact")
    axes[0, 1].legend(loc="best")
    _mark_reference_temperatures(axes[0, 1])

    colors = {50: "#9f4a34", 39: "#4257a8", 10: "#1b7f5a"}
    below = frame[frame["temperature_k"].isin([10, 39, 50])].copy()
    for temperature, group in below.groupby("temperature_k", sort=False):
        group = group.sort_values("q_rlu")
        axes[1, 0].plot(
            group["q_rlu"],
            group["contrast"],
            marker="o",
            color=colors[int(temperature)],
            label=f"{int(temperature)} K",
        )
    axes[1, 0].set_xlabel("q (r.l.u.)")
    axes[1, 0].set_ylabel("fitted contrast A")
    axes[1, 0].set_title("39 K contrast enhancement is strongest at mid-q")
    axes[1, 0].legend(loc="best")

    axes[1, 1].axhline(1.0, color="0.25", linestyle=":", linewidth=1.0)
    axes[1, 1].plot(
        contrast_by_q["q_rlu"],
        contrast_by_q["contrast39_over_contrast50"],
        marker="o",
        color="#4257a8",
        label="A(39 K) / A(50 K)",
    )
    axes[1, 1].plot(
        contrast_by_q["q_rlu"],
        contrast_by_q["contrast10_over_contrast39"],
        marker="o",
        color="#1b7f5a",
        label="A(10 K) / A(39 K)",
    )
    axes[1, 1].set_xlabel("q (r.l.u.)")
    axes[1, 1].set_ylabel("contrast ratio")
    axes[1, 1].set_title("Low-T refreezing reduces fitted contrast")
    axes[1, 1].legend(loc="best")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)

    fig.suptitle("KCuF3 XPCS fitted-contrast signatures")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _contrast_vs_q_plot(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = {50: "#9f4a34", 39: "#4257a8", 10: "#1b7f5a"}
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    below = frame[frame["temperature_k"].isin([10, 39, 50])].copy()
    for temperature, group in below.groupby("temperature_k", sort=False):
        group = group.sort_values("q_rlu")
        ax.plot(
            group["q_rlu"],
            group["contrast"],
            marker="o",
            color=colors[int(temperature)],
            label=f"{int(temperature)} K",
        )
    ax.set_xlabel("q (r.l.u.)")
    ax.set_ylabel("fitted contrast A")
    ax.set_title("Below-50 K fitted contrast versus q")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _contrast_tau_decoupling_plot(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = summary.copy()
    normalized["contrast_norm"] = (
        normalized["contrast_median"] / normalized["contrast_median"].max()
    )
    normalized["tau_norm"] = normalized["tau_median_s"] / normalized["tau_median_s"].max()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.plot(
        normalized["temperature_k"],
        normalized["contrast_norm"],
        marker="o",
        color="#a65628",
        label="median contrast / max",
    )
    ax.plot(
        normalized["temperature_k"],
        normalized["tau_norm"],
        marker="o",
        color="#1f3b73",
        label="median tau / max",
    )
    _mark_reference_temperatures(ax)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("normalized value")
    ax.set_title("Contrast and relaxation time separate below 50 K")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_contrast_signature_figures(
    fits_file: Path,
    output_dir: Path,
    *,
    roi_min: int = 0,
    roi_max: int = 8,
    q_step_rlu: float = 0.0051,
) -> dict[str, Path]:
    frame = load_fit_parameter_frame(
        fits_file,
        roi_min=roi_min,
        roi_max=roi_max,
        q_step_rlu=q_step_rlu,
    )
    summary, contrast_by_q = contrast_signature_tables(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "contrast_temperature_summary.csv"
    contrast_by_q_path = output_dir / "below_50_contrast_by_q.csv"
    summary.to_csv(summary_path, index=False)
    contrast_by_q.to_csv(contrast_by_q_path, index=False)

    paths = {
        "summary": output_dir / "contrast_signature_summary.png",
        "contrast_vs_q": output_dir / "below_50_contrast_vs_q.png",
        "contrast_tau_decoupling": output_dir / "contrast_tau_decoupling.png",
        "temperature_summary_csv": summary_path,
        "contrast_by_q_csv": contrast_by_q_path,
    }
    _contrast_signature_summary_plot(frame, summary, contrast_by_q, paths["summary"])
    _contrast_vs_q_plot(frame, paths["contrast_vs_q"])
    _contrast_tau_decoupling_plot(summary, paths["contrast_tau_decoupling"])
    return paths


def _errorbar_or_line(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    yerr: pd.Series | None,
    *,
    color: str,
) -> None:
    finite_yerr = yerr is not None and np.isfinite(yerr).any()
    if finite_yerr:
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            color=color,
            linewidth=1.5,
            capsize=3.0,
        )
    else:
        ax.plot(x, y, marker="o", color=color, linewidth=1.5)


def plot_tau_q_power_law_parameters(
    parameters: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot k(T) and n(T) from tau = k * Q^-n fits as separate figures."""
    required = {"temperature_k", "k", "n"}
    missing = sorted(required - set(parameters.columns))
    if missing:
        raise ValueError(f"parameters table is missing required columns: {missing}")

    frame = parameters.copy()
    frame = frame[
        np.isfinite(frame["temperature_k"])
        & np.isfinite(frame["k"])
        & np.isfinite(frame["n"])
        & (frame["k"] > 0)
    ].sort_values("temperature_k")
    if frame.empty:
        raise ValueError("No finite power-law parameters to plot")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "k_vs_temperature": output_dir / "tau_q_power_law_k_vs_temperature.png",
        "n_vs_temperature": output_dir / "tau_q_power_law_n_vs_temperature.png",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    _errorbar_or_line(
        ax,
        frame["temperature_k"],
        frame["k"],
        frame["k_stderr"] if "k_stderr" in frame else None,
        color="#1f6f8b",
    )
    ax.set_yscale("log")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("k in tau = k Q^-n")
    ax.set_title("Power-law prefactor versus temperature")
    ax.grid(True, alpha=0.25)
    fig.savefig(paths["k_vs_temperature"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    _errorbar_or_line(
        ax,
        frame["temperature_k"],
        frame["n"],
        frame["n_stderr"] if "n_stderr" in frame else None,
        color="#7f3c8d",
    )
    ax.axhline(1.0, color="0.35", linestyle=":", linewidth=1.0, label="n = 1")
    ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.9)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("power-law exponent n")
    ax.set_title("tau-Q exponent versus temperature")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(paths["n_vs_temperature"], dpi=180)
    plt.close(fig)

    return paths


def plot_tau_q_common_n_prefactor(
    parameters: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot k(T) for tau = k(T) * Q^-n with common n."""
    required = {"temperature_k", "k"}
    missing = sorted(required - set(parameters.columns))
    if missing:
        raise ValueError(f"parameters table is missing required columns: {missing}")
    if summary.empty or "common_n" not in summary or "common_n_stderr" not in summary:
        raise ValueError("summary table must include common_n and common_n_stderr")

    frame = parameters.copy()
    frame = frame[
        np.isfinite(frame["temperature_k"])
        & np.isfinite(frame["k"])
        & (frame["k"] > 0)
    ].sort_values("temperature_k")
    if frame.empty:
        raise ValueError("No finite common-n prefactors to plot")

    common_n = float(summary["common_n"].iloc[0])
    common_n_stderr = float(summary["common_n_stderr"].iloc[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "k_vs_temperature": output_dir / "tau_q_common_n_k_vs_temperature.png",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    _errorbar_or_line(
        ax,
        frame["temperature_k"],
        frame["k"],
        frame["k_stderr"] if "k_stderr" in frame else None,
        color="#1f6f8b",
    )
    ax.set_yscale("log")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("k(T) in tau = k(T) Q^-n")
    ax.set_title("Power-law prefactor with common Q exponent")
    if np.isfinite(common_n_stderr):
        annotation = f"common n = {common_n:.3f} +/- {common_n_stderr:.3f}"
    else:
        annotation = f"common n = {common_n:.3f}"
    ax.text(
        0.03,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75"},
    )
    ax.grid(True, alpha=0.25)
    fig.savefig(paths["k_vs_temperature"], dpi=180)
    plt.close(fig)

    return paths
