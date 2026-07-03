from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

ModelName = Literal["exp", "kww"]


@dataclass(frozen=True)
class FitOptions:
    model: ModelName = "kww"
    min_delay: float = 1.0
    max_delay: float | None = None
    baseline: float = 1.0


@dataclass(frozen=True)
class _PreparedCurve:
    path: Path
    metadata: dict[str, float | int]
    delay: np.ndarray
    g2: np.ndarray
    sigma: np.ndarray | None
    guess: list[float]


def exp_model(delay: np.ndarray, baseline: float, contrast: float, tau_s: float) -> np.ndarray:
    return baseline + contrast * np.exp(-2.0 * delay / tau_s)


def kww_model(
    delay: np.ndarray,
    baseline: float,
    contrast: float,
    tau_s: float,
    beta: float,
) -> np.ndarray:
    return baseline + contrast * np.exp(-2.0 * np.power(delay / tau_s, beta))


def _guess(delay: np.ndarray, g2: np.ndarray, baseline: float, model: ModelName) -> list[float]:
    contrast = max(float(g2[0] - baseline), 1e-5)
    target = baseline + contrast * np.exp(-2.0)
    tau_s = float(delay[np.nanargmin(np.abs(g2 - target))])
    tau_s = max(tau_s, float(np.nanmin(delay[delay > 0])))
    if model == "exp":
        return [contrast, tau_s]
    return [contrast, tau_s, 1.0]


def _clip_sigma(sigma: np.ndarray | None) -> np.ndarray | None:
    if sigma is not None and np.any(sigma > 0):
        floor = float(np.nanmedian(sigma[sigma > 0]) * 0.05)
        return np.clip(sigma, floor, None)
    return None


def _prepare_curve(path: Path, options: FitOptions) -> _PreparedCurve:
    frame = pd.read_csv(path)
    metadata = frame.iloc[0][["uid", "temperature_k", "roi", "q_index", "wait_time"]].to_dict()
    delay = frame["delay_s"].to_numpy(dtype=float)
    g2 = frame["g2"].to_numpy(dtype=float)
    g2e = frame["g2e"].to_numpy(dtype=float) if "g2e" in frame else None

    mask = np.isfinite(delay) & np.isfinite(g2) & (delay >= options.min_delay)
    if options.max_delay is not None:
        mask &= delay <= options.max_delay
    if g2e is not None:
        mask &= np.isfinite(g2e)

    x = np.asarray(delay[mask], dtype=float)
    y = np.asarray(g2[mask], dtype=float)
    sigma = np.asarray(g2e[mask], dtype=float) if g2e is not None else None
    if x.size < 8:
        raise ValueError(f"Need at least 8 fit points, got {x.size}")

    sigma = _clip_sigma(sigma)
    return _PreparedCurve(
        path=path,
        metadata=metadata,
        delay=x,
        g2=y,
        sigma=sigma,
        guess=_guess(x, y, options.baseline, options.model),
    )


def fit_curve(
    delay: np.ndarray,
    g2: np.ndarray,
    g2e: np.ndarray | None,
    options: FitOptions,
) -> dict[str, float | int | str | bool]:
    mask = np.isfinite(delay) & np.isfinite(g2) & (delay >= options.min_delay)
    if options.max_delay is not None:
        mask &= delay <= options.max_delay
    if g2e is not None:
        mask &= np.isfinite(g2e)

    x = np.asarray(delay[mask], dtype=float)
    y = np.asarray(g2[mask], dtype=float)
    sigma = np.asarray(g2e[mask], dtype=float) if g2e is not None else None
    if x.size < 8:
        raise ValueError(f"Need at least 8 fit points, got {x.size}")

    sigma = _clip_sigma(sigma)

    if options.model == "exp":
        def function(t: np.ndarray, contrast: float, tau_s: float) -> np.ndarray:
            return exp_model(
                t,
                baseline=options.baseline,
                contrast=contrast,
                tau_s=tau_s,
            )
    else:
        def function(
            t: np.ndarray,
            contrast: float,
            tau_s: float,
            beta: float,
        ) -> np.ndarray:
            return kww_model(
                t,
                baseline=options.baseline,
                contrast=contrast,
                tau_s=tau_s,
                beta=beta,
            )

    p0 = _guess(x, y, options.baseline, options.model)
    lower = [0.0, max(float(np.min(x)) * 0.1, 1e-9)]
    upper = [0.2, float(np.max(x)) * 100.0]
    names = ["contrast", "tau_s"]
    if options.model == "kww":
        lower.append(0.1)
        upper.append(3.0)
        names.append("beta")

    popt, pcov = curve_fit(
        function,
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        sigma=sigma,
        absolute_sigma=sigma is not None,
        maxfev=30000,
    )
    yhat = function(x, *popt)
    residual = y - yhat
    dof = max(1, x.size - len(popt))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    stderr = np.sqrt(np.diag(pcov)) if pcov.size else np.full(len(popt), np.nan)

    result: dict[str, float | int | str | bool] = {
        "model": options.model,
        "success": True,
        "message": "",
        "n_points": int(x.size),
        "min_delay_s": float(np.min(x)),
        "max_delay_s": float(np.max(x)),
        "baseline": float(options.baseline),
        "baseline_stderr": 0.0,
        "rmse": float(np.sqrt(ss_res / x.size)),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "reduced_chisq": (
            float(np.sum((residual / sigma) ** 2) / dof) if sigma is not None else np.nan
        ),
    }
    for name, value, err in zip(names, popt, stderr, strict=True):
        result[name] = float(value)
        result[f"{name}_stderr"] = float(err)
    if options.model == "exp":
        result["beta"] = 1.0
        result["beta_stderr"] = 0.0
    return result


def fit_reduced_file(path: Path, options: FitOptions) -> dict[str, float | int | str | bool]:
    frame = pd.read_csv(path)
    metadata = frame.iloc[0][["uid", "temperature_k", "roi", "q_index", "wait_time"]].to_dict()
    try:
        result = fit_curve(
            delay=frame["delay_s"].to_numpy(dtype=float),
            g2=frame["g2"].to_numpy(dtype=float),
            g2e=frame["g2e"].to_numpy(dtype=float),
            options=options,
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "model": options.model,
            "success": False,
            "message": str(exc),
            "n_points": 0,
        }
    result.update(metadata)
    result["reduced_file"] = str(path)
    return result


def _shared_tau_model(
    delay: np.ndarray,
    *,
    baseline: float,
    contrast: float,
    tau_s: float,
    beta: float,
    model: ModelName,
) -> np.ndarray:
    if model == "exp":
        return exp_model(delay, baseline=baseline, contrast=contrast, tau_s=tau_s)
    return kww_model(delay, baseline=baseline, contrast=contrast, tau_s=tau_s, beta=beta)


def _shared_tau_covariance(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    dof: int,
    absolute_sigma: bool,
) -> np.ndarray:
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    if not absolute_sigma:
        covariance *= float(np.sum(residual**2) / max(dof, 1))
    return covariance


def fit_shared_tau_temperature(
    paths: list[Path],
    options: FitOptions,
) -> list[dict[str, float | int | str | bool]]:
    """Fit one shared relaxation time for all reduced ROI curves at one temperature."""
    if not paths:
        return []

    prepared = [_prepare_curve(path, options) for path in sorted(paths)]
    temperatures = {int(curve.metadata["temperature_k"]) for curve in prepared}
    if len(temperatures) != 1:
        raise ValueError(f"Expected one temperature, got {sorted(temperatures)}")

    n_curves = len(prepared)
    tau_guesses = np.asarray([curve.guess[1] for curve in prepared], dtype=float)
    contrast_guesses = np.asarray([curve.guess[0] for curve in prepared], dtype=float)
    beta_guesses = (
        np.asarray([curve.guess[2] for curve in prepared], dtype=float)
        if options.model == "kww"
        else np.ones(n_curves, dtype=float)
    )
    tau_lower = max(min(float(np.min(curve.delay)) for curve in prepared) * 0.1, 1e-9)
    tau_upper = max(float(np.max(curve.delay)) for curve in prepared) * 100.0

    if options.model == "exp":
        p0 = np.concatenate([[float(np.nanmedian(tau_guesses))], contrast_guesses])
        lower = np.concatenate([[tau_lower], np.zeros(n_curves)])
        upper = np.concatenate([[tau_upper], np.full(n_curves, 0.2)])
    else:
        p0 = np.concatenate(
            [[float(np.nanmedian(tau_guesses))], contrast_guesses, beta_guesses]
        )
        lower = np.concatenate([[tau_lower], np.zeros(n_curves), np.full(n_curves, 0.1)])
        upper = np.concatenate([[tau_upper], np.full(n_curves, 0.2), np.full(n_curves, 3.0)])

    p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)

    def unpack(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        tau_s = float(params[0])
        contrasts = params[1 : 1 + n_curves]
        if options.model == "exp":
            betas = np.ones(n_curves, dtype=float)
        else:
            betas = params[1 + n_curves :]
        return tau_s, contrasts, betas

    def residuals(params: np.ndarray) -> np.ndarray:
        tau_s, contrasts, betas = unpack(params)
        pieces: list[np.ndarray] = []
        for index, curve in enumerate(prepared):
            yhat = _shared_tau_model(
                curve.delay,
                baseline=options.baseline,
                contrast=float(contrasts[index]),
                tau_s=tau_s,
                beta=float(betas[index]),
                model=options.model,
            )
            residual = curve.g2 - yhat
            if curve.sigma is not None:
                residual = residual / curve.sigma
            pieces.append(residual)
        return np.concatenate(pieces)

    fit = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        max_nfev=50000,
    )
    tau_s, contrasts, betas = unpack(fit.x)
    total_points = sum(curve.delay.size for curve in prepared)
    dof = max(1, total_points - fit.x.size)
    stderr = np.full(fit.x.size, np.nan, dtype=float)
    try:
        covariance = _shared_tau_covariance(
            fit.jac,
            fit.fun,
            dof=dof,
            absolute_sigma=any(curve.sigma is not None for curve in prepared),
        )
        stderr = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    except np.linalg.LinAlgError:
        pass

    rows: list[dict[str, float | int | str | bool]] = []
    for index, curve in enumerate(prepared):
        yhat = _shared_tau_model(
            curve.delay,
            baseline=options.baseline,
            contrast=float(contrasts[index]),
            tau_s=tau_s,
            beta=float(betas[index]),
            model=options.model,
        )
        residual = curve.g2 - yhat
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((curve.g2 - np.mean(curve.g2)) ** 2))
        curve_dof = max(1, curve.delay.size - (2 if options.model == "exp" else 3))
        if curve.sigma is not None:
            reduced_chisq = float(np.sum((residual / curve.sigma) ** 2) / curve_dof)
        else:
            reduced_chisq = np.nan

        row: dict[str, float | int | str | bool] = {
            "model": options.model,
            "fit_scope": "temperature_shared_tau",
            "shared_tau": True,
            "success": bool(fit.success),
            "message": str(fit.message),
            "n_points": int(curve.delay.size),
            "n_shared_curves": int(n_curves),
            "total_shared_points": int(total_points),
            "min_delay_s": float(np.min(curve.delay)),
            "max_delay_s": float(np.max(curve.delay)),
            "baseline": float(options.baseline),
            "baseline_stderr": 0.0,
            "rmse": float(np.sqrt(ss_res / curve.delay.size)),
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "reduced_chisq": reduced_chisq,
            "contrast": float(contrasts[index]),
            "contrast_stderr": float(stderr[1 + index]),
            "tau_s": tau_s,
            "tau_s_stderr": float(stderr[0]),
            "beta": float(betas[index]),
            "beta_stderr": 0.0
            if options.model == "exp"
            else float(stderr[1 + n_curves + index]),
            "reduced_file": str(curve.path),
        }
        row.update(curve.metadata)
        rows.append(row)

    return rows


def fit_shared_tau_reduced_files(
    paths: list[Path],
    options: FitOptions,
) -> list[dict[str, float | int | str | bool]]:
    """Group reduced files by temperature and run shared-tau fits."""
    grouped: dict[int, list[Path]] = {}
    for path in paths:
        metadata = pd.read_csv(path, nrows=1).iloc[0]
        grouped.setdefault(int(metadata["temperature_k"]), []).append(path)

    rows: list[dict[str, float | int | str | bool]] = []
    for temperature in sorted(grouped):
        rows.extend(fit_shared_tau_temperature(grouped[temperature], options))
    return rows


def fit_shared_beta_temperature(
    paths: list[Path],
    options: FitOptions,
) -> list[dict[str, float | int | str | bool]]:
    """Fit one shared KWW beta for all reduced ROI curves at one temperature."""
    if options.model != "kww":
        raise ValueError("shared-beta fitting requires the kww model")
    if not paths:
        return []

    prepared = [_prepare_curve(path, options) for path in sorted(paths)]
    temperatures = {int(curve.metadata["temperature_k"]) for curve in prepared}
    if len(temperatures) != 1:
        raise ValueError(f"Expected one temperature, got {sorted(temperatures)}")

    n_curves = len(prepared)
    contrast_guesses = np.asarray([curve.guess[0] for curve in prepared], dtype=float)
    tau_guesses = np.asarray([curve.guess[1] for curve in prepared], dtype=float)
    beta_guess = float(np.nanmedian([curve.guess[2] for curve in prepared]))
    tau_lower = np.asarray(
        [max(float(np.min(curve.delay)) * 0.1, 1e-9) for curve in prepared],
        dtype=float,
    )
    tau_upper = np.asarray([float(np.max(curve.delay)) * 100.0 for curve in prepared], dtype=float)

    p0 = np.concatenate([[beta_guess], contrast_guesses, tau_guesses])
    lower = np.concatenate([[0.1], np.zeros(n_curves), tau_lower])
    upper = np.concatenate([[3.0], np.full(n_curves, 0.2), tau_upper])
    p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)

    def unpack(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        beta = float(params[0])
        contrasts = params[1 : 1 + n_curves]
        tau_s = params[1 + n_curves :]
        return beta, contrasts, tau_s

    def residuals(params: np.ndarray) -> np.ndarray:
        beta, contrasts, tau_s = unpack(params)
        pieces: list[np.ndarray] = []
        for index, curve in enumerate(prepared):
            yhat = kww_model(
                curve.delay,
                baseline=options.baseline,
                contrast=float(contrasts[index]),
                tau_s=float(tau_s[index]),
                beta=beta,
            )
            residual = curve.g2 - yhat
            if curve.sigma is not None:
                residual = residual / curve.sigma
            pieces.append(residual)
        return np.concatenate(pieces)

    fit = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        max_nfev=50000,
    )
    beta, contrasts, tau_s = unpack(fit.x)
    total_points = sum(curve.delay.size for curve in prepared)
    dof = max(1, total_points - fit.x.size)
    stderr = np.full(fit.x.size, np.nan, dtype=float)
    try:
        covariance = _shared_tau_covariance(
            fit.jac,
            fit.fun,
            dof=dof,
            absolute_sigma=any(curve.sigma is not None for curve in prepared),
        )
        stderr = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    except np.linalg.LinAlgError:
        pass

    rows: list[dict[str, float | int | str | bool]] = []
    for index, curve in enumerate(prepared):
        yhat = kww_model(
            curve.delay,
            baseline=options.baseline,
            contrast=float(contrasts[index]),
            tau_s=float(tau_s[index]),
            beta=beta,
        )
        residual = curve.g2 - yhat
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((curve.g2 - np.mean(curve.g2)) ** 2))
        curve_dof = max(1, curve.delay.size - 3)
        if curve.sigma is not None:
            reduced_chisq = float(np.sum((residual / curve.sigma) ** 2) / curve_dof)
        else:
            reduced_chisq = np.nan

        row: dict[str, float | int | str | bool] = {
            "model": options.model,
            "fit_scope": "temperature_shared_beta",
            "shared_beta": True,
            "shared_tau": False,
            "success": bool(fit.success),
            "message": str(fit.message),
            "n_points": int(curve.delay.size),
            "n_shared_curves": int(n_curves),
            "total_shared_points": int(total_points),
            "min_delay_s": float(np.min(curve.delay)),
            "max_delay_s": float(np.max(curve.delay)),
            "baseline": float(options.baseline),
            "baseline_stderr": 0.0,
            "rmse": float(np.sqrt(ss_res / curve.delay.size)),
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "reduced_chisq": reduced_chisq,
            "contrast": float(contrasts[index]),
            "contrast_stderr": float(stderr[1 + index]),
            "tau_s": float(tau_s[index]),
            "tau_s_stderr": float(stderr[1 + n_curves + index]),
            "beta": beta,
            "beta_stderr": float(stderr[0]),
            "reduced_file": str(curve.path),
        }
        row.update(curve.metadata)
        rows.append(row)

    return rows


def fit_shared_beta_reduced_files(
    paths: list[Path],
    options: FitOptions,
) -> list[dict[str, float | int | str | bool]]:
    """Group reduced files by temperature and run shared-beta fits."""
    grouped: dict[int, list[Path]] = {}
    for path in paths:
        metadata = pd.read_csv(path, nrows=1).iloc[0]
        grouped.setdefault(int(metadata["temperature_k"]), []).append(path)

    rows: list[dict[str, float | int | str | bool]] = []
    for temperature in sorted(grouped):
        rows.extend(fit_shared_beta_temperature(grouped[temperature], options))
    return rows


def _success_mask(frame: pd.DataFrame) -> pd.Series:
    if "success" not in frame:
        return pd.Series(True, index=frame.index)
    if frame["success"].dtype == bool:
        return frame["success"]
    return frame["success"].astype(str).str.lower().isin({"true", "1", "yes"})


def _tau_q_frame(
    fits: pd.DataFrame,
    *,
    roi_min: int = 1,
    roi_max: int | None = 8,
    q_origin_rlu: float = 0.0,
    q_step_rlu: float = 0.0051,
    exclude_temperatures: set[float] | None = None,
) -> pd.DataFrame:
    required = {"temperature_k", "roi", "tau_s"}
    missing = sorted(required - set(fits.columns))
    if missing:
        raise ValueError(f"fits table is missing required columns: {missing}")
    if q_step_rlu <= 0:
        raise ValueError("q_step_rlu must be positive")
    if roi_min < 0:
        raise ValueError("roi_min must be nonnegative")
    if roi_max is not None and roi_min > roi_max:
        raise ValueError("roi_min must be less than or equal to roi_max")

    frame = fits[_success_mask(fits)].copy()
    frame["roi"] = frame["roi"].astype(int)
    frame["temperature_k"] = frame["temperature_k"].astype(float)
    frame["q_rlu"] = q_origin_rlu + frame["roi"].astype(float) * q_step_rlu

    mask = (
        np.isfinite(frame["temperature_k"])
        & np.isfinite(frame["roi"])
        & np.isfinite(frame["tau_s"])
        & np.isfinite(frame["q_rlu"])
        & (frame["tau_s"] > 0)
        & (frame["q_rlu"] > 0)
        & (frame["roi"] >= roi_min)
    )
    if roi_max is not None:
        mask &= frame["roi"] <= roi_max
    if exclude_temperatures:
        excluded = {float(value) for value in exclude_temperatures}
        mask &= ~frame["temperature_k"].isin(excluded)
    frame = frame[mask].copy()
    if frame.empty:
        raise ValueError("No positive-Q tau rows remain after filtering")
    return frame


def fit_tau_q_power_law_by_temperature(
    fits: pd.DataFrame,
    *,
    roi_min: int = 1,
    roi_max: int | None = 8,
    q_origin_rlu: float = 0.0,
    q_step_rlu: float = 0.0051,
) -> pd.DataFrame:
    """Fit tau = k * Q^-n independently at each temperature.

    The fit is an unweighted ordinary least-squares fit in log space:
    log(tau) = log(k) - n * log(Q). Q is computed from the ROI index.
    """
    frame = _tau_q_frame(
        fits,
        roi_min=roi_min,
        roi_max=roi_max,
        q_origin_rlu=q_origin_rlu,
        q_step_rlu=q_step_rlu,
    )

    rows: list[dict[str, float | int | str]] = []
    for temperature, group in frame.groupby("temperature_k", sort=True):
        group = group.sort_values("q_rlu")
        if len(group) < 2:
            raise ValueError(f"Need at least 2 positive-Q rows at {temperature:g} K")

        log_q = np.log(group["q_rlu"].to_numpy(dtype=float))
        log_tau = np.log(group["tau_s"].to_numpy(dtype=float))
        design = np.column_stack([np.ones_like(log_q), log_q])
        intercept, slope = np.linalg.lstsq(design, log_tau, rcond=None)[0]
        predicted = design @ np.asarray([intercept, slope])
        residual = log_tau - predicted
        rss = float(np.sum(residual**2))
        tss = float(np.sum((log_tau - np.mean(log_tau)) ** 2))
        dof = len(group) - 2

        intercept_stderr = np.nan
        slope_stderr = np.nan
        if dof > 0:
            try:
                covariance = (rss / dof) * np.linalg.inv(design.T @ design)
                stderr = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
                intercept_stderr = float(stderr[0])
                slope_stderr = float(stderr[1])
            except np.linalg.LinAlgError:
                pass

        k = float(np.exp(intercept))
        n = float(-slope)
        rows.append(
            {
                "fit_scope": "temperature_tau_q_power_law",
                "temperature_k": float(temperature),
                "n_points": int(len(group)),
                "roi_min": int(group["roi"].min()),
                "roi_max": int(group["roi"].max()),
                "q_min_rlu": float(group["q_rlu"].min()),
                "q_max_rlu": float(group["q_rlu"].max()),
                "q_origin_rlu": float(q_origin_rlu),
                "q_step_rlu": float(q_step_rlu),
                "k": k,
                "k_stderr": float(k * intercept_stderr)
                if np.isfinite(intercept_stderr)
                else np.nan,
                "n": n,
                "n_stderr": slope_stderr,
                "intercept_log_k": float(intercept),
                "slope_log_tau_vs_log_q": float(slope),
                "rmse_log_tau": float(np.sqrt(rss / len(group))),
                "r_squared_log_tau": float(1.0 - rss / tss) if tss > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def fit_tau_q_power_law_common_n(
    fits: pd.DataFrame,
    *,
    roi_min: int = 1,
    roi_max: int | None = 8,
    q_origin_rlu: float = 0.0,
    q_step_rlu: float = 0.0051,
    exclude_temperatures: set[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit tau = k(T) * Q^-n with one common n across temperatures."""
    frame = _tau_q_frame(
        fits,
        roi_min=roi_min,
        roi_max=roi_max,
        q_origin_rlu=q_origin_rlu,
        q_step_rlu=q_step_rlu,
        exclude_temperatures=exclude_temperatures,
    )
    temperatures = np.asarray(sorted(frame["temperature_k"].unique()), dtype=float)
    if temperatures.size < 1:
        raise ValueError("Need at least one temperature")

    temperature_to_index = {temperature: index for index, temperature in enumerate(temperatures)}
    log_q = np.log(frame["q_rlu"].to_numpy(dtype=float))
    log_tau = np.log(frame["tau_s"].to_numpy(dtype=float))

    design = np.zeros((len(frame), temperatures.size + 1), dtype=float)
    for row_index, temperature in enumerate(frame["temperature_k"].to_numpy(dtype=float)):
        design[row_index, temperature_to_index[temperature]] = 1.0
    design[:, -1] = log_q

    coefficients = np.linalg.lstsq(design, log_tau, rcond=None)[0]
    predicted = design @ coefficients
    residual = log_tau - predicted
    rss = float(np.sum(residual**2))
    tss = float(np.sum((log_tau - np.mean(log_tau)) ** 2))
    dof = len(frame) - design.shape[1]

    covariance = np.full((design.shape[1], design.shape[1]), np.nan, dtype=float)
    if dof > 0:
        covariance = (rss / dof) * np.linalg.pinv(design.T @ design)
    stderr = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    common_n = float(-coefficients[-1])
    common_n_stderr = float(stderr[-1]) if np.isfinite(stderr[-1]) else np.nan
    excluded = sorted(exclude_temperatures or [])

    rows: list[dict[str, float | int | str]] = []
    for index, temperature in enumerate(temperatures):
        group = frame[frame["temperature_k"] == temperature]
        intercept = float(coefficients[index])
        k = float(np.exp(intercept))
        intercept_stderr = float(stderr[index]) if np.isfinite(stderr[index]) else np.nan
        rows.append(
            {
                "fit_scope": "temperature_tau_q_power_law_common_n",
                "temperature_k": float(temperature),
                "n_points": int(len(group)),
                "roi_min": int(group["roi"].min()),
                "roi_max": int(group["roi"].max()),
                "q_min_rlu": float(group["q_rlu"].min()),
                "q_max_rlu": float(group["q_rlu"].max()),
                "q_origin_rlu": float(q_origin_rlu),
                "q_step_rlu": float(q_step_rlu),
                "k": k,
                "k_stderr": float(k * intercept_stderr)
                if np.isfinite(intercept_stderr)
                else np.nan,
                "common_n": common_n,
                "common_n_stderr": common_n_stderr,
                "intercept_log_k": intercept,
            }
        )

    summary = pd.DataFrame(
        [
            {
                "fit_scope": "tau_q_power_law_common_n",
                "n_temperatures": int(temperatures.size),
                "n_points": int(len(frame)),
                "n_parameters": int(design.shape[1]),
                "roi_min": int(frame["roi"].min()),
                "roi_max": int(frame["roi"].max()),
                "q_min_rlu": float(frame["q_rlu"].min()),
                "q_max_rlu": float(frame["q_rlu"].max()),
                "q_origin_rlu": float(q_origin_rlu),
                "q_step_rlu": float(q_step_rlu),
                "excluded_temperatures": ",".join(f"{value:g}" for value in excluded),
                "common_n": common_n,
                "common_n_stderr": common_n_stderr,
                "slope_log_tau_vs_log_q": float(coefficients[-1]),
                "rss_log_tau": rss,
                "rmse_log_tau": float(np.sqrt(rss / len(frame))),
                "r_squared_log_tau": float(1.0 - rss / tss) if tss > 0 else np.nan,
            }
        ]
    )
    return pd.DataFrame(rows), summary


def summarize_by_temperature(fits: pd.DataFrame) -> pd.DataFrame:
    good = fits[fits["success"] == True].copy()  # noqa: E712
    if good.empty:
        return pd.DataFrame()
    columns = ["tau_s", "beta", "contrast", "baseline", "r_squared", "reduced_chisq"]
    summary = (
        good.groupby("temperature_k")[columns]
        .agg(["count", "median", "mean", "std"])
        .sort_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary.reset_index()
