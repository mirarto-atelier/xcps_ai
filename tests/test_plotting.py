from __future__ import annotations

import numpy as np
import pandas as pd

from xcps_ai.plotting import (
    HBAR_EV_S,
    IsfGrid,
    contrast_signature_tables,
    filter_grids_by_roi,
    intermediate_scattering_function,
    plot_tau_q_common_n_prefactor,
    plot_tau_q_power_law_parameters,
    roi_to_q_rlu,
    spin_signature_tables,
    tau_to_energy,
)


def test_intermediate_scattering_function_uses_contrast_and_baseline() -> None:
    g2 = np.array([1.04, 1.01, 0.99])
    result = intermediate_scattering_function(g2, contrast=0.04, baseline=1.0)

    np.testing.assert_allclose(result, np.array([1.0, 0.5, 0.0]))


def test_intermediate_scattering_function_rejects_bad_contrast() -> None:
    g2 = np.array([1.04, 1.01])
    result = intermediate_scattering_function(g2, contrast=0.0, baseline=1.0)

    assert np.isnan(result).all()


def test_roi_to_q_rlu_uses_linear_step_from_zero() -> None:
    roi = np.array([0, 1, 8])
    result = roi_to_q_rlu(roi, q_origin_rlu=0.0, q_step_rlu=0.0051)

    np.testing.assert_allclose(result, np.array([0.0, 0.0051, 0.0408]))


def test_tau_to_energy_uses_hbar_over_tau() -> None:
    delay_s = np.array([0.0, 1.0, 2.0])
    result = tau_to_energy(delay_s, unit="feV")

    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1:], np.array([HBAR_EV_S, HBAR_EV_S / 2.0]) / 1e-15)


def test_filter_grids_by_roi_keeps_requested_range() -> None:
    grid = IsfGrid(
        temperature_k=10.0,
        roi=np.array([0, 1, 8, 9, 10]),
        delay_s=np.array([1.0, 2.0]),
        isf=np.arange(10, dtype=float).reshape(2, 5),
    )

    result = filter_grids_by_roi([grid], roi_max=8)

    np.testing.assert_array_equal(result[0].roi, np.array([0, 1, 8]))
    np.testing.assert_array_equal(result[0].isf, np.array([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]]))


def test_spin_signature_tables_compute_tau_ratios() -> None:
    frame = pd.DataFrame(
        {
            "temperature_k": [10, 39, 50, 10, 39, 50],
            "roi": [0, 0, 0, 1, 1, 1],
            "q_rlu": [0.0, 0.0, 0.0, 0.0051, 0.0051, 0.0051],
            "tau_s": [20.0, 10.0, 50.0, 30.0, 15.0, 60.0],
            "beta": [1.1, 1.8, 1.4, 1.2, 1.9, 1.5],
            "contrast": [0.01, 0.02, 0.02, 0.01, 0.02, 0.02],
            "r_squared": [0.9, 0.95, 0.95, 0.9, 0.95, 0.95],
        }
    )

    summary, ratio = spin_signature_tables(frame)

    assert summary.loc[summary["temperature_k"] == 39, "tau_median_s"].iloc[0] == 12.5
    np.testing.assert_allclose(ratio["tau39_over_tau50"], np.array([0.2, 0.25]))
    np.testing.assert_allclose(ratio["tau10_over_tau39"], np.array([2.0, 2.0]))


def test_contrast_signature_tables_compute_contrast_ratios() -> None:
    frame = pd.DataFrame(
        {
            "temperature_k": [10, 39, 50, 10, 39, 50],
            "roi": [0, 0, 0, 1, 1, 1],
            "q_rlu": [0.0, 0.0, 0.0, 0.0051, 0.0051, 0.0051],
            "tau_s": [20.0, 10.0, 50.0, 30.0, 15.0, 60.0],
            "beta": [1.1, 1.8, 1.4, 1.2, 1.9, 1.5],
            "contrast": [0.01, 0.02, 0.01, 0.03, 0.06, 0.02],
            "r_squared": [0.9, 0.95, 0.95, 0.9, 0.95, 0.95],
        }
    )

    summary, contrast_by_q = contrast_signature_tables(frame)

    assert summary.loc[summary["temperature_k"] == 39, "contrast_median"].iloc[0] == 0.04
    np.testing.assert_allclose(
        contrast_by_q["contrast39_over_contrast50"],
        np.array([2.0, 3.0]),
    )
    np.testing.assert_allclose(
        contrast_by_q["contrast10_over_contrast39"],
        np.array([0.5, 0.5]),
    )


def test_plot_tau_q_power_law_parameters_writes_separate_figures(tmp_path) -> None:
    parameters = pd.DataFrame(
        {
            "temperature_k": [10, 50],
            "k": [5.0, 12.0],
            "k_stderr": [0.2, 0.4],
            "n": [0.4, 0.8],
            "n_stderr": [0.03, 0.05],
        }
    )

    paths = plot_tau_q_power_law_parameters(parameters, tmp_path)

    assert paths["k_vs_temperature"].exists()
    assert paths["n_vs_temperature"].exists()


def test_plot_tau_q_common_n_prefactor_writes_k_figure(tmp_path) -> None:
    parameters = pd.DataFrame(
        {
            "temperature_k": [10, 50],
            "k": [5.0, 12.0],
            "k_stderr": [0.2, 0.4],
        }
    )
    summary = pd.DataFrame({"common_n": [0.45], "common_n_stderr": [0.03]})

    paths = plot_tau_q_common_n_prefactor(parameters, summary, tmp_path)

    assert paths["k_vs_temperature"].exists()
