import numpy as np
import pandas as pd

from xcps_ai.models import (
    FitOptions,
    fit_curve,
    fit_shared_beta_temperature,
    fit_shared_tau_temperature,
    fit_tau_q_power_law_by_temperature,
    fit_tau_q_power_law_common_n,
    kww_model,
)


def test_kww_fit_recovers_synthetic_parameters():
    delay = np.arange(1.0, 200.0)
    g2 = kww_model(delay, baseline=1.0, contrast=0.02, tau_s=55.0, beta=0.8)
    g2e = np.full_like(delay, 0.001)

    result = fit_curve(delay, g2, g2e, FitOptions(model="kww"))

    assert result["success"] is True
    assert abs(float(result["tau_s"]) - 55.0) < 2.0
    assert abs(float(result["beta"]) - 0.8) < 0.05


def test_shared_tau_fit_recovers_one_tau_across_rois(tmp_path):
    delay = np.arange(1.0, 200.0)
    paths = []
    parameters = [(0, 0.02, 0.8), (1, 0.05, 1.4)]
    for roi, contrast, beta in parameters:
        frame = pd.DataFrame(
            {
                "uid": 1,
                "temperature_k": 50,
                "roi": roi,
                "q_index": roi,
                "wait_time": 0,
                "delay_s": delay,
                "g2": kww_model(delay, baseline=1.0, contrast=contrast, tau_s=65.0, beta=beta),
                "g2e": np.full_like(delay, 0.001),
            }
        )
        path = tmp_path / f"roi{roi}_g2.csv"
        frame.to_csv(path, index=False)
        paths.append(path)

    rows = fit_shared_tau_temperature(paths, FitOptions(model="kww"))

    assert len(rows) == 2
    np.testing.assert_allclose([row["tau_s"] for row in rows], [65.0, 65.0], rtol=0.02)
    np.testing.assert_allclose([row["contrast"] for row in rows], [0.02, 0.05], rtol=0.05)
    np.testing.assert_allclose([row["beta"] for row in rows], [0.8, 1.4], rtol=0.05)
    assert {row["shared_tau"] for row in rows} == {True}


def test_shared_beta_fit_recovers_one_beta_across_rois(tmp_path):
    delay = np.arange(1.0, 250.0)
    paths = []
    parameters = [(0, 0.02, 45.0), (1, 0.05, 120.0)]
    for roi, contrast, tau_s in parameters:
        frame = pd.DataFrame(
            {
                "uid": 1,
                "temperature_k": 50,
                "roi": roi,
                "q_index": roi,
                "wait_time": 0,
                "delay_s": delay,
                "g2": kww_model(delay, baseline=1.0, contrast=contrast, tau_s=tau_s, beta=1.35),
                "g2e": np.full_like(delay, 0.001),
            }
        )
        path = tmp_path / f"roi{roi}_g2.csv"
        frame.to_csv(path, index=False)
        paths.append(path)

    rows = fit_shared_beta_temperature(paths, FitOptions(model="kww"))

    assert len(rows) == 2
    np.testing.assert_allclose([row["beta"] for row in rows], [1.35, 1.35], rtol=0.02)
    np.testing.assert_allclose([row["contrast"] for row in rows], [0.02, 0.05], rtol=0.05)
    np.testing.assert_allclose([row["tau_s"] for row in rows], [45.0, 120.0], rtol=0.05)
    assert {row["shared_beta"] for row in rows} == {True}


def test_tau_q_power_law_fit_recovers_k_and_n_by_temperature():
    rows = []
    for temperature, k_value, n_value in [(10, 5.0, 0.4), (50, 12.0, 0.8)]:
        for roi in range(1, 6):
            q = roi * 0.1
            rows.append(
                {
                    "success": True,
                    "temperature_k": temperature,
                    "roi": roi,
                    "tau_s": k_value * q ** (-n_value),
                }
            )
    frame = pd.DataFrame(rows)

    result = fit_tau_q_power_law_by_temperature(
        frame,
        roi_min=1,
        roi_max=5,
        q_step_rlu=0.1,
    )

    assert result["temperature_k"].tolist() == [10.0, 50.0]
    np.testing.assert_allclose(result["k"], np.array([5.0, 12.0]), rtol=1e-12)
    np.testing.assert_allclose(result["n"], np.array([0.4, 0.8]), rtol=1e-12)
    assert (result["n_points"] == 5).all()


def test_tau_q_power_law_common_n_recovers_shared_exponent_and_excludes_temperature():
    rows = []
    common_n = 0.45
    for temperature, k_value in [(10, 5.0), (50, 12.0), (250, 99.0)]:
        for roi in range(1, 6):
            q = roi * 0.1
            rows.append(
                {
                    "success": True,
                    "temperature_k": temperature,
                    "roi": roi,
                    "tau_s": k_value * q ** (-common_n),
                }
            )
    frame = pd.DataFrame(rows)

    parameters, summary = fit_tau_q_power_law_common_n(
        frame,
        roi_min=1,
        roi_max=5,
        q_step_rlu=0.1,
        exclude_temperatures={250},
    )

    assert parameters["temperature_k"].tolist() == [10.0, 50.0]
    np.testing.assert_allclose(parameters["k"], np.array([5.0, 12.0]), rtol=1e-12)
    np.testing.assert_allclose(summary["common_n"].iloc[0], common_n, rtol=1e-12)
    assert summary["excluded_temperatures"].iloc[0] == "250"
