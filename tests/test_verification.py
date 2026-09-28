import numpy as np

from india_rainfall_postprocessing.verification import csi, ets, far, pod, rmse


def test_rmse_calculation():
    forecast = np.array([1.0, 2.0, 3.0])
    observed = np.array([1.0, 4.0, 3.0])
    assert abs(rmse(forecast, observed) - np.sqrt(4.0 / 3.0)) < 1e-8


def test_pod_far_csi_ets():
    forecast = np.array([1, 0, 1, 1])
    observed = np.array([1, 1, 0, 1])
    threshold = 0.5

    assert abs(pod(forecast, observed, threshold) - 0.6666666667) < 1e-8
    assert abs(far(forecast, observed, threshold) - 0.3333333333) < 1e-8
    assert abs(csi(forecast, observed, threshold) - 0.5) < 1e-8
    assert abs(ets(forecast, observed, threshold) + 0.14285714285714285) < 1e-8


def test_binary_event_edge_cases():
    forecast = np.array([0, 0, 0])
    observed = np.array([0, 0, 0])
    assert np.isnan(pod(forecast, observed, 0.5))
    assert np.isnan(far(forecast, observed, 0.5))
    assert np.isnan(csi(forecast, observed, 0.5))
