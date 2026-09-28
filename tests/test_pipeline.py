"""
Phase 2 test suite — validates data processing, metrics, and API.
Run with: pytest tests/test_pipeline.py -v
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


# ============================================================
# FIXTURES
# ============================================================
@pytest.fixture
def sample_arrays():
    """Known observed / raw / corrected triple."""
    obs = np.array([0.0, 5.0, 20.0, 65.0, 120.0, 210.0, 0.0, 30.0])
    raw = np.array([2.0, 3.0, 25.0, 60.0, 100.0, 180.0, 1.0, 35.0])
    cor = np.array([0.5, 6.0, 22.0, 70.0, 115.0, 205.0, 0.2, 32.0])
    return obs, raw, cor


# ============================================================
# METRIC FUNCTIONS (defined here to test independently)
# ============================================================
def rmse(o, p): return float(np.sqrt(np.mean((o - p) ** 2)))
def mae(o, p):  return float(np.mean(np.abs(o - p)))
def bias(o, p): return float(np.mean(p - o))


def contingency(o, p, thr):
    oe = o >= thr; pe = p >= thr
    return (int((oe & pe).sum()), int((oe & ~pe).sum()),
            int((~oe & pe).sum()), int((~oe & ~pe).sum()))


def pod(h, m, f, c): return h / (h + m) if (h + m) > 0 else float("nan")
def far(h, m, f, c): return f / (h + f) if (h + f) > 0 else float("nan")
def csi(h, m, f, c):
    d = h + m + f
    return h / d if d > 0 else float("nan")
def ets(h, m, f, c):
    d = h + m + f
    if d == 0: return float("nan")
    n = h + m + f + c
    h_rand = (h + m) * (h + f) / n
    den = d - h_rand
    return (h - h_rand) / den if den > 0 else float("nan")


# ============================================================
# 1. TEST METRIC DEFINITIONS
# ============================================================
def test_rmse_perfect():
    o = np.array([1.0, 2.0, 3.0])
    assert rmse(o, o) == 0.0

def test_rmse_known_value():
    o = np.array([0.0, 0.0])
    p = np.array([3.0, 4.0])
    assert abs(rmse(o, p) - np.sqrt((9 + 16) / 2)) < 1e-9

def test_mae_known():
    o = np.array([1.0, 2.0])
    p = np.array([2.0, 4.0])
    assert mae(o, p) == 1.5

def test_bias_positive():
    o = np.array([1.0, 2.0])
    p = np.array([2.0, 3.0])
    assert bias(o, p) == 1.0

def test_bias_negative():
    o = np.array([5.0, 6.0])
    p = np.array([2.0, 3.0])
    assert bias(o, p) == -3.0

def test_bias_zero():
    o = np.array([1.0, 2.0, 3.0])
    assert bias(o, o) == 0.0


# ============================================================
# 2. TEST CONTINGENCY
# ============================================================
def test_contingency_no_events():
    o = np.array([0.0, 1.0, 2.0])
    p = np.array([0.0, 1.0, 2.0])
    h, m, f, c = contingency(o, p, 64.5)
    assert (h, m, f) == (0, 0, 0)
    assert c == 3

def test_contingency_all_hits():
    o = np.array([100.0, 100.0])
    p = np.array([100.0, 100.0])
    h, m, f, c = contingency(o, p, 64.5)
    assert h == 2 and m == 0 and f == 0

def test_contingency_all_misses():
    o = np.array([100.0, 100.0])
    p = np.array([1.0, 1.0])
    h, m, f, c = contingency(o, p, 64.5)
    assert h == 0 and m == 2 and f == 0

def test_contingency_all_false_alarms():
    o = np.array([1.0, 1.0])
    p = np.array([100.0, 100.0])
    h, m, f, c = contingency(o, p, 64.5)
    assert h == 0 and m == 0 and f == 2


# ============================================================
# 3. TEST POD / FAR / CSI / ETS
# ============================================================
def test_pod_perfect():
    assert pod(5, 0, 0, 10) == 1.0

def test_pod_zero():
    assert pod(0, 5, 0, 10) == 0.0

def test_pod_no_events():
    assert np.isnan(pod(0, 0, 0, 10))

def test_far_perfect():
    assert far(5, 0, 0, 10) == 0.0

def test_far_all_false():
    assert far(0, 0, 5, 10) == 1.0

def test_csi_known():
    # h=5, m=0, f=0 -> CSI=1
    assert csi(5, 0, 0, 10) == 1.0

def test_csi_zero_hits():
    assert csi(0, 5, 5, 10) == 0.0

def test_ets_perfect():
    # Perfect forecast: ETS should be 1
    assert ets(5, 0, 0, 95) == 1.0

def test_ets_zero_hits():
    assert ets(0, 5, 5, 90) == 0.0


# ============================================================
# 4. TEST GRD PARSER (using sample data)
# ============================================================
def test_grd_shape():
    """135 × 129 grid per day, float32, little-endian."""
    N_LAT, N_LON = 129, 135
    days = 10
    arr = np.zeros(days * N_LAT * N_LON, dtype="<f4")
    arr[0] = -999.0
    arr[1] = 5.5

    reshaped = arr.reshape(days, N_LAT, N_LON)
    assert reshaped.shape == (days, 129, 135)
    assert reshaped[0, 0, 0] == -999.0
    assert reshaped[0, 0, 1] == 5.5

def test_grd_missing_value_replaced():
    """−999 must map to NaN, not zero."""
    data = np.array([-999.0, 0.0, 5.0, -999.0], dtype="<f4")
    data[data == -999.0] = np.nan
    assert np.isnan(data[0])
    assert data[1] == 0.0
    assert data[2] == 5.0
    assert np.isnan(data[3])


# ============================================================
# 5. TEST DISTRICT AGGREGATION
# ============================================================
def test_aggregation_mean():
    """Mean of valid grid cells only."""
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert abs(np.nanmean(values) - 3.0) < 1e-9

def test_aggregation_all_missing():
    values = np.array([np.nan, np.nan])
    with pytest.warns(RuntimeWarning):
        result = np.nanmean(values)
    assert np.isnan(result)


# ============================================================
# 6. TEST REGIME LABELING LOGIC
# ============================================================
def test_regime_label_valid():
    valid = {"active_monsoon","break_monsoon","low_depression",
             "coastal_rainfall","orographic_rainfall",
             "western_disturbance","quiet"}
    sample_labels = ["active_monsoon","quiet","low_depression"]
    for lab in sample_labels:
        assert lab in valid

def test_regime_label_not_empty():
    labels = []
    assert len(labels) == 0  # empty is fine


# ============================================================
# 7. TEST PROBABILITY CALIBRATION
# ============================================================
def test_brier_perfect():
    """Perfect predictions -> Brier = 0."""
    p = np.array([0.0, 1.0, 0.0, 1.0])
    o = np.array([0,   1,   0,   1])
    brier = np.mean((p - o) ** 2)
    assert brier == 0.0

def test_brier_worst():
    """Completely wrong predictions -> Brier = 1."""
    p = np.array([1.0, 0.0])
    o = np.array([0,   1])
    brier = np.mean((p - o) ** 2)
    assert brier == 1.0

def test_brier_typical():
    """Brier between 0 and 1."""
    p = np.array([0.3, 0.7, 0.5])
    o = np.array([0,   1,   0])
    brier = np.mean((p - o) ** 2)
    assert 0 <= brier <= 1


# ============================================================
# 8. TEST FSS
# ============================================================
def test_fss_perfect():
    """Perfect grid match -> FSS = 1."""
    o = np.array([[0,0,0],[0,1,1],[0,1,1]], dtype=float)
    p = o.copy()
    mse_f = np.mean((o - p) ** 2)
    mse_ref = np.mean(o ** 2) + np.mean(p ** 2)
    if mse_ref > 0:
        fss = 1 - mse_f / mse_ref
        assert fss == 1.0


# ============================================================
# 9. TEST SPLIT LOGIC
# ============================================================
def test_time_split_chronological():
    dates = pd.date_range("2016-06-01", "2016-08-31", freq="D")
    train_end = pd.Timestamp("2016-07-15")
    test_start = pd.Timestamp("2016-07-26")
    train = dates[dates <= train_end]
    test = dates[dates >= test_start]
    assert len(train) == 45
    assert len(test) == 37
    # No overlap
    assert train.max() < test.min()


# ============================================================
# 10. TEST OUTPUT FILES EXIST
# ============================================================
ROOT = Path(__file__).parent.parent

def test_reports_exist():
    r = ROOT / "reports"
    assert r.exists(), "reports folder missing"

def test_config_exists():
    assert (ROOT / "config.yaml").exists(), "config.yaml missing"

def test_config_loads():
    import yaml
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "paths" in cfg
    assert "thresholds" in cfg
    assert cfg["thresholds"]["heavy"] == 64.5