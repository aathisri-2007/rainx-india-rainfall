"""
Compute all scientific metrics for the dashboard Analytics page.
Writes reports/analytics.json.
Self-contained: reads config.yaml directly.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

BASE = Path(__file__).parent
with open(BASE / "config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

def cg(path, default=None):
    parts = path.split(".")
    cur = CFG
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

IMD_DISTRICTS  = Path(cg("paths.imd_districts"))
REGIME_LABELS  = Path(cg("paths.regimes_dir")) / "regime_labels_2016.csv"
MODELS_DIR     = Path(cg("paths.models_dir"))
REPORTS_DIR    = Path(cg("paths.reports_dir"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {
    "heavy":           float(cg("thresholds.heavy", 64.5)),
    "very_heavy":      float(cg("thresholds.very_heavy", 115.6)),
    "extremely_heavy": float(cg("thresholds.extremely_heavy", 204.5)),
}
FSS_NEIGHBORHOODS = cg("spatial.fss_neighborhoods", [1, 3, 5, 7, 9])

print("=" * 60)
print("COMPUTING ANALYTICS")
print("=" * 60)

out = {}

# ============================================================
# 1. IMD data quality (whole dataset)
# ============================================================
print("Loading IMD district rainfall...")
imd = pd.read_csv(IMD_DISTRICTS, usecols=["date", "district_id", "rainfall_mm"])
out["data_quality"] = {
    "rows":                    int(len(imd)),
    "unique_dates":            int(imd["date"].nunique()),
    "unique_districts":        int(imd["district_id"].nunique()),
    "missing_rainfall":        int(imd["rainfall_mm"].isna().sum()),
    "missing_percent":         round(float(imd["rainfall_mm"].isna().mean() * 100), 4),
    "negative_rainfall":       int((imd["rainfall_mm"] < 0).sum()),
    "zero_rainfall":           int((imd["rainfall_mm"] == 0).sum()),
    "duplicate_date_district": int(imd.duplicated(["date","district_id"]).sum()),
    "date_min":                str(imd["date"].min()),
    "date_max":                str(imd["date"].max()),
    "rainfall_mean":           round(float(imd["rainfall_mm"].mean()), 3),
    "rainfall_std":            round(float(imd["rainfall_mm"].std()), 3),
}
print("  data_quality done")

# ============================================================
# 2. Test set analytics
# ============================================================
test_file = MODELS_DIR / "test_predictions.csv"
if not test_file.exists():
    raise SystemExit("test_predictions.csv not found — run train_postprocessor.py first.")

test = pd.read_csv(test_file, parse_dates=["date"])

def full_stats(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"n":0}
    return {
        "n":     int(len(s)),
        "mean":  round(float(s.mean()), 3),
        "total": round(float(s.sum()), 2),
        "min":   round(float(s.min()), 3),
        "max":   round(float(s.max()), 3),
        "std":   round(float(s.std()), 3),
        "p50":   round(float(s.quantile(0.50)), 3),
        "p75":   round(float(s.quantile(0.75)), 3),
        "p90":   round(float(s.quantile(0.90)), 3),
        "p95":   round(float(s.quantile(0.95)), 3),
        "p99":   round(float(s.quantile(0.99)), 3),
    }

out["basic_stats"] = {
    "observed":  full_stats(test["rainfall_mm"]),
    "raw_nwp":   full_stats(test["gefs_mm"]),
    "corrected": full_stats(test["pred_regime_feat"]),
}
print("  basic_stats done")

# ============================================================
# 3. Error metrics
# ============================================================
def err_metrics(obs_s, pred_s):
    o = pd.to_numeric(obs_s, errors="coerce").values
    p = pd.to_numeric(pred_s, errors="coerce").values
    m = ~(np.isnan(o) | np.isnan(p))
    o, p = o[m], p[m]
    if len(o) == 0:
        return {"n":0}
    e = p - o
    dry = o > 0.1
    mape = float(np.mean(np.abs(e[dry] / o[dry])) * 100) if dry.sum() > 0 else None
    return {
        "n":     int(len(e)),
        "bias":  round(float(np.mean(e)), 3),
        "mae":   round(float(np.mean(np.abs(e))), 3),
        "rmse":  round(float(np.sqrt(np.mean(e**2))), 3),
        "mape":  round(mape, 2) if mape is not None else None,
        "p95_abs_error": round(float(np.quantile(np.abs(e), 0.95)), 3),
        "max_abs_error": round(float(np.max(np.abs(e))), 3),
    }

raw = err_metrics(test["rainfall_mm"], test["gefs_mm"])
cor = err_metrics(test["rainfall_mm"], test["pred_regime_feat"])
out["error_metrics"] = {"raw": raw, "corrected": cor}

def pct_change(b, a):
    if b in (None, 0) or a is None: return None
    return round((b - a) / abs(b) * 100, 2)

out["improvement"] = {
    "rmse_pct":       pct_change(raw.get("rmse"), cor.get("rmse")),
    "mae_pct":        pct_change(raw.get("mae"),  cor.get("mae")),
    "bias_reduction": pct_change(abs(raw.get("bias", 0)), abs(cor.get("bias", 0))),
}
print("  error_metrics + improvement done")

# ============================================================
# 4. Correction stats
# ============================================================
test["correction"] = test["pred_regime_feat"] - test["gefs_mm"]
valid = test[test["gefs_mm"] > 0.1].copy()
valid["correction_pct"] = valid["correction"] / valid["gefs_mm"] * 100

out["correction"] = {
    "mean_mm":           round(float(test["correction"].mean()), 3),
    "median_mm":         round(float(test["correction"].median()), 3),
    "std_mm":            round(float(test["correction"].std()), 3),
    "mean_pct":          round(float(valid["correction_pct"].mean()), 2) if len(valid) else None,
    "positive_count":    int((test["correction"] > 0).sum()),
    "negative_count":    int((test["correction"] < 0).sum()),
    "unchanged_count":   int((test["correction"].abs() < 0.01).sum()),
    "max_correction_mm": round(float(test["correction"].max()), 2),
    "min_correction_mm": round(float(test["correction"].min()), 2),
}
print("  correction done")

# ============================================================
# 5. Categories
# ============================================================
cat = {}
for name, thr in THRESHOLDS.items():
    cat[name] = {
        "threshold_mm": thr,
        "observed":     int((test["rainfall_mm"] >= thr).sum()),
        "raw_nwp":      int((test["gefs_mm"] >= thr).sum()),
        "corrected":    int((test["pred_regime_feat"] >= thr).sum()),
    }
out["categories"] = cat
print("  categories done")

# ============================================================
# 6. Regime stats
# ============================================================
regime_stats = []
for regime in sorted(test["regime_label"].dropna().unique()):
    sub = test[test["regime_label"] == regime]
    if len(sub) == 0: continue
    regime_stats.append({
        "regime":    regime,
        "n":         int(len(sub)),
        "obs_mean":  round(float(sub["rainfall_mm"].mean()), 2),
        "raw_bias":  round(float((sub["gefs_mm"] - sub["rainfall_mm"]).mean()), 3),
        "corr_bias": round(float((sub["pred_regime_feat"] - sub["rainfall_mm"]).mean()), 3),
        "raw_rmse":  round(float(np.sqrt(np.mean((sub["gefs_mm"] - sub["rainfall_mm"])**2))), 3),
        "corr_rmse": round(float(np.sqrt(np.mean((sub["pred_regime_feat"] - sub["rainfall_mm"])**2))), 3),
    })
out["regime_stats"] = regime_stats
print(f"  regime_stats done ({len(regime_stats)} regimes)")

# ============================================================
# 7. Probability calibration (Brier + reliability)
# ============================================================
prob_file = MODELS_DIR / "test_predictions_with_prob.csv"
cal = {}
if prob_file.exists():
    prob = pd.read_csv(prob_file)
    for pcol in ["P_heavy","P_very_heavy","P_extremely_heavy"]:
        if pcol not in prob.columns: continue
        thr_key = pcol.replace("P_","")
        thr = THRESHOLDS[thr_key]
        obs_event = (prob["rainfall_mm"] >= thr).astype(int).values
        pvals = prob[pcol].values
        brier = round(float(np.mean((pvals - obs_event)**2)), 5)
        edges = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        bins = []
        for i in range(len(edges)-1):
            lo, hi = edges[i], edges[i+1]
            if i == len(edges)-2:
                m = (pvals >= lo) & (pvals <= hi)
            else:
                m = (pvals >= lo) & (pvals < hi)
            n = int(m.sum())
            bins.append({
                "bin":            f"{lo:.1f}-{hi:.1f}",
                "n":              n,
                "predicted_mean": round(float(pvals[m].mean()), 3) if n else None,
                "observed_freq":  round(float(obs_event[m].mean()), 3) if n else None,
            })
        cal[pcol] = {"threshold_mm": thr, "brier": brier, "reliability": bins}
out["probability_calibration"] = cal
print(f"  probability_calibration done ({len(cal)} thresholds)")

# ============================================================
# 8. FSS by neighborhood (uses district centroids from mapping)
# ============================================================
out["fss_by_neighborhood"] = {}

# Build district centroid table from grid mapping
try:
    MAPPING_CSV = Path(cg("paths.grid_mapping"))
    if MAPPING_CSV.exists():
        m = pd.read_csv(MAPPING_CSV).dropna(subset=["district_id"]).copy()
        m["district_id"] = m["district_id"].astype(int).astype(str)
        centroids = m.groupby("district_id")[["latitude", "longitude"]].mean().reset_index()
        cdict = {row["district_id"]: (row["latitude"], row["longitude"])
                 for _, row in centroids.iterrows()}

        test["district_id"] = test["district_id"].astype(str)
        test["lat_c"] = test["district_id"].map(lambda x: cdict.get(x, (np.nan, np.nan))[0])
        test["lon_c"] = test["district_id"].map(lambda x: cdict.get(x, (np.nan, np.nan))[1])

        # Snap to 0.25° grid to allow 2D array building
        test["lat_r"] = (test["lat_c"] / 0.25).round() * 0.25
        test["lon_r"] = (test["lon_c"] / 0.25).round() * 0.25

        # Only keep days where all districts are present
        from scipy.signal import convolve2d

        def fss_at_scale(window, thr):
            scores = []
            for date, g in test.dropna(subset=["lat_r", "lon_r"]).groupby("date"):
                grid_obs  = g.pivot_table(index="lat_r", columns="lon_r",
                                            values="rainfall_mm", aggfunc="mean")
                grid_pred = g.pivot_table(index="lat_r", columns="lon_r",
                                            values="pred_regime_feat", aggfunc="mean")
                # Align shapes
                idx = grid_obs.index.union(grid_pred.index)
                cols = grid_obs.columns.union(grid_pred.columns)
                grid_obs  = grid_obs.reindex(index=idx, columns=cols)
                grid_pred = grid_pred.reindex(index=idx, columns=cols)

                o_ev = np.nan_to_num((grid_obs.values  >= thr).astype(float))
                p_ev = np.nan_to_num((grid_pred.values >= thr).astype(float))
                if o_ev.size == 0: continue

                def box(a, w):
                    if w <= 1: return a
                    pad = w // 2
                    ap = np.pad(a, pad, mode="edge")
                    k = np.ones((w, w)) / (w * w)
                    out = convolve2d(ap, k, mode="valid")
                    return out[:a.shape[0], :a.shape[1]]

                o_s = box(o_ev, window)
                p_s = box(p_ev, window)
                mse_f   = np.mean((o_s - p_s) ** 2)
                mse_ref = np.mean(o_s ** 2) + np.mean(p_s ** 2)
                if mse_ref > 0:
                    scores.append(1 - mse_f / mse_ref)

            if not scores: return None
            return round(float(np.mean(scores)), 4)

        for name, thr in THRESHOLDS.items():
            rows = []
            for w in FSS_NEIGHBORHOODS:
                rows.append({"neighborhood": w, "fss": fss_at_scale(w, thr)})
            out["fss_by_neighborhood"][name] = {"threshold_mm": thr, "values": rows}
        print("  fss_by_neighborhood done (district-centroid grid)")
    else:
        print(f"  fss skipped: mapping file not found at {MAPPING_CSV}")
except Exception as e:
    print(f"  fss failed: {e}")
# ============================================================
# 9. Model metadata (from sidecar JSON)
# ============================================================
meta_file = MODELS_DIR / "regime_classifier.json"
if meta_file.exists():
    with open(meta_file, encoding="utf-8") as f:
        out["model_metadata"] = json.load(f)
    print("  model_metadata done")

# ============================================================
# 10. Save
# ============================================================
out["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()

with open(REPORTS_DIR / "analytics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)

print("\n" + "=" * 60)
print("ANALYTICS COMPLETE")
print("=" * 60)
print(f"Saved: {REPORTS_DIR / 'analytics.json'}")
print(f"Keys: {list(out.keys())}")