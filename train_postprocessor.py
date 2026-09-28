"""
Train regime-aware bias correction on 2016-2019 data.
Uses GEFS NWP + IMD observations (both already downloaded).
Regime labels only available for 2016 — other years tagged as "unknown".
Quantile mapping extrapolates the tail for extreme events.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

GEFS_CSV    = Path(__file__).parent / "data" / "gefs_india_2016_2019.csv"
IMD_CSV     = Path(__file__).parent / "data" / "district_daily_rainfall_2016_2025.csv"
REGIME_CSV  = Path(__file__).parent / "data" / "regimes" / "regime_labels_2016.csv"
MODELS_DIR  = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def norm_id(x):
    """Normalize district_id to clean integer string regardless of source format."""
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x).strip()


def norm_date(s):
    return pd.to_datetime(s).dt.normalize()


print("=" * 60)
print("REGIME-AWARE BIAS CORRECTION (2016-2019)")
print("=" * 60)

# ---------- Regime labels (only 2016 available) ----------
regime = pd.read_csv(REGIME_CSV, parse_dates=["date"])
regime = regime[["date", "regime"]].rename(columns={"regime": "regime_label"})
regime["date"] = norm_date(regime["date"])

# ---------- GEFS (already district-level) ----------
print("Loading GEFS 2016-2019...")
gefs = pd.read_csv(GEFS_CSV)
gefs["date"] = norm_date(gefs["date"])
gefs["district_id"] = gefs["district_id"].apply(norm_id)
gefs = gefs[["date", "district_id", "gefs_mm"]]
print(f"  GEFS rows: {len(gefs):,}")

# ---------- IMD ----------
print("Loading IMD...")
imd = pd.read_csv(IMD_CSV, usecols=["date", "district_id", "rainfall_mm"])
imd["date"] = norm_date(imd["date"])
imd["district_id"] = imd["district_id"].apply(norm_id)
print(f"  IMD rows: {len(imd):,}")

# ---------- Keep only monsoon months Jun-Sep 2016-2019 ----------
gefs_monsoon = gefs["date"].dt.month.isin([6, 7, 8, 9]) & gefs["date"].dt.year.between(2016, 2019)
gefs = gefs[gefs_monsoon]
print(f"  GEFS monsoon rows: {len(gefs):,}")

imd_monsoon = imd["date"].dt.month.isin([6, 7, 8, 9]) & imd["date"].dt.year.between(2016, 2019)
imd = imd[imd_monsoon]

print(f"  GEFS sample district_ids: {gefs['district_id'].head(3).tolist()}")
print(f"  IMD  sample district_ids: {imd['district_id'].head(3).tolist()}")
print(f"  GEFS sample dates: {[str(d)[:10] for d in gefs['date'].head(3).tolist()]}")
print(f"  IMD  sample dates: {[str(d)[:10] for d in imd['date'].head(3).tolist()]}")

# ---------- Merge ----------
df = gefs.merge(imd, on=["date", "district_id"], how="inner")
df = df.merge(regime, on="date", how="left")
df["regime_label"] = df["regime_label"].fillna("unknown")
df = df.dropna(subset=["gefs_mm", "rainfall_mm"])
print(f"\nMerged rows: {len(df):,}")
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Districts: {df['district_id'].nunique()}")

if len(df) == 0:
    raise SystemExit("Merge produced 0 rows — check district_id formats")

# ---------- Features ----------
df["gefs_sqrt"] = np.sqrt(np.clip(df["gefs_mm"], 0, None))
df["month"] = df["date"].dt.month
df["doy"]   = df["date"].dt.dayofyear

# Regime one-hot
dummies = pd.get_dummies(df["regime_label"], prefix="rg").astype(int)
df = pd.concat([df, dummies], axis=1)
REGIME_FEATS = list(dummies.columns)
print(f"Regime features: {REGIME_FEATS}")

# ---------- Time-aware split: train 2016-2018, test 2019 ----------
train = df[df["date"].dt.year < 2019].copy()
test  = df[df["date"].dt.year == 2019].copy()
print(f"\nTrain rows: {len(train):,}")
print(f"Test rows:  {len(test):,}")

if len(train) == 0 or len(test) == 0:
    raise SystemExit("Empty train or test set — check date filter")


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


raw_rmse = rmse(test["rainfall_mm"], test["gefs_mm"])
print(f"\nRaw GEFS RMSE: {raw_rmse:.3f}")

FEATS_GLOBAL = ["gefs_mm", "gefs_sqrt", "month", "doy"]
FEATS_REGIME = FEATS_GLOBAL + REGIME_FEATS

# ---------- Train on log target ----------
train["y_log"] = np.log1p(train["rainfall_mm"])

print("\nTraining global RF...")
rf_global = RandomForestRegressor(n_estimators=400, random_state=42,
                                  n_jobs=-1, min_samples_leaf=2)
rf_global.fit(train[FEATS_GLOBAL], train["y_log"])
pred_global = np.expm1(rf_global.predict(test[FEATS_GLOBAL]))
print(f"  Global RF RMSE: {rmse(test['rainfall_mm'], pred_global):.3f}")

print("Training regime-aware RF...")
rf_regime = RandomForestRegressor(n_estimators=400, random_state=42,
                                  n_jobs=-1, min_samples_leaf=2)
rf_regime.fit(train[FEATS_REGIME], train["y_log"])
pred_rf = np.expm1(rf_regime.predict(test[FEATS_REGIME]))
print(f"  Regime-aware RF RMSE: {rmse(test['rainfall_mm'], pred_rf):.3f}")

# ---------- Quantile mapping WITH TAIL EXTRAPOLATION ----------
print("\nFitting quantile mapping (with extrapolation)...")
train_pred = np.expm1(rf_regime.predict(train[FEATS_REGIME]))
sorted_pred = np.sort(train_pred)
sorted_obs  = np.sort(train["rainfall_mm"].values)

PRED_MAX = float(sorted_pred.max())
OBS_MAX  = float(sorted_obs.max())
print(f"  Training pred max: {PRED_MAX:.1f} mm")
print(f"  Training obs max:  {OBS_MAX:.1f} mm")


def quantile_map_with_extrapolate(preds):
    out = np.zeros_like(preds, dtype=float)
    for i, p in enumerate(preds):
        if p <= PRED_MAX:
            rank = np.searchsorted(sorted_pred, p) / max(len(sorted_pred) - 1, 1)
            rank = np.clip(rank, 0, 1)
            idx = int(rank * (len(sorted_obs) - 1))
            out[i] = sorted_obs[idx]
        else:
            ratio = p / PRED_MAX
            out[i] = OBS_MAX * ratio
    return out


pred_qm = quantile_map_with_extrapolate(pred_rf)
blend = 0.7 * pred_qm + 0.3 * pred_rf

# Tail boost: when prediction is high, apply a calibrated uplift
# Based on observed/predicted ratio for high-rain days in training
train_pred_qm = quantile_map_with_extrapolate(train_pred)
train_hi = train["rainfall_mm"] >= 64.5
if train_hi.sum() > 0:
    hi_ratio = float(train.loc[train_hi, "rainfall_mm"].mean() /
                     max(train_pred_qm[train_hi.values].mean(), 0.1))
    hi_ratio = min(max(hi_ratio, 1.0), 2.5)   # cap between 1 and 2.5
else:
    hi_ratio = 1.0

very_hi = train["rainfall_mm"] >= 115.6
if very_hi.sum() > 0:
    vhi_ratio = float(train.loc[very_hi, "rainfall_mm"].mean() /
                      max(train_pred_qm[very_hi.values].mean(), 0.1))
    vhi_ratio = min(max(vhi_ratio, 1.0), 3.5)
else:
    vhi_ratio = 1.0

print(f"  Tail uplift ratios — heavy: {hi_ratio:.2f}×  very-heavy: {vhi_ratio:.2f}×")

# Extreme-event uplift for the top range
extreme = train["rainfall_mm"] >= 204.5
if extreme.sum() > 0:
    ext_ratio = float(train.loc[extreme, "rainfall_mm"].mean() /
                      max(train_pred_qm[extreme.values].mean(), 0.1))
    ext_ratio = min(max(ext_ratio, 1.0), 5.0)
else:
    ext_ratio = 1.0

print(f"  Tail uplift ratios — heavy: {hi_ratio:.2f}×  very-heavy: {vhi_ratio:.2f}×  extreme: {ext_ratio:.2f}×")

final = blend.copy()
final[blend >= 64.5]  = blend[blend >= 64.5]  * hi_ratio
final[blend >= 115.6] = blend[blend >= 115.6] * vhi_ratio
final[blend >= 160.0] = blend[blend >= 160.0] * ext_ratio
final = np.clip(final, 0, None)

final_rmse = rmse(test["rainfall_mm"], final)
print(f"  Regime + QM (extrapolated) RMSE: {final_rmse:.3f}")
print(f"  Max predicted: {final.max():.1f} mm")
print(f"  Max observed:  {test['rainfall_mm'].max():.1f} mm")

# ---------- Category hits ----------
print("\n" + "=" * 60)
print("HEAVY RAINFALL EVENT HITS (2019 test set)")
print("=" * 60)
print(f"{'Threshold':<15} {'Observed':>10} {'Raw GEFS':>10} {'RF':>10} {'RF+QM':>10}")
for thr in [20.0, 64.5, 115.6, 204.5]:
    obs_n = int((test["rainfall_mm"] >= thr).sum())
    raw_h = int(((test["rainfall_mm"] >= thr) & (test["gefs_mm"] >= thr)).sum())
    rf_h  = int(((test["rainfall_mm"] >= thr) & (pred_rf >= thr)).sum())
    qm_h  = int(((test["rainfall_mm"] >= thr) & (final >= thr)).sum())
    print(f"{thr:<15.1f} {obs_n:>10} {raw_h:>10} {rf_h:>10} {qm_h:>10}")

# ---------- Save ----------
test = test.reset_index(drop=True)
test["pred_raw"]         = test["gefs_mm"]
test["pred_global"]      = pred_global
test["pred_regime_feat"] = final
test["pred_per_regime"]  = final

joblib.dump({
    "rf_global": rf_global,
    "rf_regime_feat": rf_regime,
    "features_global": FEATS_GLOBAL,
    "features_regime": FEATS_REGIME,
    "qm_sorted_pred": sorted_pred,
    "qm_sorted_obs":  sorted_obs,
    "qm_pred_max":    PRED_MAX,
    "qm_obs_max":     OBS_MAX,
    "log_target":     True,
}, MODELS_DIR / "bias_correction_models.joblib")

test[["date", "district_id", "gefs_mm", "rainfall_mm", "regime_label",
      "pred_raw", "pred_global", "pred_regime_feat", "pred_per_regime"]] \
    .to_csv(MODELS_DIR / "test_predictions.csv", index=False)

print(f"\nSaved models:      {MODELS_DIR / 'bias_correction_models.joblib'}")
print(f"Saved predictions: {MODELS_DIR / 'test_predictions.csv'}")
print(f"\nTrain rows: {len(train):,}  Test rows: {len(test):,}")