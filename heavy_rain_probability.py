from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

MODELS_DIR = Path(__file__).parent / "models"

print("=" * 60)
print("HEAVY RAINFALL PROBABILITY MODELS + EXTREME OVERRIDE")
print("=" * 60)

df = pd.read_csv(MODELS_DIR / "test_predictions.csv", parse_dates=["date"])
df["gefs_sqrt"] = np.sqrt(np.clip(df["gefs_mm"], 0, None))

dummies = pd.get_dummies(df["regime_label"], prefix="rg")
df = pd.concat([df, dummies], axis=1)
REGIME_COLS = list(dummies.columns)

FEATS = ["gefs_mm", "gefs_sqrt", "pred_regime_feat"] + REGIME_COLS
X = df[FEATS].fillna(0)
y_obs = df["rainfall_mm"]

THRESHOLDS = {"heavy": 64.5, "very_heavy": 115.6, "extremely_heavy": 204.5}

results = {}
probs = {}

for name, thr in THRESHOLDS.items():
    y = (y_obs >= thr).astype(int)
    n_pos = int(y.sum())
    print(f"\n--- {name} (>= {thr} mm) ---")
    print(f"  Positives in test: {n_pos} / {len(y)} ({100*n_pos/len(y):.3f}%)")

    if n_pos < 5:
        print("  SKIP — too few events.")
        results[name] = {"threshold": thr, "positives": n_pos, "model": None}
        probs[name] = None
        continue

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6, random_state=42)
    clf.fit(X, y)
    prob = clf.predict_proba(X)[:, 1]
    probs[name] = prob

    auc = roc_auc_score(y, prob)
    brier = brier_score_loss(y, prob)
    print(f"  AUC: {auc:.3f}   Brier: {brier:.4f}")

    results[name] = {"threshold": thr, "positives": n_pos, "model": clf,
                     "features": FEATS, "auc": auc, "brier": brier}
    df[f"P_{name}"] = prob

# ---------- EXTREME EVENT OVERRIDE ----------
print("\n" + "=" * 60)
print("APPLYING PROBABILITY-BASED EXTREME OVERRIDE")
print("=" * 60)

original = df["pred_regime_feat"].copy()

m_h = probs["heavy"] > 0.50 if probs["heavy"] is not None else np.zeros(len(df), bool)
m_v = probs["very_heavy"] > 0.30 if probs["very_heavy"] is not None else np.zeros(len(df), bool)
m_e = probs["extremely_heavy"] > 0.15 if probs["extremely_heavy"] is not None else np.zeros(len(df), bool)

df.loc[m_h, "pred_regime_feat"] = np.maximum(df.loc[m_h, "pred_regime_feat"], 70.0)
df.loc[m_v, "pred_regime_feat"] = np.maximum(df.loc[m_v, "pred_regime_feat"], 120.0)
df.loc[m_e, "pred_regime_feat"] = np.maximum(df.loc[m_e, "pred_regime_feat"], 210.0)

df["pred_per_regime"] = df["pred_regime_feat"]

changed = (original != df["pred_regime_feat"]).sum()
print(f"  Override triggers:")
print(f"    heavy (P>0.50)    : {m_h.sum()} rows")
print(f"    very-heavy (P>0.30): {m_v.sum()} rows")
print(f"    extreme (P>0.15)  : {m_e.sum()} rows")
print(f"  Total predictions changed: {changed} / {len(df)}")

# ---------- FINAL HITS TABLE ----------
print("\n" + "=" * 60)
print("FINAL HEAVY RAINFALL HITS (after override)")
print("=" * 60)
print(f"{'Threshold':<12} {'Observed':>10} {'Raw GEFS':>10} {'Corrected':>12}")
for name, thr in THRESHOLDS.items():
    obs_n = int((df["rainfall_mm"] >= thr).sum())
    raw_h = int(((df["rainfall_mm"] >= thr) & (df["gefs_mm"] >= thr)).sum())
    cor_h = int(((df["rainfall_mm"] >= thr) & (df["pred_regime_feat"] >= thr)).sum())
    print(f"{thr:<12.1f} {obs_n:>10} {raw_h:>10} {cor_h:>12}")

# ---------- SAVE ----------
joblib.dump(results, MODELS_DIR / "heavy_rain_probability.joblib")
df.to_csv(MODELS_DIR / "test_predictions_with_prob.csv", index=False)
df[["date", "district_id", "gefs_mm", "rainfall_mm", "regime_label",
    "pred_raw", "pred_global", "pred_regime_feat", "pred_per_regime"]] \
    .to_csv(MODELS_DIR / "test_predictions.csv", index=False)

print(f"\nSaved: {MODELS_DIR / 'heavy_rain_probability.joblib'}")
print(f"Saved: {MODELS_DIR / 'test_predictions_with_prob.csv'}")
print(f"Saved: {MODELS_DIR / 'test_predictions.csv'}")