"""
Train regime classifier on 2016 data with stratified split.
Every regime appears in both train and test.
"""
from pathlib import Path
from datetime import datetime
import sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Prefer multi-year if it exists, otherwise use 2016
REGIME_MULTI = Path(__file__).parent / "data" / "regimes" / "regime_labels_2016_2019.csv"
REGIME_2016  = Path(__file__).parent / "data" / "regimes" / "regime_labels_2016.csv"

REGIME_CSV = REGIME_MULTI if REGIME_MULTI.exists() else REGIME_2016
print("=" * 60)
print("TRAINING WEATHER REGIME CLASSIFIER")
print(f"Source: {REGIME_CSV.name}")
print("=" * 60)

df = pd.read_csv(REGIME_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

candidate = ["slp_anomaly", "hgt500_anomaly", "wind_dir_from",
             "wind_speed_ms", "rh_anomaly", "monsoon_index"]
features = [c for c in candidate if c in df.columns and df[c].notna().sum() > 5]
print(f"\nUsing features: {features}")

X = df[features].fillna(df[features].median(numeric_only=True))
y = df["regime"].astype(str)

# Drop regimes with fewer than 3 samples
counts = y.value_counts()
rare = counts[counts < 3].index.tolist()
if rare:
    print(f"Dropping regimes with <3 samples: {rare}")
    keep = ~y.isin(rare)
    df = df[keep].reset_index(drop=True)
    X = X[keep].reset_index(drop=True)
    y = y[keep].reset_index(drop=True)

# Stratified split — every regime in train AND test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")
print("\nTrain distribution:"); print(y_train.value_counts())
print("\nTest distribution:");  print(y_test.value_counts())

model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced_subsample",
    random_state=42,
    min_samples_leaf=1,
)
model.fit(X_train, y_train)

print("\nFeature importances:")
for name, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:20s}  {imp:.3f}")

y_pred = model.predict(X_test)
acc = float((y_pred == y_test).mean())
print(f"\nTest accuracy: {acc:.4f}")

print("\nClassification report (test):")
print(classification_report(y_test, y_pred, zero_division=0))

labels_sorted = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
print("\nConfusion matrix:")
print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))

# ---------- Save ----------
joblib.dump(
    {"model": model, "features": features, "labels": labels_sorted},
    MODELS_DIR / "regime_classifier.joblib"
)

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
metadata = {
    "model_name": "regime_classifier",
    "algorithm":  "RandomForestClassifier",
    "features":   features,
    "training_period":   ["2016 monsoon (stratified)"],
    "validation_period": ["—"],
    "test_period":       ["2016 monsoon (stratified)"],
    "split_method":      "Stratified 70/30",
    "training_note":     "Multi-year extension (2017-2019) requires NCEP reanalysis; pipeline ready to ingest.",
    "trained_at": datetime.utcnow().isoformat() + "Z",
    "version":    "2.1.0",
    "metrics": {
        "test_accuracy":   acc,
        "n_train":         int(len(X_train)),
        "n_test":          int(len(X_test)),
        "labels":          labels_sorted,
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {"regime": lab,
             "precision": round(float(report[lab]["precision"]), 3),
             "recall":    round(float(report[lab]["recall"]), 3),
             "f1":        round(float(report[lab]["f1-score"]), 3),
             "support":   int(report[lab]["support"])}
            for lab in labels_sorted if lab in report
        ],
    },
}
with open(MODELS_DIR / "regime_classifier.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, default=str)

df["regime_pred"] = model.predict(X)
df[["date", "regime", "regime_pred"] + features].to_csv(
    MODELS_DIR / "regime_predictions.csv", index=False
)

print(f"\nSaved model:      {MODELS_DIR / 'regime_classifier.joblib'}")
print(f"Saved metadata:   {MODELS_DIR / 'regime_classifier.json'}")
print(f"Saved predictions:{MODELS_DIR / 'regime_predictions.csv'}")