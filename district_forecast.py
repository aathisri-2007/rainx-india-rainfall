from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODELS_DIR  = Path(__file__).parent / "models"
DISTRICTS   = Path(__file__).parent / "data" / "district_daily_rainfall_2016_2025.csv"
OUT_DIR     = Path(__file__).parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DISTRICT-LEVEL RAINFALL PRODUCT")
print("=" * 60)

# Latest day in the corrected forecast
df = pd.read_csv(MODELS_DIR / "test_predictions.csv", parse_dates=["date"])
latest = df["date"].max()
print(f"Latest forecast date: {latest.date()}")

today = df[df["date"] == latest].copy()
print(f"Districts on that day: {len(today)}")

# Add heavy-rain probabilities
probs = Path(MODELS_DIR / "test_predictions_with_prob.csv")
if probs.exists():
    prob_df = pd.read_csv(probs, parse_dates=["date"])
    prob_today = prob_df[prob_df["date"] == latest][
        ["district_id", "P_heavy", "P_very_heavy", "P_extremely_heavy"]
    ]
    today = today.merge(prob_today, on="district_id", how="left")

# Get district + state names
imd = pd.read_csv(
    DISTRICTS,
    usecols=["district_id", "district", "state"],
).drop_duplicates("district_id")
imd["district_id"] = imd["district_id"].astype(int).astype(str)

today["district_id"] = today["district_id"].astype(str)
today = today.merge(imd, on="district_id", how="left")

# Clean output
out_cols = ["date", "district_id", "district", "state",
            "regime_label", "gefs_mm", "pred_regime_feat"]
if "P_heavy" in today.columns:
    out_cols += ["P_heavy", "P_very_heavy", "P_extremely_heavy"]

# Sort by corrected rainfall descending
today = today.sort_values("pred_regime_feat", ascending=False)
today[out_cols].to_csv(OUT_DIR / "district_forecast_latest.csv", index=False)

print(f"\nTop 15 districts by corrected forecast:")
print(today[["district", "state", "gefs_mm", "pred_regime_feat",
             "regime_label"]].head(15).to_string(index=False))

# State-level aggregation
state_agg = today.groupby("state").agg(
    districts=("district", "count"),
    mean_gefs=("gefs_mm", "mean"),
    mean_corrected=("pred_regime_feat", "mean"),
    max_corrected=("pred_regime_feat", "max"),
).round(2).sort_values("max_corrected", ascending=False)

state_agg.to_csv(OUT_DIR / "state_forecast_latest.csv")
print(f"\nTop 10 states by max corrected rainfall:")
print(state_agg.head(10).to_string())

# Heavy rainfall alerts (>= 64.5 mm corrected)
alerts = today[today["pred_regime_feat"] >= 64.5][
    ["date", "district", "state", "gefs_mm", "pred_regime_feat"]
]
alerts.to_csv(OUT_DIR / "district_heavy_alerts.csv", index=False)
print(f"\nHeavy rainfall alerts (>=64.5mm corrected): {len(alerts)} districts")
if len(alerts) > 0:
    print(alerts.head(15).to_string(index=False))

print(f"\nOutputs saved to: {OUT_DIR}")