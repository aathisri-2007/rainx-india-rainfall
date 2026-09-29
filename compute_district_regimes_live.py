"""
Derive per-district regimes using the global daily regime + district geography
+ rainfall intensity. Runs on cloud (no reanalysis files needed).
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
REPORTS = BASE / "reports"
MAPPING = BASE / "data" / "grid_to_district.csv"

print("Loading grid mapping...")
mapping = pd.read_csv(MAPPING).dropna(subset=["district_id"]).copy()
mapping["district_id"] = mapping["district_id"].astype(int).astype(str)
mapping["latitude"]  = mapping["latitude"].round(2)
mapping["longitude"] = mapping["longitude"].round(2)

centroids = mapping.groupby("district_id")[["latitude", "longitude"]].mean().reset_index()
centroids["district_id"] = centroids["district_id"].astype(str)


def geography(lat, lon):
    if 8 <= lat <= 21 and 73 <= lon <= 78:   return "orographic_west"
    if 27 <= lat <= 35 and 72 <= lon <= 90:  return "orographic_north"
    if 8 <= lat <= 22 and 79.5 <= lon <= 82.5: return "coastal_east"
    if 8 <= lat <= 22 and 72 <= lon <= 75:   return "coastal_west"
    if 23 <= lat <= 37 and 68 <= lon <= 78:  return "nw_india"
    if 18 <= lat <= 25 and 74 <= lon <= 85:  return "central"
    return "interior"


centroids["geo"] = centroids.apply(lambda r: geography(r.latitude, r.longitude), axis=1)
print("\nGeographic classification:")
print(centroids["geo"].value_counts())

live_file = REPORTS / "realtime_forecast.csv"
if not live_file.exists():
    raise SystemExit("realtime_forecast.csv not found")

live = pd.read_csv(live_file)
live["district_id"] = live["district_id"].astype(str)

if "regime_label" in live.columns:
    global_regime = live["regime_label"].mode().iloc[0]
else:
    global_regime = "quiet"
print(f"\nGlobal daily regime: {global_regime}")

live = live.merge(centroids[["district_id", "geo"]], on="district_id", how="left")

rainfall = live["pred_regime_feat"].fillna(0).values
if rainfall.max() > 0:
    q50 = float(np.quantile(rainfall, 0.50))
    q75 = float(np.quantile(rainfall, 0.75))
    q90 = float(np.quantile(rainfall, 0.90))
    q95 = float(np.quantile(rainfall, 0.95))
else:
    q50 = q75 = q90 = q95 = 0.0

print(f"\nRainfall quantiles (mm): P50={q50:.2f}  P75={q75:.2f}  P90={q90:.2f}  P95={q95:.2f}")


def refine(row):
    geo = row.get("geo", "interior")
    rain = row.get("pred_regime_feat", 0) or 0
    g = global_regime

    if g == "low_depression":   return "low_depression"
    if g == "break_monsoon":    return "break_monsoon"

    if g == "active_monsoon":
        if geo == "orographic_west"  and rain >= q75: return "orographic_rainfall"
        if geo == "orographic_north" and rain >= q75: return "orographic_rainfall"
        if geo == "coastal_east"     and rain >= q75: return "coastal_rainfall"
        if geo == "coastal_west"     and rain >= q75: return "coastal_rainfall"
        if rain >= q90: return "active_monsoon"
        return "quiet"

    if g == "western_disturbance":
        if geo == "nw_india" and rain >= q75: return "western_disturbance"
        if rain >= q95: return "western_disturbance"
        return "quiet"

    if g == "coastal_rainfall":
        if geo in ("coastal_east", "coastal_west") and rain >= q50: return "coastal_rainfall"
        if geo in ("orographic_west", "orographic_north") and rain >= q75: return "orographic_rainfall"
        return "quiet"

    if geo == "orographic_west"  and rain >= q90: return "orographic_rainfall"
    if geo == "orographic_north" and rain >= q90: return "orographic_rainfall"
    if geo in ("coastal_east", "coastal_west") and rain >= q90: return "coastal_rainfall"
    if geo == "nw_india" and rain >= q95: return "western_disturbance"
    return "quiet"


live["regime_label"] = live.apply(refine, axis=1)

def confidence(row):
    geo = row.get("geo", "interior")
    regime = row["regime_label"]
    rain = row.get("pred_regime_feat", 0) or 0
    match = {
        ("orographic_west",  "orographic_rainfall"):  0.85,
        ("orographic_north", "orographic_rainfall"):  0.82,
        ("coastal_east",     "coastal_rainfall"):     0.80,
        ("coastal_west",     "coastal_rainfall"):     0.80,
        ("nw_india",         "western_disturbance"):  0.78,
        ("central",          "low_depression"):       0.85,
    }
    base = match.get((geo, regime), 0.60)
    if rain >= q90: base = min(base + 0.05, 0.95)
    return round(base, 3)


live["regime_confidence"] = live.apply(confidence, axis=1)
live["global_regime"] = global_regime

print("\nPer-district regime distribution:")
print(live["regime_label"].value_counts())

live = live.drop(columns=["geo"], errors="ignore")
live.to_csv(live_file, index=False)
print(f"\nUpdated {live_file}")
print(f"  Districts: {len(live)}")
print(f"  Unique regimes: {live['regime_label'].nunique()}")