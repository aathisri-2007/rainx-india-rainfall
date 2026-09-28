"""
Rebuild regime labels using 2016-2019 data via OPeNDAP.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "core"))

import numpy as np
import pandas as pd
from district_features import load_reanalysis, monsoon_climatology, fix_lon
import xarray as xr

OUT = Path(__file__).parent / "data" / "regimes"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading reanalysis via OPeNDAP (2016-2019)...")
slp_ds  = load_reanalysis(None, "slp",      years=[2016,2017,2018,2019])
u_ds    = load_reanalysis(None, "uwnd_10m", years=[2016,2017,2018,2019])
v_ds    = load_reanalysis(None, "vwnd_10m", years=[2016,2017,2018,2019])
rh_ds   = load_reanalysis(None, "rhum",     years=[2016,2017,2018,2019])
hgt_ds  = load_reanalysis(None, "hgt_500",  years=[2016,2017,2018,2019])

if slp_ds is None:
    raise SystemExit("Could not load slp — OPeNDAP may be down")

print("\nComputing region means...")


def region_mean(ds, var, lat_lo, lat_hi, lon_lo, lon_hi):
    da = fix_lon(ds[var])
    latvals = da.latitude.values
    ls = slice(lat_hi, lat_lo) if latvals[0] > latvals[-1] else slice(lat_lo, lat_hi)
    return da.sel(latitude=ls, longitude=slice(lon_lo, lon_hi)).mean(dim=["latitude","longitude"]).to_series()


slp_trough = region_mean(slp_ds, "slp", 18, 25, 72, 85) / 100.0
uwnd_as    = region_mean(u_ds,   "uwnd", 10, 20, 60, 72)
vwnd_as    = region_mean(v_ds,   "vwnd", 10, 20, 60, 72)
rhum_ci    = region_mean(rh_ds,  "rhum", 18, 25, 74, 84)
hgt_nw     = region_mean(hgt_ds, "hgt",  25, 35, 70, 80)

# Resample to daily
slp_trough = slp_trough.resample("1D").mean()
uwnd_as    = uwnd_as.resample("1D").mean()
vwnd_as    = vwnd_as.resample("1D").mean()
rhum_ci    = rhum_ci.resample("1D").mean()
hgt_nw     = hgt_nw.resample("1D").mean()

# Anomalies (30-day rolling)
def anomaly(s, w=30):
    return s - s.rolling(w, min_periods=10, center=True).mean()

slp_anom  = anomaly(slp_trough)
rhum_anom = anomaly(rhum_ci)
hgt_anom  = anomaly(hgt_nw)

wind_from = (270 - np.degrees(np.arctan2(vwnd_as.values, uwnd_as.values))) % 360
wind_dir  = pd.Series(wind_from, index=uwnd_as.index)
wind_speed = pd.Series(np.sqrt(uwnd_as.values**2 + vwnd_as.values**2), index=uwnd_as.index)

# Keep only monsoon months
df = pd.DataFrame({
    "slp_anomaly":    slp_anom,
    "rh_anomaly":     rhum_anom,
    "hgt500_anomaly": hgt_anom,
    "wind_dir_from":  wind_dir,
    "wind_speed_ms":  wind_speed,
})
df = df[df.index.month.isin([6,7,8,9])].dropna()

# Monsoon index
df["monsoon_index"] = -df["slp_anomaly"]*0.6 + df["rh_anomaly"]*0.3 + (df["wind_speed_ms"]-9)*0.4

# Quantile thresholds (data-driven)
q = df.quantile
slp_q15 = q("slp_anomaly", 0.15)
slp_q35 = q("slp_anomaly", 0.35)
rh_q50  = q("rh_anomaly", 0.50)
wsp_q60 = q("wind_speed_ms", 0.60)
wsp_q75 = q("wind_speed_ms", 0.75)
mon_q70 = q("monsoon_index", 0.70)
hgt_q75 = q("hgt500_anomaly", 0.75)


def assign(row):
    if row.slp_anomaly < slp_q15:
        return "low_depression"
    if row.hgt500_anomaly > hgt_q75:
        return "break_monsoon"
    if (0 < row.hgt500_anomaly <= hgt_q75) and 250 <= row.wind_dir_from <= 320:
        return "western_disturbance"
    if row.monsoon_index > mon_q70 and 215 <= row.wind_dir_from <= 260 and row.rh_anomaly > rh_q50:
        return "active_monsoon"
    if 220 <= row.wind_dir_from <= 265 and row.wind_speed_ms > wsp_q75 and row.rh_anomaly > rh_q50:
        return "orographic_rainfall"
    if 200 <= row.wind_dir_from <= 290 and row.wind_speed_ms > wsp_q60 and row.rh_anomaly > rh_q50:
        return "coastal_rainfall"
    return "quiet"


df["regime"] = df.apply(assign, axis=1)
df = df.reset_index().rename(columns={"index": "date"})
df["date"] = pd.to_datetime(df["date"])

out = OUT / "regime_labels_2016_2019.csv"
df.to_csv(out, index=False)

print(f"\n{'='*50}")
print(f"REGIME LABELS COMPLETE")
print(f"{'='*50}")
print(f"Rows: {len(df)}")
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"\nRegime distribution:")
print(df["regime"].value_counts())
print(f"\nSaved: {out}")
