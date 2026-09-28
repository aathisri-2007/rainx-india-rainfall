from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

REAN = Path(__file__).parent / "data" / "reanalysis"
OUT  = Path(__file__).parent / "data" / "regimes"
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    ds = xr.open_dataset(REAN / f"{name}.nc")
    rename = {}
    for c in list(ds.coords):
        if c == "lat": rename[c] = "latitude"
        if c == "lon": rename[c] = "longitude"
    if rename: ds = ds.rename(rename)
    if "time" not in ds.coords:
        for t in ["Time", "TIME", "t"]:
            if t in ds.coords:
                ds = ds.rename({t: "time"}); break
    return ds.resample(time="1D").mean()


def region_mean(ds, var, lat_lo, lat_hi, lon_lo, lon_hi):
    latvals = ds["latitude"].values
    ls = slice(lat_hi, lat_lo) if latvals[0] > latvals[-1] else slice(lat_lo, lat_hi)
    return ds[var].sel(latitude=ls, longitude=slice(lon_lo, lon_hi)) \
                  .mean(dim=["latitude","longitude"]).to_series()


print("Loading reanalysis...")
slp  = load("slp")
uwnd = load("uwnd_10m")
vwnd = load("vwnd_10m")
air  = load("air_2m")
rhum = load("rhum")

have_hgt = (REAN / "hgt_500.nc").exists()
if have_hgt:
    hgt = load("hgt_500")
    print("  hgt_500 loaded")

slp_trough = region_mean(slp,  "slp",  18, 25, 72, 85) / 100.0
uwnd_as    = region_mean(uwnd, "uwnd", 10, 20, 60, 72)
vwnd_as    = region_mean(vwnd, "vwnd", 10, 20, 60, 72)
rhum_ci    = region_mean(rhum, "rhum", 18, 25, 74, 84)
air_nw     = region_mean(air,  "air",  25, 35, 70, 80)
hgt_nw     = region_mean(hgt, "hgt", 25, 35, 70, 80) if have_hgt else None

common = slp_trough.index.intersection(uwnd_as.index).intersection(vwnd_as.index)
common = common.intersection(rhum_ci.index).intersection(air_nw.index)
if hgt_nw is not None:
    common = common.intersection(hgt_nw.index)

slp_trough = slp_trough.loc[common]
uwnd_as    = uwnd_as.loc[common]
vwnd_as    = vwnd_as.loc[common]
rhum_ci    = rhum_ci.loc[common]
air_nw     = air_nw.loc[common]
if hgt_nw is not None:
    hgt_nw = hgt_nw.loc[common]


def anomaly(s, w=30):
    return s - s.rolling(w, min_periods=10, center=True).mean()

slp_anom  = anomaly(slp_trough)
rhum_anom = anomaly(rhum_ci)
air_anom  = anomaly(air_nw)
hgt_anom  = anomaly(hgt_nw) if hgt_nw is not None else None

u = uwnd_as.values; v = vwnd_as.values
wind_from = (270 - np.degrees(np.arctan2(v, u))) % 360
wind_dir  = pd.Series(wind_from, index=uwnd_as.index)
wind_speed = pd.Series(np.sqrt(u**2 + v**2), index=uwnd_as.index)

mask = (common >= "2016-06-01") & (common <= "2016-08-31")
common = common[mask]

slp_anom   = slp_anom.loc[common]
rhum_anom  = rhum_anom.loc[common]
air_anom   = air_anom.loc[common]
wind_dir   = wind_dir.loc[common]
wind_speed = wind_speed.loc[common]
uwnd_as    = uwnd_as.loc[common]
vwnd_as    = vwnd_as.loc[common]
if hgt_anom is not None:
    hgt_anom = hgt_anom.loc[common]

monsoon_index = (-slp_anom)*0.6 + rhum_anom*0.3 + (wind_speed - 9)*0.4

def q(s, p): return float(s.quantile(p))

slp_q15  = q(slp_anom, 0.15)
slp_q65  = q(slp_anom, 0.65)
rh_q50   = q(rhum_anom, 0.50)
wsp_q60  = q(wind_speed, 0.60)
wsp_q75  = q(wind_speed, 0.75)
mon_q70  = q(monsoon_index, 0.70)
hgt_q75  = q(hgt_anom, 0.75) if hgt_anom is not None else None

print(f"\nMonsoon days: {len(common)}")
print(f"SLP q15={slp_q15:.2f} q65={slp_q65:.2f}")
if hgt_anom is not None:
    print(f"Hgt500 q75={hgt_q75:.2f}")


def assign(slp_a, hgt_a, wdir, wspd, rh_a, mon_idx):
    if slp_a < slp_q15:
        return "low_depression"
    if hgt_a is not None and not np.isnan(hgt_a) and hgt_a > hgt_q75:
        return "break_monsoon"
    if (hgt_a is not None and not np.isnan(hgt_a)
            and hgt_a > 0 and hgt_a <= hgt_q75
            and 250 <= wdir <= 320):
        return "western_disturbance"
    if mon_idx > mon_q70 and 215 <= wdir <= 260 and rh_a > rh_q50:
        return "active_monsoon"
    if 220 <= wdir <= 265 and wspd > wsp_q75 and rh_a > rh_q50:
        return "orographic_rainfall"
    if 200 <= wdir <= 290 and wspd > wsp_q60 and rh_a > rh_q50:
        return "coastal_rainfall"
    return "quiet"


records = []
for date in common:
    def safe(x): return 0.0 if np.isnan(x) else float(x)
    slp_a = safe(slp_anom.loc[date]); rh_a = safe(rhum_anom.loc[date])
    mi = safe(monsoon_index.loc[date])
    wdir = float(wind_dir.loc[date]); wspd = float(wind_speed.loc[date])
    if hgt_anom is not None:
        hv = hgt_anom.loc[date]
        hgt_a = None if np.isnan(hv) else float(hv)
    else:
        hgt_a = None
    records.append({
        "date": date,
        "slp_anomaly": slp_a,
        "hgt500_anomaly": hgt_a if hgt_a is not None else np.nan,
        "wind_dir_from": wdir,
        "wind_speed_ms": wspd,
        "rh_anomaly": rh_a,
        "monsoon_index": mi,
        "regime": assign(slp_a, hgt_a, wdir, wspd, rh_a, mi),
    })

df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
out_file = OUT / "regime_labels_2016.csv"
df.to_csv(out_file, index=False)

print("\n" + "="*60)
print("REGIME LABELING COMPLETE")
print("="*60)
print(f"Rows: {len(df)}")
print("\nRegime distribution:")
print(df["regime"].value_counts())
print(f"\nSaved: {out_file}")