"""
Fetch today's GFS forecast and produce a real-time corrected rainfall forecast.
"""
from herbie import Herbie
from datetime import datetime, timedelta
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
import joblib

# ---------------- CONFIG ----------------
BASE        = Path(__file__).parent
REAN        = Path(__file__).parent / "data" / "reanalysis"
MAPPING_CSV = Path(__file__).parent / "data" / "grid_to_district.csv"
REPORTS     = BASE / "reports"
MODELS      = BASE / "models"
REPORTS.mkdir(exist_ok=True)

HERBIE_CACHE = Path(__file__).parent / "cache"
HERBIE_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HERBIE_SAVE_DIR"] = str(HERBIE_CACHE)

LAT_MIN, LAT_MAX = 5.0, 40.0
LON_MIN, LON_MAX = 60.0, 105.0

# Region boxes: lat_lo, lat_hi, lon_lo, lon_hi
SLP_TROUGH = (18, 25, 72, 85)
UAS        = (10, 20, 60, 72)
RHUM_CI    = (18, 25, 74, 84)
HGT_NW     = (25, 35, 70, 80)

# GFS search strings (using the exact strings Herbie found)
GFS_SEARCH = {
    "slp":    ":PRMSL:mean sea level:",
    "uwnd":   ":UGRD:10 m above ground:",
    "vwnd":   ":VGRD:10 m above ground:",
    "rhum":   ":RH:2 m above ground:",
    "hgt500": ":HGT:500 mb:",
    "apcp":   ":APCP:surface:0-1 day acc fcst:",
}


# ---------------- HELPERS ----------------
def fix_lon(ds):
    """Convert 0-360 longitude to -180..180 so slices work."""
    if "longitude" not in ds.coords:
        return ds
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")
    return ds


def region_mean(da, lat_lo, lat_hi, lon_lo, lon_hi):
    """Mean of a dataarray over a lat/lon box (assumes -180..180 lon)."""
    da = fix_lon(da)
    latvals = da.latitude.values
    ls = slice(lat_hi, lat_lo) if latvals[0] > latvals[-1] else slice(lat_lo, lat_hi)
    sub = da.sel(latitude=ls, longitude=slice(lon_lo, lon_hi))
    return float(sub.mean().values)


def load_reanalysis_climatology():
    """
    Pre-computed JJA 2016 climatology values.
    These are constants — same for every run.
    """
    print("Using pre-computed JJA 2016 climatology.")
    return {
        "slp_trough_pa": 100243.9,
        "u_as":         5.5,
        "v_as":         5.0,
        "rh_ci":        79.19,
        "hgt_nw":       5863.4,
    }


def clim_mean(clim, name, var, box):
    ds = clim[name]
    da = ds[var]
    if "level" in da.dims or "level" in da.coords:
        try:
            da = da.sel(level=500.0)
        except Exception:
            pass
    return region_mean(da, *box)


# ---------------- GFS FETCH ----------------
def fetch_gfs(days_back=1):
    run_time = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d 00:00')
    print(f"Fetching GFS run: {run_time}")

    out = {}
    for key, search in GFS_SEARCH.items():
        try:
            H = Herbie(run_time, model='gfs', product='pgrb2.0p25', fxx=24,
                       priority=['aws', 'nomads', 'google'],
                       save_dir=str(HERBIE_CACHE))
            ds = H.xarray(search)
            out[key] = ds
            print(f"  OK {key}")
        except Exception as e:
            print(f"  FAIL {key}: {e}")
    return out, run_time


# ---------------- MAIN ----------------
def main():
    clim = load_reanalysis_climatology()

    slp_clim_pa = clim["slp_trough_pa"]
    u_clim      = clim["u_as"]
    v_clim      = clim["v_as"]
    rh_clim     = clim["rh_ci"]
    hgt_clim    = clim["hgt_nw"]

    print(f"\nClimatology baselines:")
    print(f"  slp_trough = {slp_clim_pa:.1f} Pa")
    print(f"  rhum_ci    = {rh_clim:.2f} %")
    print(f"  hgt_nw     = {hgt_clim:.1f} m")

    gfs, run_time = fetch_gfs(days_back=1)
    needed = ["slp", "uwnd", "vwnd", "rhum", "hgt500", "apcp"]
    missing = [k for k in needed if k not in gfs]
    if missing:
        raise SystemExit(f"Missing GFS variables: {missing}")

    # Diagnostic
    sample = gfs["slp"]
    print("  Sample longitude range:",
          float(sample.longitude.min()), "to", float(sample.longitude.max()))
    print("  Sample latitude range:",
          float(sample.latitude.min()), "to", float(sample.latitude.max()))

    # Extract region means
    slp_da = list(gfs["slp"].data_vars.values())[0]
    u_da   = list(gfs["uwnd"].data_vars.values())[0]
    v_da   = list(gfs["vwnd"].data_vars.values())[0]
    rh_da  = list(gfs["rhum"].data_vars.values())[0]
    hg_da  = list(gfs["hgt500"].data_vars.values())[0]

    slp_now_pa = region_mean(slp_da, *SLP_TROUGH)
    u_now      = region_mean(u_da,   *UAS)
    v_now      = region_mean(v_da,   *UAS)
    rh_now     = region_mean(rh_da,  *RHUM_CI)
    hgt_now    = region_mean(hg_da,  *HGT_NW)

    slp_anom   = (slp_now_pa - slp_clim_pa) / 100.0
    rh_anom    = rh_now - rh_clim
    hgt_anom   = hgt_now - hgt_clim

    wind_from  = (270 - np.degrees(np.arctan2(v_now, u_now))) % 360
    wind_speed = float(np.sqrt(u_now**2 + v_now**2))

    monsoon_index = (-slp_anom) * 0.6 + rh_anom * 0.3 + (wind_speed - 9) * 0.4

    print(f"\nGFS-derived features:")
    print(f"  slp_anomaly    = {slp_anom:.2f} hPa")
    print(f"  hgt500_anomaly = {hgt_anom:.2f} m")
    print(f"  wind_dir_from  = {wind_from:.0f} deg")
    print(f"  wind_speed_ms  = {wind_speed:.2f} m/s")
    print(f"  rh_anomaly     = {rh_anom:.2f} %")
    print(f"  monsoon_index  = {monsoon_index:.2f}")

    # Regime prediction
    reg_bundle = joblib.load(MODELS / "regime_classifier.joblib")
    reg_model  = reg_bundle["model"]
    reg_feats  = reg_bundle["features"]

    feat_row = {
        "slp_anomaly":    slp_anom,
        "hgt500_anomaly": hgt_anom,
        "wind_dir_from":  wind_from,
        "wind_speed_ms":  wind_speed,
        "rh_anomaly":     rh_anom,
        "monsoon_index":  monsoon_index,
    }
    X_reg = pd.DataFrame([feat_row])
    for f in reg_feats:
        if f not in X_reg.columns:
            X_reg[f] = 0.0
    X_reg = X_reg[reg_feats]
    regime = reg_model.predict(X_reg)[0]
    print(f"\nPredicted regime: {regime}")

    # Rainfall
    apcp = gfs["apcp"]
    apcp_var = list(apcp.data_vars)[0]
    da = apcp[apcp_var]
    da = fix_lon(da)
    latvals = da.latitude.values
    ls = slice(LAT_MAX, LAT_MIN) if latvals[0] > latvals[-1] else slice(LAT_MIN, LAT_MAX)
    sub = da.sel(latitude=ls, longitude=slice(LON_MIN, LON_MAX))

    df = sub.to_dataframe(name="gefs_mm").reset_index()
    df["latitude"]  = df["latitude"].round(2)
    df["longitude"] = df["longitude"].round(2)
    df["gefs_mm"]   = df["gefs_mm"].clip(lower=0)

    mapping = pd.read_csv(MAPPING_CSV).dropna(subset=["district_id"]).copy()
    mapping["district_id"] = mapping["district_id"].astype(int).astype(str)
    mapping["latitude"]  = mapping["latitude"].round(2)
    mapping["longitude"] = mapping["longitude"].round(2)

    merged = df.merge(
        mapping[["latitude", "longitude", "district_id", "district", "state"]],
        on=["latitude", "longitude"], how="inner"
    )
    if len(merged) == 0:
        raise SystemExit("No districts matched after merge. Check longitude normalization.")

    district_raw = merged.groupby(
        ["district_id", "district", "state"], as_index=False
    )["gefs_mm"].mean()

    # Bias correction
    bias_bundle = joblib.load(MODELS / "bias_correction_models.joblib")
    model_rf    = bias_bundle["rf_regime_feat"]
    feat_reg    = bias_bundle["features_regime"]

    target_date = datetime.utcnow()
    month = target_date.month
    doy   = target_date.timetuple().tm_yday

    X = district_raw.copy()
    X["gefs_sqrt"] = np.sqrt(X["gefs_mm"].clip(lower=0))
    X["month"] = month
    X["doy"]   = doy
    X["regime_label"] = regime

    for col in feat_reg:
        if col.startswith("rg_"):
            X[col] = 1 if col == f"rg_{regime}" else 0

    for c in feat_reg:
        if c not in X.columns:
            X[c] = 0.0

    log_pred = model_rf.predict(X[feat_reg])
    X["corrected_mm"] = np.clip(np.expm1(log_pred), 0, None).round(2)
    X["gefs_mm"] = X["gefs_mm"].round(2)

    # Probabilities
    try:
        prob_bundle = joblib.load(MODELS / "heavy_rain_probability.joblib")
        for name in ["heavy", "very_heavy", "extremely_heavy"]:
            entry = prob_bundle.get(name)
            if not entry or entry.get("model") is None:
                continue
            clf = entry["model"]
            feats = entry["features"]
            Xp = X.copy()
            for f in feats:
                if f not in Xp.columns:
                    Xp[f] = 0.0
            p = clf.predict_proba(Xp[feats])[:, 1]
            X[f"P_{name}"] = p.round(4)
    except Exception as e:
        print(f"Probability model skipped: {e}")

    # Save
    X["date"] = target_date.strftime("%Y-%m-%d")
    X["regime_label"] = regime
    X = X.rename(columns={"corrected_mm": "pred_regime_feat"})

    out = REPORTS / "realtime_forecast.csv"
    X.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Districts: {len(X)}")
    print(f"Mean raw GEFS:       {X['gefs_mm'].mean():.2f} mm")
    print(f"Mean corrected:      {X['pred_regime_feat'].mean():.2f} mm")
    print(f"Districts >= 64.5mm: {(X['pred_regime_feat'] >= 64.5).sum()}")


if __name__ == "__main__":
    main()