"""Per-district atmospheric features from a gridded dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def fix_lon(ds):
    if "longitude" in ds.coords and float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")
    return ds


def load_grid_mapping(mapping_csv):
    m = pd.read_csv(mapping_csv).dropna(subset=["district_id"]).copy()
    m["district_id"] = m["district_id"].astype(int).astype(str)
    m["latitude"] = m["latitude"].round(2)
    m["longitude"] = m["longitude"].round(2)
    return m


def load_reanalysis(rean_dir, name, years=None):
    """
    Load NCEP/NCAR reanalysis — tries multiple servers automatically.
    """
    # Try each mirror until one works
    MIRRORS = [
        "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis",
        "https://140.172.38.220/thredds/dodsC/Datasets/ncep.reanalysis",
        "https://downloads.psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis",
    ]

    paths = {
        "slp":      "surface/slp.{year}.nc",
        "uwnd_10m": "surface/uwnd.sig995.{year}.nc",
        "vwnd_10m": "surface/vwnd.sig995.{year}.nc",
        "rhum":     "surface/rhum.sig995.{year}.nc",
        "air_2m":   "surface/air.sig995.{year}.nc",
        "hgt_500":  "pressure/hgt.{year}.nc",
    }

    if name not in paths:
        print(f"    Unknown variable: {name}")
        return None

    years_to_use = years or [2016, 2017, 2018, 2019]
    parts = []

    for y in years_to_use:
        rel_path = paths[name].format(year=y)
        sub = None

        for base in MIRRORS:
            url = f"{base}/{rel_path}"
            try:
                ds = xr.open_dataset(url)

                # Rename coords
                ren = {}
                for c in list(ds.coords):
                    if c == "lat": ren[c] = "latitude"
                    if c == "lon": ren[c] = "longitude"
                if ren:
                    ds = ds.rename(ren)

                # Normalize longitude
                if float(ds.longitude.max()) > 180:
                    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
                    ds = ds.sortby("longitude")

                # Slice to India + monsoon
                latvals = ds.latitude.values
                ls = slice(40, 5) if latvals[0] > latvals[-1] else slice(5, 40)
                sub = ds.sel(latitude=ls, longitude=slice(60, 105))
                sub = sub.sel(time=sub.time.dt.month.isin([6, 7, 8, 9]))

                # 500 hPa level
                if name == "hgt_500" and ("level" in sub.dims or "level" in sub.coords):
                    try:
                        sub = sub.sel(level=500.0)
                    except Exception:
                        pass

                # Keep only 1 variable + load + resample
                var_name = list(sub.data_vars)[0]
                sub = sub[[var_name]].load()
                sub = sub.resample(time="1D").mean()

                mirror_host = base.split("//")[1].split("/")[0]
                print(f"    {name} {y}: {sub.sizes.get('time','?')} days (via {mirror_host})")
                break

            except Exception as e:
                sub = None
                continue

        if sub is None:
            print(f"    FAILED {name} {y} on all mirrors")

        if sub is not None:
            parts.append(sub)

    if not parts:
        return None

    if len(parts) > 1:
        ds = xr.concat(parts, dim="time", data_vars="minimal",
                       coords="minimal", compat="override")
    else:
        ds = parts[0]

    _, idx = np.unique(ds.time.values, return_index=True)
    return ds.isel(time=idx)


def monsoon_climatology(ds, var):
    if ds is None:
        return None
    try:
        da = ds[var] if var in ds.data_vars else ds[list(ds.data_vars)[0]]
        return float(da.mean().values)
    except Exception:
        return None


def district_mean(da, cells):
    if not cells:
        return np.nan
    lats = [c[0] for c in cells]
    lons = [c[1] for c in cells]
    try:
        sub = da.sel(
            latitude=xr.DataArray(lats, dims="cell"),
            longitude=xr.DataArray(lons, dims="cell"),
            method="nearest",
        )
        return float(sub.mean().values)
    except Exception:
        return np.nan


def per_district_features(slp_da, u_da, v_da, rh_da, hgt_da,
                          mapping, slp_clim, rh_clim, hgt_clim):
    groups = mapping.groupby("district_id")[["latitude", "longitude"]].apply(
        lambda g: list(zip(g.latitude.values, g.longitude.values))
    )
    cells_by_district = groups.to_dict()

    rows = []
    for did, cells in cells_by_district.items():
        slp_val = district_mean(slp_da, cells)
        u_val   = district_mean(u_da,   cells)
        v_val   = district_mean(v_da,   cells)
        rh_val  = district_mean(rh_da,  cells)
        hgt_val = district_mean(hgt_da, cells)

        if np.isnan(u_val) or np.isnan(v_val):
            wind_from, wind_speed = np.nan, np.nan
        else:
            wind_from = float((270 - np.degrees(np.arctan2(v_val, u_val))) % 360)
            wind_speed = float(np.sqrt(u_val**2 + v_val**2))

        rows.append({
            "district_id":    did,
            "slp_anomaly":    (slp_val - slp_clim) / 100.0 if (slp_clim is not None and not np.isnan(slp_val)) else np.nan,
            "hgt500_anomaly": (hgt_val - hgt_clim) if (hgt_clim is not None and not np.isnan(hgt_val)) else np.nan,
            "wind_dir_from":  wind_from,
            "wind_speed_ms":  wind_speed,
            "rh_anomaly":     (rh_val - rh_clim) if (rh_clim is not None and not np.isnan(rh_val)) else np.nan,
        })

    df = pd.DataFrame(rows)
    df["monsoon_index"] = (
        -df["slp_anomaly"].fillna(0) * 0.6
        + df["rh_anomaly"].fillna(0) * 0.3
        + (df["wind_speed_ms"].fillna(9) - 9) * 0.4
    )
    return df