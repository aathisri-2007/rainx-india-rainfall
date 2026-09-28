from pathlib import Path
import xarray as xr

RAW = Path(__file__).parent / "data" / "reanalysis"
src = RAW / "hgt_500_full.nc"
dst = RAW / "hgt_500.nc"

print("Opening:", src)
print("Exists:", src.exists())

if not src.exists():
    raise SystemExit(
        "\nFull file not found. Download it first from:\n"
        "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/pressure/hgt.2016.nc\n"
        f"Save as: {src}\n"
    )

size_mb = src.stat().st_size / (1024 * 1024)
print(f"Size: {size_mb:.1f} MB")

if size_mb < 50:
    raise SystemExit(
        f"\nFile too small ({size_mb:.1f} MB). Browser download may not be complete.\n"
        "Delete it and re-download fully."
    )

ds = xr.open_dataset(src)
print("\nDims:", dict(ds.sizes))
print("Coords:", list(ds.coords))
print("Vars:", list(ds.data_vars))

# Identify lat / lon coordinate names
lat_name = "lat" if "lat" in ds.coords else "latitude"
lon_name = "lon" if "lon" in ds.coords else "longitude"

LAT_MIN, LAT_MAX = 5.0, 40.0
LON_MIN, LON_MAX = 60.0, 105.0

lat_vals = ds[lat_name].values
if lat_vals[0] > lat_vals[-1]:
    sel = {lat_name: slice(LAT_MAX, LAT_MIN),
           lon_name: slice(LON_MIN, LON_MAX)}
else:
    sel = {lat_name: slice(LAT_MIN, LAT_MAX),
           lon_name: slice(LON_MIN, LON_MAX)}

sub = ds.sel(**sel)

# Select 500 hPa level if the file has a level dimension
if "level" in sub.dims or "level" in sub.coords:
    levels = sub["level"].values if "level" in sub.coords else None
    print("Levels available:", levels)
    if levels is not None and 500.0 in levels:
        sub = sub.sel(level=500.0)
        print("Selected 500 hPa level")
    elif levels is not None:
        # pick closest to 500
        import numpy as np
        idx = np.argmin(np.abs(levels - 500.0))
        print(f"500 not exact, closest: {levels[idx]}")
        sub = sub.sel(level=levels[idx])

# Rename to standard for downstream
ren = {}
if lat_name != "latitude": ren[lat_name] = "latitude"
if lon_name != "longitude": ren[lon_name] = "longitude"
if ren:
    sub = sub.rename(ren)

sub.to_netcdf(dst)
print("\nSaved:", dst)
print("India slice dims:", dict(sub.sizes))
print("Coords:", list(sub.coords))
print("Vars:", list(sub.data_vars))