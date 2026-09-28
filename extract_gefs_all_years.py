"""
Extract India rainfall from all GEFS reforecast files (2016-2019).
Reads APCP GRIB2 files, extracts 24h accumulated precipitation over India,
maps grid points to districts, writes one combined CSV.
"""
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

GEFS_DIR = Path(__file__).parent / "data" / "gefs"
MAPPING_CSV = Path(__file__).parent / "data" / "grid_to_district.csv"
OUT_DIR = Path(__file__).parent / "data" / "gefs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 5.0, 40.0
LON_MIN, LON_MAX = 60.0, 105.0


def open_gefs(path):
    return xr.open_dataset(
        path, engine="cfgrib",
        backend_kwargs={"indexpath": "", "filter_by_keys": {"typeOfLevel": "surface"}},
    )


def extract_file(path, target_date):
    ds = open_gefs(path)
    sub = ds.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
    var = list(sub.data_vars)[0]
    tp = sub[var]
    # Handle APCP step naming
    if "step" in tp.dims:
        try:
            tp24 = tp.sel(step=pd.Timedelta(hours=24))
        except Exception:
            tp24 = tp.isel(step=-1)
    else:
        tp24 = tp

    df = tp24.to_dataframe(name="gefs_mm").reset_index()
    df["latitude"] = df["latitude"].round(2)
    df["longitude"] = df["longitude"].round(2)
    df["gefs_mm"] = df["gefs_mm"].clip(lower=0)
    df["date"] = pd.Timestamp(target_date)
    ds.close()
    return df[["date", "latitude", "longitude", "gefs_mm"]]


def main():
    print("Loading district mapping...")
    mapping = pd.read_csv(MAPPING_CSV).dropna(subset=["district_id"]).copy()
    mapping["district_id"] = mapping["district_id"].astype(int).astype(str)
    mapping["latitude"] = mapping["latitude"].round(2)
    mapping["longitude"] = mapping["longitude"].round(2)

    all_rows = []
    total_files = 0

    for year_dir in sorted(GEFS_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        files = sorted(year_dir.glob("apcp_sfc_*.grib2"))
        print(f"\n=== {year_dir.name}: {len(files)} files ===")

        for i, f in enumerate(files, 1):
            # Filename: apcp_sfc_YYYYMMDD00_c00.grib2
            parts = f.name.split("_")
            yyyymmdd = parts[2][:8]
            try:
                target = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
                df = extract_file(f, target)
                all_rows.append(df)
            except Exception as e:
                print(f"  FAIL {f.name}: {str(e)[:80]}")

            if i % 30 == 0:
                print(f"  [{i}/{len(files)}] processed", flush=True)
            total_files += 1

    if not all_rows:
        print("No data extracted.")
        return

    print(f"\nMerging {len(all_rows)} days...")
    big = pd.concat(all_rows, ignore_index=True)
    print(f"Grid rows before district map: {len(big):,}")

    print("Mapping grid to districts...")
    merged = big.merge(
        mapping[["latitude", "longitude", "district_id"]],
        on=["latitude", "longitude"], how="inner",
    )
    print(f"Rows after district map: {len(merged):,}")

    print("Aggregating to district level...")
    district = merged.groupby(["date", "district_id"], as_index=False)["gefs_mm"].mean()

    out = OUT_DIR / "gefs_india_2016_2019.csv"
    district.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Saved: {out}")
    print(f"Rows: {len(district):,}")
    print(f"Dates: {district['date'].nunique()}")
    print(f"Districts: {district['district_id'].nunique()}")
    print(f"Date range: {district['date'].min()} -> {district['date'].max()}")
    print(f"Mean rainfall: {district['gefs_mm'].mean():.3f} mm")


if __name__ == "__main__":
    main()