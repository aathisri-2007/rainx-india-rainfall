"""Verify that all reanalysis files can be opened by xarray."""
from pathlib import Path
import xarray as xr

REAN = Path(__file__).parent / "data" / "reanalysis"
YEARS = [2016, 2017, 2018, 2019]
VARS = ["air_2m", "rhum", "slp", "uwnd_10m", "vwnd_10m", "hgt_500"]

good, bad = [], []

for year in YEARS:
    for var in VARS:
        p = REAN / f"{var}_{year}.nc"
        if not p.exists():
            bad.append((p.name, "missing"))
            print(f"❌ {p.name}: MISSING")
            continue
        size_mb = p.stat().st_size / (1024 * 1024)
        try:
            ds = xr.open_dataset(p)
            _ = ds[list(ds.data_vars)[0]].shape
            n_time = ds.sizes.get("time", "?")
            ds.close()
            good.append(p.name)
            print(f"✅ {p.name}: {size_mb:.1f} MB, time={n_time}")
        except Exception as e:
            bad.append((p.name, str(e)[:60]))
            print(f"❌ {p.name}: {size_mb:.1f} MB — {str(e)[:60]}")

print(f"\n{'='*60}")
print(f"Good: {len(good)}  |  Bad: {len(bad)}")
print(f"{'='*60}")

if bad:
    print("\nBroken files (need re-download):")
    for name, err in bad:
        print(f"  {name}  →  {err}")
    # Write a list for the fix script
    with open(REAN / "_broken.txt", "w") as f:
        f.write("\n".join(name for name, _ in bad))
    print(f"\nSaved broken list to: {REAN / '_broken.txt'}")