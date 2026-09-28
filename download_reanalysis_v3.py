"""
Download NCEP/NCAR Reanalysis 2016-2019 — tries multiple mirrors.
"""
from pathlib import Path
import requests
import time

OUT = Path(__file__).parent / "data" / "reanalysis"
OUT.mkdir(parents=True, exist_ok=True)

# Alternative mirror bases (tried in order)
MIRRORS = [
    # 1. Original NOAA PSL THREDDS
    ("https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/surface",
     "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/pressure"),
    # 2. Direct IP mirror of PSL
    ("https://140.172.38.220/thredds/fileServer/Datasets/ncep.reanalysis/surface",
     "https://140.172.38.220/thredds/fileServer/Datasets/ncep.reanalysis/pressure"),
    # 3. NOAA CPC backup
    ("https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface",
     "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure"),
    # 4. NOAA CPC alternate
    ("https://www.esrl.noaa.gov/psd/thredds/fileServer/Datasets/ncep.reanalysis/surface",
     "https://www.esrl.noaa.gov/psd/thredds/fileServer/Datasets/ncep.reanalysis/pressure"),
]

FILES = [
    ("air_2m",    "surface",  "air.sig995.{y}.nc"),
    ("rhum",      "surface",  "rhum.sig995.{y}.nc"),
    ("slp",       "surface",  "slp.{y}.nc"),
    ("uwnd_10m",  "surface",  "uwnd.sig995.{y}.nc"),
    ("vwnd_10m",  "surface",  "vwnd.sig995.{y}.nc"),
    ("hgt_500",   "pressure", "hgt.{y}.nc"),
]

YEARS = [2016, 2017, 2018, 2019]


def try_download(path_rel):
    """Try each mirror, return (response, working_url) or (None, None)."""
    for sfc_base, prs_base in MIRRORS:
        base = sfc_base if "surface" in path_rel else prs_base
        url = base + "/" + path_rel.split("/", 1)[1]
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                return r, url
        except Exception:
            continue
    return None, None


def download_one(fname, kind, pattern, year):
    out = OUT / f"{fname}_{year}.nc"
    if out.exists() and out.stat().st_size > 8_000_000:
        return ("skip", out.stat().st_size, None)

    rel = f"{kind}/{pattern.format(y=year)}"

    for sfc_base, prs_base in MIRRORS:
        base = sfc_base if kind == "surface" else prs_base
        url = f"{base}/{pattern.format(y=year)}"
        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                continue
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    if chunk:
                        f.write(chunk)
            size = out.stat().st_size
            if size > 8_000_000:
                return ("ok", size, url.split("//")[1].split("/")[0])
            else:
                out.unlink()
        except Exception:
            continue
    return ("fail", 0, None)


def main():
    ok = skip = fail = 0
    working_mirror = None

    for year in YEARS:
        print(f"\n=== {year} ===", flush=True)
        for fname, kind, pattern in FILES:
            print(f"  {fname}_{year}.nc ...", end=" ", flush=True)
            status, size, mirror = download_one(fname, kind, pattern, year)
            if status == "ok":
                ok += 1
                if not working_mirror:
                    working_mirror = mirror
                print(f"OK ({size/1e6:.1f} MB) via {mirror}", flush=True)
            elif status == "skip":
                skip += 1
                print(f"complete ({size/1e6:.1f} MB)", flush=True)
            else:
                fail += 1
                print("FAIL", flush=True)

    print(f"\n{'='*50}")
    print(f"DONE. ok={ok} skip={skip} fail={fail}")
    if working_mirror:
        print(f"Working mirror: {working_mirror}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()