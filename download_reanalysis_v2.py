"""
Download NCEP/NCAR Reanalysis 2016-2019 — one file at a time, verify size against server.
"""
from pathlib import Path
import requests

OUT = Path(__file__).parent / "data" / "reanalysis"
OUT.mkdir(parents=True, exist_ok=True)

BASE_SFC = "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/surface"
BASE_PRS = "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/pressure"

FILES = [
    ("air_2m",    BASE_SFC, "air.sig995.{y}.nc"),
    ("rhum",      BASE_SFC, "rhum.sig995.{y}.nc"),
    ("slp",       BASE_SFC, "slp.{y}.nc"),
    ("uwnd_10m",  BASE_SFC, "uwnd.sig995.{y}.nc"),
    ("vwnd_10m",  BASE_SFC, "vwnd.sig995.{y}.nc"),
    ("hgt_500",   BASE_PRS, "hgt.{y}.nc"),
]

YEARS = [2016, 2017, 2018, 2019]


def get_server_size(url):
    """Return file size in bytes from HEAD request, or None."""
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        return int(r.headers.get("Content-Length", 0))
    except Exception:
        return None


def download_exact(url, out):
    """Download the full file, verifying against server Content-Length."""
    server_size = get_server_size(url)
    if server_size and out.exists() and out.stat().st_size == server_size:
        return "skip", server_size

    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            return f"http:{r.status_code}", 0

        with open(out, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        actual = out.stat().st_size
        if server_size and actual != server_size:
            return f"size-mismatch ({actual} vs {server_size})", actual
        return "ok", actual
    except Exception as e:
        return f"err:{str(e)[:50]}", 0


def main():
    ok = skip = fail = 0
    for year in YEARS:
        print(f"\n=== {year} ===", flush=True)
        for name, base, pattern in FILES:
            url = f"{base}/{pattern.format(y=year)}"
            out = OUT / f"{name}_{year}.nc"
            print(f"  {out.name}...", end=" ", flush=True)
            result, size = download_exact(url, out)
            if result == "ok":
                ok += 1
                print(f"OK ({size/1e6:.1f} MB)", flush=True)
            elif result == "skip":
                skip += 1
                print(f"complete ({size/1e6:.1f} MB)", flush=True)
            else:
                fail += 1
                print(f"FAIL — {result}", flush=True)

    print(f"\n{'='*50}")
    print(f"DONE. ok={ok} skip={skip} fail={fail}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()