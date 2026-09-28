"""
Download NCEP/NCAR Reanalysis 2016-2019 — one at a time with resume + retries.
"""
from pathlib import Path
import requests
import time

OUT = Path(__file__).parent / "data" / "reanalysis"
OUT.mkdir(parents=True, exist_ok=True)

BASE_SFC = "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/surface"
BASE_PRS = "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/pressure"

YEARS = [2016, 2017, 2018, 2019]

VARS = [
    ("air_2m",   BASE_SFC, "air.sig995.{y}.nc"),
    ("rhum",     BASE_SFC, "rhum.sig995.{y}.nc"),
    ("slp",      BASE_SFC, "slp.{y}.nc"),
    ("uwnd_10m", BASE_SFC, "uwnd.sig995.{y}.nc"),
    ("vwnd_10m", BASE_SFC, "vwnd.sig995.{y}.nc"),
    ("hgt_500",  BASE_PRS, "hgt.{y}.nc"),
]

EXPECTED_MIN_BYTES = 8_000_000   # files should be ~10 MB; under 8 MB = incomplete


def download_resumable(url, out, max_attempts=10):
    """Download with resume + retries. Returns 'ok', 'skip', or error message."""
    for attempt in range(1, max_attempts + 1):
        current_size = out.stat().st_size if out.exists() else 0

        # If already complete, skip
        if current_size >= EXPECTED_MIN_BYTES:
            # Do one HEAD to check full size? Skip for now.
            return "skip"

        headers = {}
        if current_size > 0:
            headers["Range"] = f"bytes={current_size}-"

        try:
            r = requests.get(url, stream=True, timeout=60, headers=headers)
            if r.status_code == 416:
                return "ok"  # already complete
            if r.status_code not in (200, 206):
                time.sleep(2)
                continue

            mode = "ab" if (current_size > 0 and r.status_code == 206) else "wb"
            if mode == "wb":
                current_size = 0

            with open(out, mode) as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        f.write(chunk)

            # Check if we got enough
            if out.stat().st_size >= EXPECTED_MIN_BYTES:
                return "ok"

            # Partial — loop and resume
            print(f"    partial {out.stat().st_size/1e6:.1f} MB, retrying...", flush=True)
            time.sleep(2)

        except Exception as e:
            print(f"    attempt {attempt} error: {str(e)[:60]}", flush=True)
            time.sleep(3)

    return f"failed after {max_attempts} attempts"


def main():
    total = len(YEARS) * len(VARS)
    print(f"Downloading {total} files (sequential with resume)\n")

    ok = skip = fail = 0
    for i, year in enumerate(YEARS, 1):
        print(f"=== Year {year} ({i}/{len(YEARS)}) ===")
        for name, base, pattern in VARS:
            url = f"{base}/{pattern.format(y=year)}"
            out = OUT / f"{name}_{year}.nc"
            print(f"  {out.name}...", end=" ", flush=True)
            result = download_resumable(url, out)
            if result == "ok":
                ok += 1
                print("OK", flush=True)
            elif result == "skip":
                skip += 1
                print("already complete", flush=True)
            else:
                fail += 1
                print(result, flush=True)

    print(f"\n{'='*50}")
    print(f"DONE. ok={ok} skip={skip} fail={fail}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()