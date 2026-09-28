"""
Download GEFS v12 reforecast (APCP only) for 2017-2019 monsoon.
Same URL structure as your 2016 files, so pipeline works unchanged.
Fast: uses .idx byte-range to fetch only APCP.
"""
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
import time

OUT = Path(__file__).parent / "data" / "gefs"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
YEARS = [2017, 2018, 2019]


def download_one(dt_str):
    y, m, d = dt_str.split("-")
    ds = f"{y}{m}{d}"
    cycle = "00"
    member = "c00"

    idx_url  = f"{BASE}/{y}/{ds}{cycle}/{member}/Days:1-10/apcp_sfc_{ds}{cycle}_{member}.grib2.idx"
    grib_url = f"{BASE}/{y}/{ds}{cycle}/{member}/Days:1-10/apcp_sfc_{ds}{cycle}_{member}.grib2"

    out_file = OUT / str(y) / f"apcp_sfc_{ds}{cycle}_{member}.grib2"
    if out_file.exists() and out_file.stat().st_size > 5000:
        return (dt_str, "skip", None)

    try:
        # If .idx doesn't exist, fall back to downloading whole file (~20 MB)
        r = requests.get(idx_url, timeout=30)
        if r.status_code != 200:
            # Full file fallback
            r2 = requests.get(grib_url, stream=True, timeout=180)
            if r2.status_code != 200:
                return (dt_str, f"full:{r2.status_code}", None)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                for chunk in r2.iter_content(chunk_size=1024*1024):
                    if chunk: f.write(chunk)
            return (dt_str, "ok", out_file.stat().st_size)

        # Use .idx byte range
        lines = r.text.strip().split("\n")
        start_byte = None
        end_byte = None
        for i, line in enumerate(lines):
            if ":APCP:surface:" in line:
                start_byte = int(line.split(":")[1])
                if i + 1 < len(lines):
                    end_byte = int(lines[i + 1].split(":")[1]) - 1
                break

        if start_byte is None:
            return (dt_str, "no-apcp", None)

        headers = {"Range": f"bytes={start_byte}-{end_byte if end_byte else ''}"}
        r2 = requests.get(grib_url, headers=headers, timeout=60)
        if r2.status_code not in (200, 206):
            return (dt_str, f"grib:{r2.status_code}", None)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as f:
            f.write(r2.content)

        return (dt_str, "ok", out_file.stat().st_size)
    except Exception as e:
        return (dt_str, f"err:{str(e)[:60]}", None)


def main():
    tasks = []
    for y in YEARS:
        cur = date(y, 6, 1)
        end = date(y, 9, 30)
        while cur <= end:
            tasks.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)

    print(f"Total days: {len(tasks)}  (parallel workers: 16)")
    print(f"Source: GEFS v12 reforecast (same URL as your 2016 files)\n")

    ok = skip = fail = 0
    total_bytes = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(download_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            dt_str, status, size = fut.result()
            if status == "ok":
                ok += 1
                total_bytes += (size or 0)
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                if fail <= 10:   # only print first 10 failures
                    print(f"  {dt_str}: {status}")
            if i % 30 == 0:
                elapsed = max(time.time() - t0, 0.1)
                rate = i / elapsed * 60
                mb = total_bytes / (1024 * 1024)
                print(f"[{i}/{len(tasks)}] ok={ok} skip={skip} fail={fail} "
                      f"({rate:.0f}/min, {mb:.1f} MB)", flush=True)

    print(f"\nDONE in {time.time()-t0:.0f}s. ok={ok} skip={skip} fail={fail}")
    print(f"Total downloaded: {total_bytes/(1024*1024):.1f} MB")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()