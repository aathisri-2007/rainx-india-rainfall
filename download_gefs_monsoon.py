from pathlib import Path
import requests
from datetime import date, timedelta
import time

OUTPUT_DIR = Path(__file__).parent / "data" / "gefs"
BASE_URL = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"

START = date(2016, 6, 1)
END = date(2016, 9, 30)

def download_file(url, output_file):
    if output_file.exists() and output_file.stat().st_size > 1_000_000:
        return "skip"
    try:
        r = requests.get(url, stream=True, timeout=180)
        if r.status_code != 200:
            return f"fail:{r.status_code}"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return "ok"
    except Exception as e:
        return f"err:{e}"

def main():
    current = START
    total = ok = skip = fail = 0

    while current <= END:
        ds = current.strftime("%Y%m%d")
        dt = ds + "00"
        fn = f"apcp_sfc_{dt}_c00.grib2"
        url = f"{BASE_URL}/{current.year}/{dt}/c00/Days:1-10/{fn}"
        out = OUTPUT_DIR / str(current.year) / fn

        result = download_file(url, out)
        total += 1

        if result == "ok":
            ok += 1
            print(f"[{total}] downloaded {fn}", flush=True)
        elif result == "skip":
            skip += 1
        else:
            fail += 1
            print(f"[{total}] FAIL {fn} -> {result}", flush=True)

        current += timedelta(days=1)
        time.sleep(0.1)

    print(f"\nDONE. total={total} new={ok} skip={skip} fail={fail}")

if __name__ == "__main__":
    main()