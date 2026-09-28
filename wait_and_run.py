"""Retry until NOAA PSL comes back, then run build_regime_labels_multiyear.py."""
import subprocess, sys, time
from pathlib import Path

BASE = Path(__file__).parent

for attempt in range(20):  # up to ~40 minutes
    print(f"\n=== Attempt {attempt+1}/20 ===")
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'core'); "
         "from district_features import load_reanalysis; "
         "ds = load_reanalysis(None, 'slp', years=[2016]); "
         "print('OK', ds.sizes['time'])"],
        cwd=str(BASE), capture_output=True, text=True
    )
    if r.returncode == 0 and "OK" in r.stdout:
        print("NOAA is back! Running build_regime_labels_multiyear.py...")
        subprocess.run([sys.executable, "build_regime_labels_multiyear.py"], cwd=str(BASE))
        break
    print("  Still down. Waiting 2 minutes...")
    time.sleep(120)
else:
    print("NOAA still down after 40 minutes. Try again later.")