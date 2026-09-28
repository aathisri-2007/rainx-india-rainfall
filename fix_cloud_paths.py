"""
Replace all hardcoded C:\projects paths with relative paths.
Run once before deploying to cloud.
"""
from pathlib import Path

ROOT = Path(r"C:\projects\sih2b")

# Mapping: old path → new relative expression
REPLACEMENTS = [
    # Sih2b project root → BASE (assumes each file defines BASE or has access to Path(__file__))
    ('Path(r"C:\\projects\\sih2b")',                    'Path(__file__).parent'),
    ('Path(r"C:\\projects\\sih2b\\reports")',           'Path(__file__).parent / "reports"'),
    ('Path(r"C:\\projects\\sih2b\\models")',            'Path(__file__).parent / "models"'),
    ('Path(r"C:\\projects\\sih2b\\data\\admin\\district_nwic.GeoJSON")',
     'Path(__file__).parent / "data" / "admin" / "district_nwic.GeoJSON"'),
    ('Path(r"C:\\projects\\sih2b\\data\\admin\\district_nwic_wgs84.geojson")',
     'Path(__file__).parent / "data" / "admin" / "district_nwic_wgs84.geojson"'),

    # IMD_Rainfall → project data folder
    ('Path(r"C:\\projects\\IMD_Rainfall\\gefs_india\\gefs_india_2016_2019.csv")',
     'Path(__file__).parent / "data" / "gefs_india_2016_2019.csv"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\gefs_india\\gefs_india_2016_monsoon.csv")',
     'Path(__file__).parent / "data" / "gefs_india_2016_monsoon.csv"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\gefs_india")',
     'Path(__file__).parent / "data" / "gefs"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\converted\\district_daily_rainfall_2016_2025.csv")',
     'Path(__file__).parent / "data" / "district_daily_rainfall_2016_2025.csv"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\converted\\grid_to_district.csv")',
     'Path(__file__).parent / "data" / "grid_to_district.csv"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\converted\\imd_rainfall_2016_2025.csv")',
     'Path(__file__).parent / "data" / "imd_rainfall_2016_2025.csv"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\converted")',
     'Path(__file__).parent / "data"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\regimes\\regime_labels_2016_2019.csv")',
     'Path(__file__).parent / "data" / "regimes" / "regime_labels_2016_2019.csv"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\regimes\\regime_labels_2016.csv")',
     'Path(__file__).parent / "data" / "regimes" / "regime_labels_2016.csv"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\regimes")',
     'Path(__file__).parent / "data" / "regimes"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\reanalysis")',
     'Path(__file__).parent / "data" / "reanalysis"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\GEFS")',
     'Path(__file__).parent / "data" / "gefs"'),
    ('Path(r"C:\\projects\\IMD_Rainfall\\herbie_cache")',
     'Path(__file__).parent / "cache"'),
    ('Path(r"C:\\projects\\IMD_Rainfall")',
     'Path(__file__).parent / "data"'),

    # Raw string versions (for r"..." without Path())
    ('r"C:\\projects\\IMD_Rainfall\\converted\\grid_to_district.csv"',
     'str(Path(__file__).parent / "data" / "grid_to_district.csv")'),
    ('r"C:\\projects\\IMD_Rainfall\\converted\\district_daily_rainfall_2016_2025.csv"',
     'str(Path(__file__).parent / "data" / "district_daily_rainfall_2016_2025.csv")'),
    ('r"C:\\projects\\IMD_Rainfall\\converted\\imd_rainfall_2016_2025.csv"',
     'str(Path(__file__).parent / "data" / "imd_rainfall_2016_2025.csv")'),
]

SKIP_FILES = {"fix_cloud_paths.py", "fix_paths.py", "api.py"}  # api.py already done

files_changed = 0
total_subs = 0

for f in ROOT.rglob("*.py"):
    if f.name in SKIP_FILES:
        continue
    if "scripts" in f.parts:  # scripts folder is dev-only, skip
        continue

    try:
        text = f.read_text(encoding="utf-8")
    except Exception:
        continue

    original = text
    for old, new in REPLACEMENTS:
        if old in text:
            count = text.count(old)
            text = text.replace(old, new)
            total_subs += count
            print(f"  {f.relative_to(ROOT)}: {count} × {old[:50]}...")

    if text != original:
        f.write_text(text, encoding="utf-8")
        files_changed += 1

print()
print("=" * 60)
print(f"Files modified: {files_changed}")
print(f"Total replacements: {total_subs}")
print("=" * 60)
print()
print("NOTE: Some files may still have 'C:\\projects' references.")
print("Run this check to see what's left:")
print("  findstr /s /i \"C:\\\\projects\" *.py")