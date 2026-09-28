"""
Replace old OneDrive paths with new C:\projects paths in all project files.
Run once from the project root.
"""
from pathlib import Path

ROOT = Path(r"C:\projects\sih2b")

REPLACEMENTS = [
    (r"C:\Users\SARA\OneDrive\Desktop\sih2b",           r"C:\projects\sih2b"),
    (r"C:\Users\SARA\OneDrive\Desktop\IMD_Rainfall",    r"C:\projects\IMD_Rainfall"),
    ("C:/Users/SARA/OneDrive/Desktop/sih2b",            "C:/projects/sih2b"),
    ("C:/Users/SARA/OneDrive/Desktop/IMD_Rainfall",     "C:/projects/IMD_Rainfall"),
]

EXTS = {".py", ".yaml", ".yml", ".html", ".json", ".md", ".txt"}

changed_files = []
total_replacements = 0

for f in ROOT.rglob("*"):
    if not f.is_file():
        continue
    if f.suffix.lower() not in EXTS:
        continue
    if f.name == "fix_paths.py":
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
            total_replacements += count
            print(f"  {f.relative_to(ROOT)}: {count} replacement(s)")

    if text != original:
        f.write_text(text, encoding="utf-8")
        changed_files.append(f.relative_to(ROOT))

print()
print("=" * 60)
print(f"Files modified: {len(changed_files)}")
print(f"Total replacements: {total_replacements}")
print("=" * 60)