#!/usr/bin/env python3
"""
Merges multicamcows_subset + OpenCows2020 into one combined_dataset/ folder
using symlinks (no duplicate storage). train_local_v2.py / check_dataset.py
just need DATA_DIR pointed at combined_dataset afterward.
"""
import os
from pathlib import Path

BASE = Path(os.path.expanduser("~/agripulse-ai"))
MULTICAM_DIR = BASE / "multicamcows_subset"
OPENCOWS_ROOT = BASE / "opencows2020" / "extracted"
COMBINED_DIR = BASE / "combined_dataset"

COMBINED_DIR.mkdir(exist_ok=True)

def symlink_cow_folder(src_dir, dest_name):
    dest = COMBINED_DIR / dest_name
    if dest.exists() or dest.is_symlink():
        return False
    os.symlink(src_dir.resolve(), dest)
    return True

# ---- 1. MultiCamCows (already correct structure, just link each cow folder) ----
added = 0
for cow_dir in sorted(MULTICAM_DIR.iterdir()):
    if cow_dir.is_dir():
        if symlink_cow_folder(cow_dir, cow_dir.name):
            added += 1
print(f"Linked {added} MultiCamCows cow folders.")

# ---- 2. OpenCows2020 — find identification/images, merge train+test per id ----
candidates = list(OPENCOWS_ROOT.rglob("identification/images"))
if not candidates:
    print(f"ERROR: could not find identification/images under {OPENCOWS_ROOT}")
    raise SystemExit(1)

images_root = candidates[0]
print(f"Found OpenCows2020 images at: {images_root}")

merged = {}  # id -> list of image dirs (could be train and/or test dir)
for split in ("train", "test"):
    split_dir = images_root / split
    if not split_dir.exists():
        continue
    for id_dir in sorted(split_dir.iterdir()):
        if not id_dir.is_dir():
            continue
        merged.setdefault(id_dir.name, []).append(id_dir)

added = 0
for cow_id, dirs in merged.items():
    dest_name = f"opencows_{cow_id}"
    dest = COMBINED_DIR / dest_name
    if dest.exists() or dest.is_symlink():
        continue
    if len(dirs) == 1:
        # only train OR only test for this id - link directly
        os.symlink(dirs[0].resolve(), dest)
    else:
        # both train and test - make a real dir with symlinked images inside
        dest.mkdir()
        for d in dirs:
            for img in d.glob("*.jpg"):
                link = dest / img.name
                if not link.exists():
                    os.symlink(img.resolve(), link)
    added += 1

print(f"Linked {added} OpenCows2020 cow folders (as opencows_<id>).")
print(f"\nCombined dataset ready at: {COMBINED_DIR}")
print("Next: point DATA_DIR in train_local_v2.py / check_dataset.py at combined_dataset,")
print("then re-run check_dataset.py before training.")
