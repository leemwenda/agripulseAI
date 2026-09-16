#!/usr/bin/env python3
"""
AgriPulse dataset balancer - caps images per cow to avoid class imbalance
before training. Creates symlinks in a new folder, leaves original data untouched.
"""
import os
import random
import sys
from pathlib import Path

SOURCE_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_subset")
OUTPUT_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_balanced")
MAX_PER_COW = 150
SEED = 42

def main():
    src_root = Path(SOURCE_DIR)
    out_root = Path(OUTPUT_DIR)

    if not src_root.exists():
        print(f"Source not found: {src_root}")
        sys.exit(1)

    random.seed(SEED)
    out_root.mkdir(parents=True, exist_ok=True)

    cow_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    print(f"Balancing dataset: cap {MAX_PER_COW} images/cow\n")

    total_before = 0
    total_after = 0

    for cow_dir in cow_dirs:
        images = sorted([f for f in cow_dir.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        total_before += len(images)

        if len(images) > MAX_PER_COW:
            selected = random.sample(images, MAX_PER_COW)
        else:
            selected = images

        out_cow_dir = out_root / cow_dir.name
        out_cow_dir.mkdir(parents=True, exist_ok=True)

        # clear any stale symlinks from a previous run
        for existing in out_cow_dir.iterdir():
            if existing.is_symlink() or existing.is_file():
                existing.unlink()

        for img_path in selected:
            link_path = out_cow_dir / img_path.name
            if not link_path.exists():
                os.symlink(img_path.resolve(), link_path)

        total_after += len(selected)
        print(f"{cow_dir.name}: {len(images)} -> {len(selected)}")

    print(f"\n{'='*50}")
    print(f"Total before: {total_before}")
    print(f"Total after:  {total_after}")
    print(f"\nBalanced dataset (symlinks) written to: {out_root}")
    print("Original data untouched. Point your notebook's data path at the new folder.")

if __name__ == "__main__":
    main()
