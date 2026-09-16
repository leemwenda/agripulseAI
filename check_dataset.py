#!/usr/bin/env python3
"""
AgriPulse dataset verifier - checks MultiCamCows subset before training.
Run from anywhere; edit DATA_DIR below if your path differs.
"""
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Missing Pillow. Install with: pip install Pillow")
    sys.exit(1)

DATA_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_subset")

def main():
    root = Path(DATA_DIR)
    if not root.exists():
        print(f"Path not found: {root}")
        print("Edit DATA_DIR at the top of this script if your folder is elsewhere.")
        sys.exit(1)

    cow_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not cow_dirs:
        print(f"No cow subfolders found in {root}")
        sys.exit(1)

    print(f"Scanning: {root}\n")
    total_images = 0
    total_bad = 0
    counts = {}

    for cow_dir in cow_dirs:
        images = [f for f in cow_dir.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        good, bad = 0, 0
        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    im.verify()
                good += 1
            except Exception:
                bad += 1
                print(f"  [CORRUPT] {img_path}")
        counts[cow_dir.name] = good
        total_images += good
        total_bad += bad
        flag = "  <-- too few for train/val/test split (need 6+)" if good < 6 else ""
        print(f"{cow_dir.name}: {good} good, {bad} corrupt{flag}")

    print(f"\n{'='*50}")
    print(f"Total cows: {len(cow_dirs)}")
    print(f"Total good images: {total_images}")
    print(f"Total corrupt/unreadable: {total_bad}")
    if total_bad > 0:
        print(f"\n⚠ {total_bad} corrupt file(s) found - remove or re-download these before training.")

    thin = [c for c, n in counts.items() if n < 6]
    if thin:
        print(f"\n⚠ These cows have <6 images (too thin for a train/val/test split): {', '.join(thin)}")
    else:
        print("\nAll cows have enough images for a basic train/val/test split.")

if __name__ == "__main__":
    main()
