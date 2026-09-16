#!/usr/bin/env python3
"""
Zips the balanced MultiCamCows subset for Colab upload.
Resolves symlinks to real file copies so the zip isn't broken.
"""
import os
import shutil
import zipfile
from pathlib import Path

BALANCED_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_balanced")
ZIP_OUTPUT = os.path.expanduser("~/agripulse-ai/multicamcows_balanced.zip")

def main():
    src_root = Path(BALANCED_DIR)
    if not src_root.exists():
        print(f"Not found: {src_root}")
        print("Run balance_dataset.py first.")
        return

    zip_path = Path(ZIP_OUTPUT)
    if zip_path.exists():
        zip_path.unlink()

    print(f"Zipping {src_root} -> {zip_path}")
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for cow_dir in sorted(src_root.iterdir()):
            if not cow_dir.is_dir():
                continue
            for img_path in sorted(cow_dir.iterdir()):
                # resolve() follows the symlink to the real file on disk
                real_path = img_path.resolve()
                if not real_path.exists():
                    print(f"  [SKIP - broken link] {img_path}")
                    continue
                arcname = f"{cow_dir.name}/{img_path.name}"
                zf.write(real_path, arcname)
                file_count += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nDone: {file_count} files zipped")
    print(f"Zip size: {size_mb:.1f} MB")
    print(f"Upload this file to Colab: {zip_path}")

if __name__ == "__main__":
    main()
