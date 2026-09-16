#!/usr/bin/env python3
"""
Downloads OpenCows2020 (46 individual Holstein-Friesian cows, dairy, top-down view).
Hosted on University of Bristol's data.bris repository - direct download, no bot-blocking.
License: Non-Commercial Government Licence (research/prototyping use is fine).
"""
import os
import subprocess
import sys

DOWNLOAD_URL = "https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip"
OUTPUT_DIR = os.path.expanduser("~/agripulse-ai/opencows2020")
ZIP_PATH = os.path.join(OUTPUT_DIR, "OpenCows2020.zip")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading OpenCows2020 (2.1 GiB) to {ZIP_PATH}")
    print("This will take a while depending on your connection - progress shown below.\n")

    # -c resumes a partial download if this gets interrupted and re-run
    result = subprocess.run(
        ["wget", "-c", DOWNLOAD_URL, "-O", ZIP_PATH],
    )

    if result.returncode != 0:
        print("\nDownload failed or was interrupted. Re-run this script to resume "
              "(wget -c continues from where it left off).")
        sys.exit(1)

    print(f"\nDownload complete: {ZIP_PATH}")
    size_gb = os.path.getsize(ZIP_PATH) / (1024 ** 3)
    print(f"File size: {size_gb:.2f} GiB")
    print(f"\nNext: unzip {ZIP_PATH} -d {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
