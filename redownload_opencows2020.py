#!/usr/bin/env python3
"""
Re-downloads OpenCows2020 from scratch (not resumed, to avoid the corruption
that happened last time when a resumed download's bytes didn't stitch
together cleanly) and verifies the zip's integrity before declaring success.
"""
import os
import subprocess
import sys
import zipfile

DOWNLOAD_URL = "https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip"
OUTPUT_DIR = os.path.expanduser("~/agripulse-ai/opencows2020")
ZIP_PATH = os.path.join(OUTPUT_DIR, "OpenCows2020.zip")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(ZIP_PATH):
        print(f"Removing previous (corrupted) file: {ZIP_PATH}")
        os.remove(ZIP_PATH)

    print(f"Downloading OpenCows2020 (2.1 GiB) fresh - no resume this time")
    print("This will take a while - progress shown below.\n")

    # no -c this time - full fresh download to avoid resume-stitch corruption
    result = subprocess.run(["wget", DOWNLOAD_URL, "-O", ZIP_PATH])

    if result.returncode != 0:
        print("\nDownload failed or was interrupted.")
        sys.exit(1)

    print(f"\nDownload complete: {ZIP_PATH}")
    size_gb = os.path.getsize(ZIP_PATH) / (1024 ** 3)
    print(f"File size: {size_gb:.2f} GiB")

    print("\nVerifying zip integrity before extracting...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                print(f"CORRUPTED: bad file found in zip: {bad_file}")
                print("The download is still broken. Consider trying at a different "
                      "time of day, or on a more stable connection.")
                sys.exit(1)
            else:
                print("Zip integrity check PASSED - file is valid.")
                print(f"\nNext: run extract_opencows2020.py to unzip it.")
    except zipfile.BadZipFile as e:
        print(f"CORRUPTED: not a valid zip file at all: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
