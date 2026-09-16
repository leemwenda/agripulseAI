#!/usr/bin/env python3
"""
Extracts OpenCows2020.zip and reports what's inside (folder structure,
image counts) so we know how to point training/registration scripts at it.
"""
import os
import zipfile

ZIP_PATH = os.path.expanduser("~/agripulse-ai/opencows2020/OpenCows2020.zip")
EXTRACT_DIR = os.path.expanduser("~/agripulse-ai/opencows2020/extracted")

def main():
    if not os.path.exists(ZIP_PATH):
        print(f"Zip not found: {ZIP_PATH}")
        return

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    print(f"Extracting {ZIP_PATH}...")
    print("(2.1GiB - this may take a few minutes)")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)

    print(f"\nExtracted to: {EXTRACT_DIR}\n")
    print("Top-level contents:")
    for item in sorted(os.listdir(EXTRACT_DIR)):
        item_path = os.path.join(EXTRACT_DIR, item)
        if os.path.isdir(item_path):
            n_files = sum(len(files) for _, _, files in os.walk(item_path))
            print(f"  [DIR]  {item}/  ({n_files} files inside)")
        else:
            print(f"  [FILE] {item}")

    print("\nFirst few levels of structure (for reference):")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        depth = root.replace(EXTRACT_DIR, "").count(os.sep)
        if depth > 2:
            dirs[:] = []  # don't descend further
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth == 2:
            for f in files[:3]:
                print(f"{indent}  {f}")
            if len(files) > 3:
                print(f"{indent}  ... ({len(files)} files total)")

if __name__ == "__main__":
    main()
