#!/usr/bin/env python3
import os

DATA_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_subset")

def main():
    for cow in sorted(os.listdir(DATA_DIR)):
        cow_path = os.path.join(DATA_DIR, cow)
        if not os.path.isdir(cow_path):
            continue
        camera_folders = set()
        for dirpath, dirnames, filenames in os.walk(cow_path):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in filenames):
                # the immediate parent folder name of the images = camera/subfolder id
                camera_folders.add(os.path.basename(dirpath))
        print(f"{cow}: camera/subfolders = {sorted(camera_folders)}")

if __name__ == "__main__":
    main()
