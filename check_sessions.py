#!/usr/bin/env python3
"""
Diagnostic: checks whether cows share capture sessions (date/camera folders).
If cow_002, cow_004, cow_006 share sessions that cow_001/003/005 don't,
that explains the false duplicate flags - the model may be picking up on
background/lighting/camera artifacts from the shared session, not the cow.
"""
import os

DATA_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_subset")

def main():
    for cow in sorted(os.listdir(DATA_DIR)):
        cow_path = os.path.join(DATA_DIR, cow)
        if not os.path.isdir(cow_path):
            continue
        # top-level subfolders under each cow = capture sessions (e.g. "2023Aug16")
        sessions = [d for d in os.listdir(cow_path) if os.path.isdir(os.path.join(cow_path, d))]
        print(f"{cow}: sessions = {sorted(sessions)}")

if __name__ == "__main__":
    main()
