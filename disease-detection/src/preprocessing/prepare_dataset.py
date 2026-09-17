#!/usr/bin/env python3
"""
prepare_dataset.py - Phase 3: organize raw Kaggle LSD images into a clean
train/val/test split.

Usage:
    python3 prepare_dataset.py --raw_dir data/raw/lsd_v1 --out_dir data/processed

This auto-discovers class folders under raw_dir (whatever they're actually
named) and maps them to two canonical classes - "healthy" and
"lumpy_skin_disease" - via keyword matching, since Kaggle datasets use
inconsistent folder naming (e.g. "Normal", "Lumpy Cows", "Cows_Healthy").
If a folder doesn't match either keyword set, it's skipped and reported
rather than silently dropped or wrongly guessed.

KNOWN LIMITATION: this is a random split, not a leakage-safe one. The raw
Kaggle dataset provides no per-animal identifiers, so we can't guarantee the
same individual cow never appears in both train and test - this is a known
gap flagged in the project's Phase 5 (data structure) and Phase 23
(real-world testing) requirements. Treat test-set metrics as optimistic
until validated on genuinely new, unseen photos.
"""
import argparse
import os
import random
import shutil
from pathlib import Path

HEALTHY_KEYWORDS = ["healthy", "normal", "heal"]
LSD_KEYWORDS = ["lumpy", "lsd", "infected", "disease"]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def classify_folder_name(name):
    lower = name.lower()
    if any(k in lower for k in LSD_KEYWORDS):
        return "lumpy_skin_disease"
    if any(k in lower for k in HEALTHY_KEYWORDS):
        return "healthy"
    return None


def find_class_images(raw_dir):
    """Walks raw_dir, mapping every leaf folder containing images to a
    canonical class name. Returns (class_images, unmatched_folder_names)."""
    raw_dir = Path(raw_dir)
    class_images = {"healthy": [], "lumpy_skin_disease": []}
    unmatched_folders = set()

    for root, _dirs, files in os.walk(raw_dir):
        images = [f for f in files if Path(f).suffix.lower() in IMG_EXTENSIONS]
        if not images:
            continue
        folder_name = Path(root).name
        cls = classify_folder_name(folder_name)
        if cls is None:
            unmatched_folders.add(folder_name)
            continue
        for f in images:
            class_images[cls].append(Path(root) / f)

    return class_images, unmatched_folders


def split_and_copy(class_images, out_dir, train_frac=0.7, val_frac=0.15, seed=42):
    random.seed(seed)
    out_dir = Path(out_dir)
    stats = {}

    for cls, paths in class_images.items():
        paths = list(paths)
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        splits = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:],
        }

        stats[cls] = {}
        for split_name, split_paths in splits.items():
            split_dir = out_dir / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(split_paths):
                dst = split_dir / f"{cls}_{i:04d}{src.suffix.lower()}"
                shutil.copy2(src, dst)
            stats[cls][split_name] = len(split_paths)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    class_images, unmatched = find_class_images(args.raw_dir)

    print("Discovered images per class:")
    for cls, paths in class_images.items():
        print(f"  {cls}: {len(paths)}")

    if unmatched:
        print("\nWARNING: found image folders that didn't match a known class "
              "and were SKIPPED:")
        for name in sorted(unmatched):
            print(f"  {name}")
        print("If any of these should count as healthy/lumpy, rename the "
              "folder or edit HEALTHY_KEYWORDS/LSD_KEYWORDS above and re-run.")

    if not class_images["healthy"] or not class_images["lumpy_skin_disease"]:
        print("\nERROR: at least one class has zero images. Stopping - fix "
              "the folder mapping before proceeding.")
        return

    stats = split_and_copy(class_images, args.out_dir, args.train_frac, args.val_frac)

    print("\nFinal split:")
    for cls, splits in stats.items():
        print(f"  {cls}: {splits}")


if __name__ == "__main__":
    main()
