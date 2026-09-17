#!/usr/bin/env python3
"""
Organize raw class folders into train/val/test splits.

Generic over N classes: reads the class list (name + source folder) from
config.yaml instead of hardcoding "healthycows" / "lumpycows". Adding a new
disease to the model does NOT require touching this file — just add a new
entry under `classes:` in configs/config.yaml.

Usage:
    python3 prepare_dataset.py --config ../../configs/config.yaml
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def split_list(items, ratios, seed):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    ratios = cfg["data"]["split"]
    seed = cfg["data"].get("seed", 42)
    classes = cfg["classes"]

    if not classes:
        raise SystemExit("No classes defined in config.yaml under `classes:` — nothing to do.")

    print("Discovered images per class:")
    per_class_counts = {}
    per_class_images = {}
    for c in classes:
        name, subdir = c["name"], c["dir"]
        folder = raw_dir / subdir
        if not folder.is_dir():
            print(f"  WARNING: {folder} does not exist — skipping class '{name}'")
            continue
        imgs = list_images(folder)
        if not imgs:
            print(f"  WARNING: {folder} has no images — skipping class '{name}'")
            continue
        per_class_images[name] = imgs
        per_class_counts[name] = len(imgs)
        print(f"  {name}: {len(imgs)}")

    if len(per_class_images) < 2:
        raise SystemExit(
            "Need at least 2 classes with images to train a classifier. "
            "Check your raw_dir and class `dir` values in config.yaml."
        )

    # Clean and recreate processed_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        for name in per_class_images:
            (out_dir / split / name).mkdir(parents=True, exist_ok=True)

    print("\nFinal split:")
    for name, imgs in per_class_images.items():
        splits = split_list(imgs, ratios, seed)
        for split_name, split_imgs in splits.items():
            for img_path in split_imgs:
                shutil.copy2(img_path, out_dir / split_name / name / img_path.name)
        print(f"  {name}: {{'train': {len(splits['train'])}, "
              f"'val': {len(splits['val'])}, 'test': {len(splits['test'])}}}")

    print(f"\nClasses in this dataset: {list(per_class_images.keys())}")
    print(f"Processed dataset written to: {out_dir}")


if __name__ == "__main__":
    main()
