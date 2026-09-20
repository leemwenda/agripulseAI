#!/usr/bin/env python3
"""
Check for train/val/test leakage in data/processed.

Uses a perceptual difference hash (dHash) so re-saved, resized or lightly
cropped copies of the same photo are caught, not just byte-identical files.
It will NOT catch two different photos of the same animal - only near-copies -
so a clean result here is necessary but not sufficient.

Usage (from disease-detection/):
    python3 check_leakage.py
    python3 check_leakage.py --data_dir data/processed --max_dist 6
    python3 check_leakage.py --extra data/realworld_eval   # also compare vs train
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


def collect(folder):
    """Returns list of (hash, path) for every image under folder."""
    out = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in EXTS:
            try:
                out.append((dhash(p), p))
            except Exception as exc:
                print(f"  skipped {p}: {exc}")
    return out


def compare(name, items, train_items, max_dist, show):
    hits = []
    for h, p in items:
        best_d, best_p = 999, None
        for th, tp in train_items:
            d = hamming(h, th)
            if d < best_d:
                best_d, best_p = d, tp
        if best_d <= max_dist:
            hits.append((best_d, p, best_p))
    pct = 100 * len(hits) / max(len(items), 1)
    print(f"{name}: {len(hits)} of {len(items)} images ({pct:.1f}%) have a near-duplicate in train")
    for d, p, tp in sorted(hits, key=lambda x: x[0])[:show]:
        print(f"    dist={d:2d}  {p}  ~  {tp}")
    return len(hits), len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--max_dist", type=int, default=5,
                    help="max Hamming distance (of 64 bits) counted as near-duplicate")
    ap.add_argument("--show", type=int, default=8, help="example pairs to print per split")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra folders to compare against train (e.g. data/realworld_eval)")
    args = ap.parse_args()

    root = Path(args.data_dir)
    print(f"Hashing images under {root} ...")
    train = collect(root / "train")
    print(f"  train: {len(train)} images")
    if not train:
        raise SystemExit(f"No images found under {root / 'train'} - check --data_dir. Nothing was compared.")

    total_hits = 0
    for split in ("val", "test"):
        items = collect(root / split)
        h, _ = compare(split, items, train, args.max_dist, args.show)
        total_hits += h

    for extra in args.extra:
        items = collect(extra)
        compare(str(extra), items, train, args.max_dist, args.show)

    # duplicates inside train are harmless for leakage but inflate class size
    seen, dups = {}, 0
    for h, p in train:
        if h in seen:
            dups += 1
        else:
            seen[h] = p
    print(f"\nExact-hash duplicates inside train: {dups}")

    print("\nVerdict:")
    if total_hits == 0:
        print("  No near-duplicates found between train and val/test.")
        print("  (Different photos of the same animal can still leak - this only catches copies.)")
    else:
        print(f"  {total_hits} val/test images have near-copies in train, so the reported")
        print("  accuracy is inflated by at least that much. Fix the split before trusting it.")


if __name__ == "__main__":
    main()
