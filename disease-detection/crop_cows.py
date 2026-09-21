#!/usr/bin/env python3
"""
Crop every image in a dataset to the cow, once, and cache the result.

Why: a small classifier trained on whole photos learns the easiest shortcut it can find - barn
versus pasture, one camera's colours, a fence - instead of the animal's skin. On the Kaggle data
that gives a great test score and false alarms on your own farm photos. Cropping to the cow removes
most of that shortcut. The app crops the same way at prediction time (triage/vision.py), so what
the model trains on is what it sees in use.

The folder structure is copied, so sub-folders (e.g. one folder per animal) survive.
Images with no cow found, or a cow too small to judge, are skipped and listed in _skipped.txt.
Re-running skips files that already exist, so an interrupted run can be resumed.

Usage (from disease-detection/):
    python3 crop_cows.py --src data/raw/cattle_health_v1 --dst data/cropped/cattle_health_v1
    python3 crop_cows.py --src data/farm                 --dst data/cropped/farm
Faster R-CNN on CPU takes roughly 1-2 s per image, so a few thousand images takes a while. Use the
accurate detector (the default) so training crops match the ones the app makes for single photos.
"""
import argparse
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "triage"))
import vision  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--fast", action="store_true", help="use the light SSDLite detector (less accurate)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images (for a quick test)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"--src {src} is not a folder")
    files = [p for p in sorted(src.rglob("*")) if p.is_file() and p.suffix.lower() in EXTS]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No images found under {src}")
    dst.mkdir(parents=True, exist_ok=True)

    done, skipped, cached = 0, [], 0
    used = set()
    start = time.time()
    for i, p in enumerate(files, 1):
        rel = p.relative_to(src).with_suffix(".jpg")
        out = dst / rel
        n = 1
        while out in used:                      # a.png and a.jpg in the same folder
            out = dst / rel.with_name(f"{rel.stem}_{n}.jpg")
            n += 1
        used.add(out)
        if out.exists():
            cached += 1
            continue
        try:
            img = vision.prepare(Image.open(p))
        except Exception as exc:
            skipped.append(f"unreadable\t{p}\t{exc}")
            continue
        box = vision.cow_box(img, fast=args.fast)
        if box is None:
            skipped.append(f"no_cow\t{p}")
            continue
        crop = vision.pad_crop(img, box)
        if crop is None:
            skipped.append(f"too_small\t{p}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out, quality=92)
        done += 1
        if i % 25 == 0:
            rate = (time.time() - start) / max(done, 1)
            print(f"  {i}/{len(files)}  cropped {done}  skipped {len(skipped)}  ({rate:.1f}s per image)")

    (dst / "_skipped.txt").write_text("\n".join(skipped) + ("\n" if skipped else ""))
    print(f"\nCropped {done} new, {cached} already done, skipped {len(skipped)} of {len(files)} images.")
    if skipped:
        print(f"Skipped files (no cow found / too small / unreadable) are listed in {dst / '_skipped.txt'}")
        print("If many real cows are skipped, look at a few by eye: they may be lying down, far away or "
              "cut off, and are better replaced with clearer photos.")


if __name__ == "__main__":
    main()
