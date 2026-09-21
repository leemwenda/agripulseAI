#!/usr/bin/env python3
"""
Diagnostic: run the exact detection+crop the app uses on one image, save
the crop and a copy of the original with the box drawn on it, so you can
SEE what the classifier actually judged instead of guessing.

Usage (from disease-detection/):
    python3 diagnose_image.py --image data/realworld_eval/healthy/farm_01.jpg
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent / "triage"))
import vision  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    img = vision.prepare(Image.open(args.image))
    box = vision.cow_box(img, fast=False)

    if box is None:
        print("No cow detected at all - this confirms a genuine detection miss, "
              "not a crop/background issue.")
        return

    print(f"Detected cow box: {box}")

    boxed = img.copy()
    ImageDraw.Draw(boxed).rectangle(box, outline="red", width=4)
    boxed_path = Path(args.image).with_name(Path(args.image).stem + "_boxed.jpg")
    boxed.save(boxed_path)
    print(f"Saved original with detection box drawn on it: {boxed_path}")

    crop = vision.pad_crop(img, box)
    if crop is None:
        print("Crop was rejected as too small (MIN_CROP_SIDE) - falls back to the full image.")
        crop = img
    crop_path = Path(args.image).with_name(Path(args.image).stem + "_crop.jpg")
    crop.save(crop_path)
    print(f"Saved the exact crop the classifier judged: {crop_path}")

    result = vision.analyze(Image.open(args.image), checkpoint=args.checkpoint)
    print(f"\nFull analysis: {result}")


if __name__ == "__main__":
    main()
