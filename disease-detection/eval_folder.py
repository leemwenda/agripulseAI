#!/usr/bin/env python3
"""
Score a checkpoint on any folder laid out as <folder>/<class_name>/*.jpg, using EXACTLY the pipeline
the app uses: cow detection -> crop (if the checkpoint was trained on crops) -> calibrated
probabilities -> the app's decision rule. Use it on a folder of real farm photos that were NEVER
used for training (data/realworld_eval) for the honest number, and compare checkpoints with it.

Columns: one per model class, plus
    uncertain   the app would say "uncertain"
    no_cow      no cow found (the app would ask for another photo)

Usage (from disease-detection/):
    python3 eval_folder.py --checkpoint models/cattle_health_v5.pt --data_dir data/realworld_eval
    python3 eval_folder.py --checkpoint models/cattle_health_classifier.pt --data_dir data/realworld_eval
    python3 eval_folder.py --checkpoint models/cattle_health_v5.pt --data_dir data/processed/test --no_detector
        (--no_detector: the images are already cow crops, so skip detection)
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "triage"))
import vision  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--errors", type=int, default=10, help="how many wrong/uncertain files to list")
    ap.add_argument("--floor", type=float, default=vision.DISEASE_FLOOR)
    ap.add_argument("--fast", action="store_true", help="use the light detector, like the live camera")
    ap.add_argument("--no_detector", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    first = vision._classifier(args.checkpoint)
    classes = first["classes"]
    actual_names = [d.name for d in sorted(root.iterdir()) if d.is_dir() and d.name in classes]
    skipped = [d.name for d in sorted(root.iterdir()) if d.is_dir() and d.name not in classes]
    if skipped:
        print(f"Ignoring folders that are not model classes: {skipped}")
    cols = classes + ["uncertain", "no_cow"]
    table = {a: {c: 0 for c in cols} for a in actual_names}
    problems = []

    for a in actual_names:
        for p in sorted((root / a).rglob("*")):
            if p.suffix.lower() not in EXTS:
                continue
            try:
                res = vision.analyze(Image.open(p), fast=args.fast, checkpoint=args.checkpoint,
                                     detect=not args.no_detector)
            except Exception as exc:
                print(f"  skipped {p}: {exc}")
                continue
            if res["status"] == "no_cow":
                pred, top1 = "no_cow", 0.0
            else:
                kind, top = vision.decide(res["probs"], res["margin"], res["thresholds"], args.floor)
                pred, top1 = ("uncertain" if kind == "uncertain" else top), res["top_prob"]
            table[a][pred] += 1
            if pred != a:
                problems.append((a, pred, top1, p))

    print(f"\nCheckpoint: {args.checkpoint}  ({first['version']})   temperature {first['temperature']:.2f}   "
          f"thresholds {first['thresholds'] or 'none'}   trained on cow crops: {first['cropped']}")
    print(f"Data: {root}")
    print("\nConfusion matrix (rows = actual, cols = what the app would say):")
    print("  " + " " * 24 + "".join(f"{c[:14]:>16}" for c in cols))
    n_total = n_correct = 0
    for a in actual_names:
        row = table[a]
        print(f"  {a:<24}" + "".join(f"{row[c]:>16d}" for c in cols))
        n_total += sum(row.values())
        n_correct += row[a]
    if n_total == 0:
        print("\nNo images found. Put photos in <data_dir>/<class_name>/ first.")
        return
    print(f"\nImages: {n_total}   correct: {n_correct} ({n_correct / n_total:.1%})")
    for a in actual_names:
        s = sum(table[a].values())
        if not s:
            continue
        wrong = s - table[a][a] - table[a]["uncertain"] - table[a]["no_cow"]
        if a == "healthy":
            print(f"  healthy: FALSE ALARMS {wrong} of {s} ({wrong / s:.1%})  |  uncertain {table[a]['uncertain']}  |  no cow {table[a]['no_cow']}")
        else:
            print(f"  {a}: found {table[a][a]} of {s} ({table[a][a] / s:.1%})  |  wrong label {wrong}  |  "
                  f"uncertain {table[a]['uncertain']}  |  no cow {table[a]['no_cow']}")

    if problems and args.errors:
        print(f"\nFirst {min(args.errors, len(problems))} non-correct files (actual -> app says, top prob):")
        for a, pred, top1, p in problems[:args.errors]:
            print(f"  {a} -> {pred}  ({top1:.2f})  {p}")


if __name__ == "__main__":
    main()
