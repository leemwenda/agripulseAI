#!/usr/bin/env python3
"""
Score a checkpoint on any folder laid out as <folder>/<class_name>/*.jpg.

Use it on data/processed/test for the dataset number and on
data/realworld_eval for the honest farm-photo number, so the two can be
compared for every retrain. Applies the same "uncertain" rule as the app
(top-1 minus top-2 probability below the checkpoint's uncertain_margin).

Usage (from disease-detection/):
    python3 eval_folder.py --checkpoint models/cattle_health_classifier.pt --data_dir data/realworld_eval
    python3 eval_folder.py --checkpoint models/cattle_health_v4.pt --data_dir data/realworld_eval --errors 20
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--errors", type=int, default=10, help="how many wrong/uncertain files to list")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    margin = ckpt.get("uncertain_margin", 0.15)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((ckpt["image_size"], ckpt["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    root = Path(args.data_dir)
    actual_names = [d.name for d in sorted(root.iterdir()) if d.is_dir() and d.name in classes]
    skipped = [d.name for d in sorted(root.iterdir()) if d.is_dir() and d.name not in classes]
    if skipped:
        print(f"Ignoring folders that are not model classes: {skipped}")
    cols = classes + ["uncertain"]
    table = {a: {c: 0 for c in cols} for a in actual_names}
    problems = []

    for a in actual_names:
        for p in sorted((root / a).iterdir()):
            if p.suffix.lower() not in EXTS:
                continue
            try:
                x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            except Exception as exc:
                print(f"  skipped {p}: {exc}")
                continue
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1).squeeze(0).cpu()
            top2 = torch.topk(probs, min(2, len(classes)))
            top1, second = top2.values[0].item(), (top2.values[1].item() if len(classes) > 1 else 0.0)
            pred = "uncertain" if (top1 - second) < margin else classes[top2.indices[0].item()]
            table[a][pred] += 1
            if pred != a:
                problems.append((a, pred, top1, p))

    print(f"\nCheckpoint: {args.checkpoint}  ({ckpt.get('model_version')})")
    print(f"Data: {root}")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print("  " + " " * 24 + "".join(f"{c[:14]:>16}" for c in cols))
    n_total = n_correct = n_uncertain = 0
    for a in actual_names:
        row = table[a]
        print(f"  {a:<24}" + "".join(f"{row[c]:>16d}" for c in cols))
        n_total += sum(row.values())
        n_correct += row[a]
        n_uncertain += row["uncertain"]

    if n_total == 0:
        print("\nNo images found. Put photos in <data_dir>/<class_name>/ first.")
        return
    print(f"\nImages: {n_total}   correct: {n_correct} ({n_correct / n_total:.1%})   "
          f"uncertain: {n_uncertain} ({n_uncertain / n_total:.1%})")
    for a in actual_names:
        s = sum(table[a].values())
        if s:
            wrong = s - table[a][a] - table[a]["uncertain"]
            print(f"  {a}: recall {table[a][a] / s:.1%}  |  wrong label {wrong}  |  uncertain {table[a]['uncertain']}")

    if problems and args.errors:
        print(f"\nFirst {min(args.errors, len(problems))} non-correct files (actual -> predicted, top prob):")
        for a, pred, top1, p in problems[:args.errors]:
            print(f"  {a} -> {pred}  ({top1:.2f})  {p}")


if __name__ == "__main__":
    main()
