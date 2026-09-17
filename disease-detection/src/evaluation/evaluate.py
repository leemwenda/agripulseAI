#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the held-out test set.

Reads the class list from the CHECKPOINT (saved by train_baseline.py), not
from config.yaml — so this works unchanged for a 2-class or 12-class model.

Usage:
    python3 evaluate.py --checkpoint models/cattle_health_classifier.pt \\
                         --data_dir data/processed
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    image_size = ckpt["image_size"]

    print(f"Model version: {ckpt['model_version']}")
    print(f"Trained at: {ckpt['trained_at']}")
    print(f"Classes: {classes}\n")

    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=eval_tf)
    # ImageFolder derives its own class order from the folder names present in
    # data/test — assert it matches the checkpoint so results aren't silently
    # mislabeled if someone evaluates against a differently-composed test set.
    if test_ds.classes != classes:
        raise SystemExit(
            f"Test set classes {test_ds.classes} don't match checkpoint "
            f"classes {classes}. Re-run prepare_dataset.py with the same "
            f"config used for training."
        )

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    print(f"Test set size: {len(test_ds)}\n")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(all_labels, all_preds)
    header = "".join(f"{c:>20}" for c in classes)
    print(f"{'':>20}{header}")
    for name, row in zip(classes, cm):
        print(f"{name:>20}{row}")


if __name__ == "__main__":
    main()
