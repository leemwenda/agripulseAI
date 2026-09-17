#!/usr/bin/env python3
"""
evaluate.py - Phase 4/5: evaluate a trained checkpoint on the held-out test
set. Reports accuracy, precision, recall, F1, and a confusion matrix - not
just raw accuracy, per the project's evaluation requirements. Recall on the
disease class matters most here: missing a sick animal is worse than a
false alarm on a healthy one.

Usage:
    python3 evaluate.py --checkpoint models/lsd_classifier_v1.pt --data_dir data/processed
"""
import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]

    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model version: {checkpoint.get('model_version', 'unknown')}")
    print(f"Trained at: {checkpoint.get('trained_at', 'unknown')}")
    print(f"Classes: {classes}\n")

    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = ImageFolder(f"{args.data_dir}/test", transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print(f"Test set size: {len(test_ds)}\n")
    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"          {'  '.join(classes)}")
    for i, row in enumerate(cm):
        print(f"{classes[i]:>10}  {row}")


if __name__ == "__main__":
    main()
