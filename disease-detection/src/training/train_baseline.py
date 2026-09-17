#!/usr/bin/env python3
"""
Train a baseline image classifier over N classes (read from config.yaml).

Key design choice: the trained checkpoint stores its own class list
(model_version, classes, image_size) alongside the weights. evaluate.py and
predict.py read the class list FROM THE CHECKPOINT, not from config.yaml —
so an old checkpoint always stays self-describing even if you later add more
diseases to config.yaml for the next training run.

Usage:
    python3 train_baseline.py --config ../../configs/config.yaml
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(num_classes):
    # ResNet18: lighter than ResNet50, trains faster on CPU. Swap freely —
    # nothing else in this file depends on the specific backbone.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def make_loaders(processed_dir, image_size, batch_size):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(Path(processed_dir) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(processed_dir) / "val", transform=eval_tf)

    # ImageFolder sorts class names alphabetically — this becomes the
    # canonical class order baked into the checkpoint.
    classes = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, classes


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = make_loaders(
        cfg["data"]["processed_dir"], cfg["image_size"], cfg["batch_size"]
    )
    print(f"Classes: {classes}")
    print(f"Train images: {len(train_loader.dataset)}, Val images: {len(val_loader.dataset)}")

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    checkpoint_path = models_dir / "cattle_health_classifier.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch}/{cfg['epochs']} - train_loss: {train_loss:.4f} "
              f"- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "image_size": cfg["image_size"],
                "model_version": cfg["model_version"],
                "uncertain_margin": cfg.get("uncertain_margin", 0.15),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "val_acc": val_acc,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["patience"]:
                print(f"Early stopping - no improvement for {cfg['patience']} epochs.")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
