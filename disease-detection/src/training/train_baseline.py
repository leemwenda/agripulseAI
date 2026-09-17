#!/usr/bin/env python3
"""
train_baseline.py - Phase 4: train a baseline Healthy vs Lumpy Skin Disease
classifier using transfer learning (ResNet18, ImageNet-pretrained, frozen
backbone - same CPU-friendly approach already used for the re-id model in
train_local_v2.py).

Usage:
    python3 train_baseline.py --config configs/config.yaml
"""
import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_transforms():
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, eval_tf = get_transforms()
    train_ds = ImageFolder(os.path.join(config["data_dir"], "train"), transform=train_tf)
    val_ds = ImageFolder(os.path.join(config["data_dir"], "val"), transform=eval_tf)

    print(f"Classes: {train_ds.classes}")
    print(f"Train images: {len(train_ds)}, Val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    model = build_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=config["learning_rate"])

    best_val_acc = 0.0
    best_epoch = -1
    patience = config.get("early_stopping_patience", 5)
    epochs_without_improvement = 0

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_dir"], "lsd_classifier_v1.pt")

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        print(f"Epoch {epoch + 1}/{config['epochs']} - "
              f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "epoch": epoch,
                "val_acc": val_acc,
                "model_version": "lsd-classifier-v1.0",
                "trained_at": datetime.now().isoformat(),
                "config": config,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping - no improvement for {patience} epochs.")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
