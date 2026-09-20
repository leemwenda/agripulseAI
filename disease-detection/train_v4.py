#!/usr/bin/env python3
"""
Regularised retrain of the cattle health classifier.

Why this exists: in the v3 run train loss fell to 0.07 while val loss rose from
0.44 (epoch 1) to 0.64, i.e. the model memorised the training set, and early
stopping was watching val_acc so it kept a checkpoint with a worse val loss.

Changes vs the baseline:
  * strong augmentation (random crops, flips, rotation, colour jitter)
  * stem + layer1 frozen; lower LR on the backbone than on the classifier head
  * AdamW weight decay, label smoothing, class-weighted loss, cosine LR schedule
  * early stopping and "best" checkpoint chosen by VAL LOSS, not val accuracy

The checkpoint keeps the exact format predict.py / gradio_disease_app.py load
(plain ResNet18 with a single Linear head; keys model_state_dict, classes,
image_size, model_version, uncertain_margin, trained_at, val_acc). It is saved
to a NEW file, so the current production checkpoint is not overwritten.

Usage (from disease-detection/):
    python3 train_v4.py
    python3 train_v4.py --out models/cattle_health_v4.pt --version cattle-health-v4
"""
import argparse
import copy
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def confusion(model, loader, device, num_classes):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                cm[t, p] += 1
    return cm


def print_report(cm, classes):
    print("\nPer-class results (test set):")
    print(f"  {'class':<24}{'precision':>10}{'recall':>9}{'support':>9}")
    for i, c in enumerate(classes):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        print(f"  {c:<24}{prec:>10.2f}{rec:>9.2f}{cm[i, :].sum():>9d}")
    print(f"  accuracy: {np.trace(cm) / cm.sum():.3f}")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print("  " + " " * 24 + "".join(f"{c[:14]:>16}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:<24}" + "".join(f"{cm[i, j]:>16d}" for j in range(len(classes))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="models/cattle_health_v4.pt")
    ap.add_argument("--version", default="cattle-health-v4")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    size = cfg["image_size"]
    bs = cfg["batch_size"]
    epochs = args.epochs or cfg["epochs"]
    patience = args.patience or cfg["patience"]
    lr = cfg["learning_rate"]
    margin = cfg.get("uncertain_margin", 0.15)
    seed = cfg["data"].get("seed", 42)
    root = Path(cfg["data"]["processed_dir"])

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        NORM,
    ])
    # identical to the preprocessing predict.py / the Gradio app use at inference
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        NORM,
    ])

    train_ds = datasets.ImageFolder(root / "train", train_tf)
    val_ds = datasets.ImageFolder(root / "val", eval_tf)
    test_ds = datasets.ImageFolder(root / "test", eval_tf)
    classes = train_ds.classes
    assert val_ds.classes == classes and test_ds.classes == classes, "class folders differ between splits"
    print(f"Classes: {classes}")
    print(f"Train {len(train_ds)}  Val {len(val_ds)}  Test {len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2)

    # class-weighted training loss (val loss stays plain so epochs are comparable)
    counts = np.bincount(train_ds.targets, minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).to(device)
    print(f"Class counts {counts.tolist()} -> loss weights {[round(w, 2) for w in weights.tolist()]}")
    train_loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss()

    # plain ResNet18 + single Linear head: must match build_model() in predict.py
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    for name, p in model.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1")):
            p.requires_grad = False
    model = model.to(device)

    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")]
    head = list(model.fc.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": lr * 0.3}, {"params": head, "lr": lr * 3}],
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, best_val_acc, best_state, bad_epochs = float("inf"), 0.0, None, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, train_loss_fn, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, eval_loss_fn, device)
        scheduler.step()
        flag = ""
        if va_loss < best_val_loss:
            best_val_loss, best_val_acc = va_loss, va_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            flag = "  -> new best (val_loss)"
        else:
            bad_epochs += 1
        print(f"Epoch {epoch}/{epochs} - train_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val_loss {va_loss:.4f} acc {va_acc:.3f}{flag}")
        if bad_epochs >= patience:
            print(f"Early stopping: val loss has not improved for {patience} epochs.")
            break

    model.load_state_dict(best_state)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in best_state.items()},
        "classes": classes,
        "image_size": size,
        "model_version": args.version,
        "uncertain_margin": margin,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "val_acc": best_val_acc,
        "val_loss": best_val_loss,
    }, out)
    print(f"\nBest val_loss {best_val_loss:.4f} (val_acc {best_val_acc:.3f}). Saved to {out}")

    print_report(confusion(model, test_dl, device, len(classes)), classes)


if __name__ == "__main__":
    main()
