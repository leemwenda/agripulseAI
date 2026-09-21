#!/usr/bin/env python3
"""
Retrain the cattle health classifier to REDUCE FALSE ALARMS.

Builds on train_v4.py (frozen stem, strong augmentation, weight decay, label smoothing, class weights,
best checkpoint chosen by validation loss) and adds what false alarms need:

  1. FARM PHOTOS COUNT MORE. Your own real photos (files starting "farm_", made by prepare_dataset.py)
     are sampled --farm_weight times more often, because they look like what the app will actually see.
  2. CALIBRATION. A temperature is fitted on the validation set so a stated 0.95 means about 95%, instead
     of the raw softmax's habit of saying 0.999 about photos it has never seen.
  3. PER-DISEASE THRESHOLDS. For every disease class, the lowest calibrated probability is chosen at which
     at most (1 - --target_spec) of the validation images that are NOT that disease would be called that
     disease. The app names a disease only above max(0.80, this threshold).
  4. HONEST REPORT. The test report uses the same decision rule as the app (triage/vision.py decide()),
     lists false alarms and missed cases separately, and repeats the numbers for farm photos only.

The checkpoint keeps the format the apps load (plain ResNet18, single Linear head) plus new keys:
temperature, class_thresholds, cropped. It is saved to a NEW file; nothing is overwritten.

Usage (from disease-detection/):
    python3 train_v5.py
    python3 train_v5.py --out models/cattle_health_v5.pt --version cattle-health-v5 --target_spec 0.98
"""
import argparse
import copy
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent / "triage"))
import vision  # noqa: E402  (decide() is shared with the app so this report matches deployment)

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def farm_flags(ds):
    return np.array([Path(p).name.startswith("farm_") for p, _ in ds.samples])


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


def collect_logits(model, loader, device):
    model.eval()
    outs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            outs.append(model(x.to(device)).cpu())
            ys.append(y)
    return torch.cat(outs), torch.cat(ys).numpy()


def fit_temperature(logits, targets):
    """Single temperature T minimising validation NLL of softmax(logits / T)."""
    y = torch.as_tensor(targets)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.5, 5.0))


def choose_threshold(neg_probs, target_spec, lo=0.5, hi=0.99):
    """Lowest threshold letting at most (1 - target_spec) of the negatives through."""
    neg = np.sort(np.asarray(neg_probs))[::-1]
    allowed = int(np.floor((1.0 - target_spec) * len(neg)))
    t = float(neg[allowed]) + 1e-6 if allowed < len(neg) else lo
    return float(min(max(t, lo), hi))


def deployed_predictions(logits, classes, temperature, margin, thresholds, floor):
    probs = torch.softmax(logits / temperature, dim=1).numpy()
    preds = []
    for row in probs:
        kind, top = vision.decide({c: float(p) for c, p in zip(classes, row)}, margin, thresholds, floor)
        preds.append("uncertain" if kind == "uncertain" else top)
    return preds


def report(title, targets, preds, classes):
    cols = list(classes) + ["uncertain"]
    idx = {c: i for i, c in enumerate(cols)}
    cm = np.zeros((len(classes), len(cols)), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, idx[p]] += 1
    print(f"\n{title}  (rows = actual, cols = predicted by the app's rule)")
    print("  " + " " * 24 + "".join(f"{c[:14]:>16}" for c in cols))
    for i, c in enumerate(classes):
        print(f"  {c:<24}" + "".join(f"{cm[i, j]:>16d}" for j in range(len(cols))))
    if "healthy" in classes:
        h = classes.index("healthy")
        n_h = int(cm[h].sum())
        fa = int(cm[h].sum() - cm[h, h] - cm[h, idx["uncertain"]])
        print(f"  FALSE ALARMS: {fa} of {n_h} healthy images were called a disease"
              f"  ({fa / max(n_h, 1):.1%})")
    for i, c in enumerate(classes):
        if c == "healthy" or cm[i].sum() == 0:
            continue
        n = int(cm[i].sum())
        print(f"  {c}: found {int(cm[i, i])}/{n} ({cm[i, i] / n:.1%}), "
              f"called uncertain {int(cm[i, idx['uncertain']])}, "
              f"wrongly called healthy {int(cm[i, idx['healthy']]) if 'healthy' in idx else 0}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="models/cattle_health_v5.pt")
    ap.add_argument("--version", default="cattle-health-v5")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--farm_weight", type=float, default=4.0, help="how much more often farm photos are sampled")
    ap.add_argument("--target_spec", type=float, default=0.97,
                    help="share of validation images NOT of a disease that must stay below its threshold")
    ap.add_argument("--floor", type=float, default=vision.DISEASE_FLOOR, help="app-wide minimum to name a disease")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    size, bs = cfg["image_size"], cfg["batch_size"]
    epochs = args.epochs or cfg["epochs"]
    patience = args.patience or cfg["patience"]
    lr = cfg["learning_rate"]
    margin = cfg.get("uncertain_margin", 0.15)
    seed = cfg["data"].get("seed", 42)
    cropped = bool(cfg["data"].get("cropped", False))
    root = Path(cfg["data"]["processed_dir"])

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}   cow-cropped data: {cropped}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        NORM,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])
    eval_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), NORM])

    train_ds = datasets.ImageFolder(root / "train", train_tf)
    val_ds = datasets.ImageFolder(root / "val", eval_tf)
    test_ds = datasets.ImageFolder(root / "test", eval_tf)
    classes = train_ds.classes
    assert val_ds.classes == classes and test_ds.classes == classes, "class folders differ between splits"
    n_farm = {k: int(farm_flags(d).sum()) for k, d in (("train", train_ds), ("val", val_ds), ("test", test_ds))}
    print(f"Classes: {classes}")
    print(f"Train {len(train_ds)}  Val {len(val_ds)}  Test {len(test_ds)}   (farm photos: {n_farm})")
    if n_farm["train"] == 0:
        print("NOTE: no farm photos in train. Add your own photos under data/farm/<class>/ (see "
              "prepare_dataset.py) - they are the best cure for false alarms on your own cattle.")

    fw = np.where(farm_flags(train_ds), args.farm_weight, 1.0)
    if n_farm["train"] and args.farm_weight != 1.0:
        sampler = WeightedRandomSampler(torch.as_tensor(fw, dtype=torch.double), num_samples=len(train_ds), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=args.workers)
    else:
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=args.workers)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=args.workers)

    counts = np.maximum(np.bincount(train_ds.targets, minlength=len(classes)), 1)
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).to(device)
    print(f"Class counts {counts.tolist()} -> loss weights {[round(w, 2) for w in weights.tolist()]}")
    train_loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss()

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    for name, p in model.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1")):
            p.requires_grad = False
    model = model.to(device)

    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": lr * 0.3}, {"params": list(model.fc.parameters()), "lr": lr * 3}],
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, best_state, bad = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, train_loss_fn, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, eval_loss_fn, device)
        scheduler.step()
        flag = ""
        if va_loss < best_val_loss:
            best_val_loss, best_state, bad = va_loss, copy.deepcopy(model.state_dict()), 0
            flag = "  -> new best (val_loss)"
        else:
            bad += 1
        print(f"Epoch {epoch}/{epochs} - train_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val_loss {va_loss:.4f} acc {va_acc:.3f}{flag}")
        if bad >= patience:
            print(f"Early stopping: val loss has not improved for {patience} epochs.")
            break
    model.load_state_dict(best_state)

    # ---- calibration and thresholds, both fitted on the validation set only
    val_logits, val_y = collect_logits(model, val_dl, device)
    temperature = fit_temperature(val_logits, val_y)
    nll = nn.CrossEntropyLoss()
    print(f"\nCalibration: temperature {temperature:.2f}   "
          f"val NLL {nll(val_logits, torch.as_tensor(val_y)).item():.3f} -> "
          f"{nll(val_logits / temperature, torch.as_tensor(val_y)).item():.3f}")
    val_probs = torch.softmax(val_logits / temperature, dim=1).numpy()
    thresholds = {}
    for i, c in enumerate(classes):
        if c == "healthy":
            continue
        neg = val_probs[val_y != i, i]
        if len(neg) < 20:
            thresholds[c] = float(args.floor)
            print(f"  {c}: only {len(neg)} validation images of other classes - using the floor {args.floor:.2f}")
            continue
        thresholds[c] = choose_threshold(neg, args.target_spec)
        print(f"  {c}: threshold {thresholds[c]:.3f}  (>= {args.target_spec:.0%} of {len(neg)} other images stay below it)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in best_state.items()},
        "classes": classes,
        "image_size": size,
        "model_version": args.version,
        "uncertain_margin": margin,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "val_loss": best_val_loss,
        "temperature": temperature,
        "class_thresholds": thresholds,
        "cropped": cropped,
        "target_spec": args.target_spec,
    }, out)
    print(f"\nSaved to {out}")

    # ---- report on the untouched test split, using exactly the app's decision rule
    test_logits, test_y = collect_logits(model, test_dl, device)
    preds = deployed_predictions(test_logits, classes, temperature, margin, thresholds, args.floor)
    report("TEST SET, whole", test_y, preds, classes)
    farm = farm_flags(test_ds)
    if farm.any():
        report("TEST SET, farm photos only", test_y[farm], [p for p, f in zip(preds, farm) if f], classes)
    else:
        print("\n(no farm photos in the test split: the numbers above come from the download only and are "
              "optimistic. Run eval_folder.py on data/realworld_eval for the honest number.)")


if __name__ == "__main__":
    main()
