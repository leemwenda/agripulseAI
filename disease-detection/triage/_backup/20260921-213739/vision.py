#!/usr/bin/env python3
"""
Photo analysis for the triage app.

Step 1: a COCO object detector finds the cow (the classifier has no concept of
        "not a cow" and would label a wall or a dog).
        fast=False -> Faster R-CNN (accurate; used for single photos)
        fast=True  -> SSDLite (light; used for the live camera so it keeps up on CPU)
Step 2: if the checkpoint was trained on cow crops (checkpoint["cropped"] is True) the photo is
        cropped to the cow first. This removes the background (barn, pasture, fence), which is the
        easiest thing for a small model to "learn" instead of the animal.
Step 3: the cattle health classifier scores the cow. Scores are divided by the checkpoint's
        temperature (calibration) so 0.99 really means very likely, not "never saw this before".
Step 4: decide() names a disease only if the calibrated probability clears BOTH the app floor and
        the per-disease threshold stored in the checkpoint (chosen on validation data to keep false
        alarms rare). Old checkpoints without these fields behave exactly as before.

analyze(pil_image, fast=False, checkpoint=None, detect=True) returns a dict:
    {"status": "no_cow"}
    {"status": "ok", "probs": {...}, "classes": [...], "margin": 0.15, "thresholds": {...},
     "temperature": 1.3, "cropped": True, "top": "healthy", "top_prob": 0.91,
     "uncertain": False, "model_version": "..."}
"""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "cattle_health_classifier.pt"
COCO_COW_CLASS_INDEX = 21          # "cow" in torchvision's 91-class COCO list
COW_SCORE_THRESHOLD = 0.4
MAX_SIDE = 1024                    # downscale big phone photos: much faster on CPU
CROP_PAD = 0.08                    # extra margin around the cow box, as a fraction of its size
MIN_CROP_SIDE = 48                 # a cow smaller than this (pixels) is too small to judge
DISEASE_FLOOR = 0.80               # never name a disease below this calibrated probability

_cache = {}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _classifier(path=None):
    path = Path(path) if path else CHECKPOINT_PATH
    key = f"clf:{path.resolve()}"
    if key not in _cache:
        if not path.exists():
            raise FileNotFoundError(f"No classifier checkpoint at {path}. Train one first.")
        dev = _device()
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev).eval()
        tf = transforms.Compose([
            transforms.Resize((ckpt["image_size"], ckpt["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _cache[key] = {
            "model": model, "classes": ckpt["classes"], "tf": tf, "dev": dev,
            "margin": ckpt.get("uncertain_margin", 0.15),
            "version": ckpt.get("model_version", "unknown"),
            "temperature": float(ckpt.get("temperature", 1.0)),
            "thresholds": dict(ckpt.get("class_thresholds", {})),
            "cropped": bool(ckpt.get("cropped", False)),
        }
    return _cache[key]


def _cow_detector(fast):
    key = "det_fast" if fast else "det_acc"
    if key not in _cache:
        dev = _device()
        if fast:
            det = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        else:
            det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        _cache[key] = (det.to(dev).eval(), dev)
    return _cache[key]


def prepare(pil_img):
    """RGB copy, downscaled so the longest side is at most MAX_SIDE."""
    img = pil_img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def cow_box(pil_img, fast=False):
    """(x0, y0, x1, y1) of the largest cow found, or None."""
    det, dev = _cow_detector(fast)
    tensor = transforms.functional.to_tensor(pil_img).to(dev)
    with torch.no_grad():
        pred = det([tensor])[0]
    best = None
    for label, score, box in zip(pred["labels"], pred["scores"], pred["boxes"]):
        if label.item() == COCO_COW_CLASS_INDEX and score.item() >= COW_SCORE_THRESHOLD:
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            area = (x1 - x0) * (y1 - y0)
            if best is None or area > best[0]:
                best = (area, (x0, y0, x1, y1))
    return None if best is None else best[1]


def has_cow(pil_img, fast=False):
    return cow_box(pil_img, fast=fast) is not None


def pad_crop(img, box, pad=CROP_PAD):
    """Crop img to box plus a margin. Returns None if the result is too small to judge."""
    w, h = img.size
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    left, top = max(0, int(x0 - px)), max(0, int(y0 - py))
    right, bottom = min(w, int(x1 + px)), min(h, int(y1 + py))
    if right - left < MIN_CROP_SIDE or bottom - top < MIN_CROP_SIDE:
        return None
    return img.crop((left, top, right, bottom))


def decide(probs, margin, thresholds=None, floor=DISEASE_FLOOR):
    """-> (kind, top_label). kind is 'uncertain', 'healthy' or 'disease'.

    A disease is named only when the top class is clearly ahead of the runner-up (margin) AND its
    calibrated probability reaches max(floor, that disease's own threshold from the checkpoint).
    Shared by the app, the evaluation script and the training report so they can never disagree.
    """
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, top_p = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if (top_p - second) < margin:
        return "uncertain", top
    if top == "healthy":
        return "healthy", top
    need = max(floor, (thresholds or {}).get(top, 0.0))
    return ("disease", top) if top_p >= need else ("uncertain", top)


def analyze(pil_img, fast=False, checkpoint=None, detect=True):
    img = prepare(pil_img)
    box = None
    if detect:
        box = cow_box(img, fast=fast)
        if box is None:
            return {"status": "no_cow"}
    clf = _classifier(checkpoint)
    view = img
    if detect and clf["cropped"]:
        view = pad_crop(img, box) or img
    with torch.no_grad():
        logits = clf["model"](clf["tf"](view).unsqueeze(0).to(clf["dev"])).squeeze(0).cpu()
        probs = torch.softmax(logits / clf["temperature"], dim=0)
    prob_dict = {c: float(p) for c, p in zip(clf["classes"], probs)}
    ranked = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    top, top_prob = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "status": "ok",
        "probs": prob_dict,
        "classes": clf["classes"],
        "margin": clf["margin"],
        "thresholds": clf["thresholds"],
        "temperature": clf["temperature"],
        "cropped": clf["cropped"],
        "top": top,
        "top_prob": top_prob,
        "uncertain": (top_prob - second) < clf["margin"],
        "model_version": clf["version"],
    }
