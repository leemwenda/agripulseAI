#!/usr/bin/env python3
"""
Photo analysis for the triage app.

Step 1: a COCO object detector checks that a cow is actually in the frame (the
        classifier has no concept of "not a cow" and would label a wall or a dog).
        Two detectors:  fast=False -> Faster R-CNN (accurate, handles fences and odd
        framing; used for single photos)   fast=True -> SSDLite (light, used for the
        live camera so it keeps up on CPU).
Step 2: the cattle health classifier (models/cattle_health_classifier.pt) scores
        the photo.

Classifier loading and preprocessing are the same as predict.py, so any checkpoint
the disease app loads works here too. Models load lazily on first use, so the app
starts fast and the symptom checklist works even if torch is missing.

analyze(pil_image, fast=False) returns a dict:
    {"status": "no_cow"}
    {"status": "ok", "probs": {...}, "classes": [...], "margin": 0.15,
     "top": "healthy", "top_prob": 0.91, "uncertain": False, "model_version": "..."}
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

_cache = {}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _classifier():
    if "clf" not in _cache:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"No classifier checkpoint at {CHECKPOINT_PATH}. Train one first.")
        dev = _device()
        ckpt = torch.load(CHECKPOINT_PATH, map_location=dev, weights_only=False)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev).eval()
        tf = transforms.Compose([
            transforms.Resize((ckpt["image_size"], ckpt["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _cache["clf"] = (model, ckpt["classes"], ckpt.get("uncertain_margin", 0.15),
                         tf, ckpt.get("model_version", "unknown"), dev)
    return _cache["clf"]


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


def has_cow(pil_img, fast=False):
    det, dev = _cow_detector(fast)
    tensor = transforms.functional.to_tensor(pil_img).to(dev)
    with torch.no_grad():
        pred = det([tensor])[0]
    return any(
        label.item() == COCO_COW_CLASS_INDEX and score.item() >= COW_SCORE_THRESHOLD
        for label, score in zip(pred["labels"], pred["scores"])
    )


def analyze(pil_img, fast=False):
    img = pil_img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    if not has_cow(img, fast=fast):
        return {"status": "no_cow"}
    model, classes, margin, tf, version, dev = _classifier()
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0).to(dev)), dim=1).squeeze(0).cpu()
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}
    ranked = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    top, top_prob = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "status": "ok",
        "probs": prob_dict,
        "classes": classes,
        "margin": margin,
        "top": top,
        "top_prob": top_prob,
        "uncertain": (top_prob - second) < margin,
        "model_version": version,
    }
