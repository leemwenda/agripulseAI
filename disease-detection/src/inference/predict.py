#!/usr/bin/env python3
"""
Run inference on a single image.

Works for any number of classes because it reads the class list from the
checkpoint. Returns "uncertain" whenever the top two class probabilities are
within `uncertain_margin` of each other (also stored in the checkpoint) —
i.e. the model isn't confident enough to distinguish, say, an early-stage
disease from healthy.

Usage:
    python3 predict.py --checkpoint models/cattle_health_classifier.pt \\
                        --image path/to/photo.jpg
"""
import argparse
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def predict(checkpoint_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    image_size = ckpt["image_size"]
    margin = ckpt.get("uncertain_margin", 0.15)

    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    prob_dict = {cls: round(p.item(), 4) for cls, p in zip(classes, probs)}
    sorted_probs = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_prob = sorted_probs[0]
    second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0

    result = {
        "species": "cattle",
        "model_version": ckpt["model_version"],
        "all_probabilities": prob_dict,
    }

    if (top_prob - second_prob) < margin:
        result["status"] = "uncertain"
        result["message"] = "Unable to confidently identify a condition from this image."
    else:
        result["status"] = top_label
        result["disease"] = top_label
        result["confidence"] = top_prob

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image), indent=2))


if __name__ == "__main__":
    main()
