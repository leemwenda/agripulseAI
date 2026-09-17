#!/usr/bin/env python3
"""
predict.py - Phase 6: single-image inference for the LSD classifier.

Treats the output as a screening result, not a diagnosis - per the
project's Phase 24 requirement, a low-confidence prediction is reported as
"uncertain" rather than being forced into a class.

Usage:
    python3 predict.py --checkpoint models/lsd_classifier_v1.pt --image path/to/photo.jpg
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

CONFIDENCE_THRESHOLD = 0.6  # below this, report "unable to confidently classify"


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def predict(checkpoint_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]

    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_prob, top_idx = probs.max(dim=0)
    predicted_class = classes[top_idx.item()]
    confidence = top_prob.item()

    result = {
        "species": "cattle",
        "model_version": checkpoint.get("model_version", "unknown"),
        "all_probabilities": {cls: round(probs[i].item(), 4) for i, cls in enumerate(classes)},
    }

    if confidence < CONFIDENCE_THRESHOLD:
        result["status"] = "uncertain"
        result["message"] = "Unable to confidently identify a condition from this image."
    else:
        result["status"] = "healthy" if predicted_class == "healthy" else "suspected"
        result["disease"] = predicted_class
        result["confidence"] = round(confidence, 4)
        if predicted_class != "healthy":
            result["recommendation"] = "Veterinary examination recommended to confirm."

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    result = predict(args.checkpoint, args.image)
    print(result)


if __name__ == "__main__":
    main()
