#!/usr/bin/env python3
"""
AgriPulse bulk cow registration.

Automatically registers all cows found in the dataset folder, using multiple
photos per cow (different cameras/angles/days, wherever the folder structure
provides them) - no manual upload through the Gradio UI needed.

Run: python3 bulk_register.py
Then gradio_app.py will already have these cows in its database.
"""
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

DATA_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_subset")
CHECKPOINT_PATH = "cow_embedding_resnet50_v1.pt"
DATABASE_PATH = "cow_database.json"
PHOTOS_PER_COW = 5  # how many reference photos to register per cow


class CowEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def collect_images_by_cow(root):
    """Recursively finds all images under each cow folder, regardless of nesting
    (handles structures like cow_001/2023Aug16/001/*.jpg)."""
    cow_images = {}
    for cow_folder in sorted(os.listdir(root)):
        cow_path = os.path.join(root, cow_folder)
        if not os.path.isdir(cow_path):
            continue
        images = []
        for dirpath, _, filenames in os.walk(cow_path):
            for f in filenames:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(dirpath, f))
        if images:
            cow_images[cow_folder] = sorted(images)
    return cow_images


def get_embedding(model, image_path, device):
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0).cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return

    model = CowEmbeddingNet().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cow_images = collect_images_by_cow(DATA_DIR)
    print(f"\nFound {len(cow_images)} cows in {DATA_DIR}")

    random.seed(42)
    database = {}

    for cow_id, images in cow_images.items():
        # Spread photo selection across the full set (different days/cameras)
        # rather than just taking the first N, for more angle variety.
        if len(images) > PHOTOS_PER_COW:
            step = len(images) // PHOTOS_PER_COW
            selected = images[::step][:PHOTOS_PER_COW]
        else:
            selected = images

        embeddings = [get_embedding(model, img, device) for img in selected]
        database[cow_id] = embeddings
        print(f"  {cow_id}: registered {len(embeddings)} photos (from {len(images)} available)")

    # Save in the same format gradio_app.py / agripulse_inference.py expect
    raw = {cow_id: [e.tolist() for e in embs] for cow_id, embs in database.items()}
    with open(DATABASE_PATH, "w") as f:
        json.dump(raw, f)

    print(f"\nSaved database: {DATABASE_PATH}")
    print(f"Total cows registered: {len(database)}")
    print("\nYou can now open gradio_app.py's 'Identify Cow' tab directly - "
          "no manual registration needed.")


if __name__ == "__main__":
    main()
