#!/usr/bin/env python3
"""
AgriPulse bulk registration - combined dataset version.

Registers your original 6 MultiCamCows cows PLUS a sample of OpenCows2020
cows, so the Gradio UI has a richer, more realistic set to test identification
and duplicate detection against.

Run: python3 bulk_register_combined.py
Then: python3 gradio_app.py
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

MULTICAMCOWS_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_subset")
OPENCOWS_TEST_DIR = os.path.expanduser(
    "~/agripulse-ai/opencows2020/extracted/10m32xl88x2b61zlkkgz3fml17/identification/images/test"
)
CHECKPOINT_PATH = "cow_embedding_resnet50_v1.pt"
DATABASE_PATH = "cow_database.json"
PHOTOS_PER_COW = 5
NUM_OPENCOWS_TO_ADD = 15  # keep it manageable for browsing in the UI


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


def get_embedding(model, image_path, device):
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0).cpu()


def collect_images_by_cow(root):
    cow_images = {}
    if not os.path.exists(root):
        return cow_images
    for cow_folder in sorted(os.listdir(root)):
        cow_path = os.path.join(root, cow_folder)
        if not os.path.isdir(cow_path) or cow_folder.startswith("."):
            continue
        images = []
        for dirpath, _, filenames in os.walk(cow_path):
            for f in filenames:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(dirpath, f))
        if images:
            cow_images[cow_folder] = sorted(images)
    return cow_images


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

    random.seed(42)
    database = {}

    # --- MultiCamCows (all 6) ---
    mcc_images = collect_images_by_cow(MULTICAMCOWS_DIR)
    print(f"\nMultiCamCows: {len(mcc_images)} cows")
    for cow_id, images in mcc_images.items():
        if len(images) > PHOTOS_PER_COW:
            step = len(images) // PHOTOS_PER_COW
            selected = images[::step][:PHOTOS_PER_COW]
        else:
            selected = images
        embeddings = [get_embedding(model, img, device) for img in selected]
        full_id = f"mcc_{cow_id}"
        database[full_id] = embeddings
        print(f"  {full_id}: {len(embeddings)} photos")

    # --- OpenCows2020 (sample) ---
    oc_images = collect_images_by_cow(OPENCOWS_TEST_DIR)
    print(f"\nOpenCows2020 test split: {len(oc_images)} cows found")
    sample_ids = random.sample(list(oc_images.keys()), min(NUM_OPENCOWS_TO_ADD, len(oc_images)))
    for cow_id in sample_ids:
        images = oc_images[cow_id]
        if len(images) > PHOTOS_PER_COW:
            selected = random.sample(images, PHOTOS_PER_COW)
        else:
            selected = images
        embeddings = [get_embedding(model, img, device) for img in selected]
        full_id = f"oc_{cow_id}"
        database[full_id] = embeddings
        print(f"  {full_id}: {len(embeddings)} photos")

    raw = {cow_id: [e.tolist() for e in embs] for cow_id, embs in database.items()}
    with open(DATABASE_PATH, "w") as f:
        json.dump(raw, f)

    print(f"\nSaved database: {DATABASE_PATH}")
    print(f"Total cows registered: {len(database)}")
    print("\nRun 'python3 gradio_app.py' now to test in the browser.")


if __name__ == "__main__":
    main()
