#!/usr/bin/env python3
"""
Tests the current AgriPulse model on OpenCows2020 - 46 cows the model has
NEVER seen at all (unlike MultiCamCows val cows, which were at least
evaluated during training). This is the real generalization test.

Does three things:
  1. Registers a sample of OpenCows2020 cows using SOME of their photos
  2. Identifies each cow using a DIFFERENT held-out photo of itself
     -> checks if same-cow recognition works on totally unseen animals
  3. Runs duplicate detection across them
     -> checks if the false-positive problem persists with more individuals

Run: python3 test_opencows2020.py
"""
import itertools
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

OPENCOWS_DIR = os.path.expanduser(
    "~/agripulse-ai/opencows2020/extracted/10m32xl88x2b61zlkkgz3fml17/identification/images/test"
)
CHECKPOINT_PATH = "cow_embedding_resnet50_v1.pt"
NUM_COWS_TO_TEST = 15         # sample this many cows out of the 46, for speed
PHOTOS_PER_COW_REGISTER = 5
MATCH_THRESHOLD = 0.5
DUPLICATE_THRESHOLD = 0.65


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
    for cow_folder in sorted(os.listdir(root)):
        cow_path = os.path.join(root, cow_folder)
        if not os.path.isdir(cow_path) or cow_folder.startswith("."):
            continue
        images = []
        for dirpath, _, filenames in os.walk(cow_path):
            for f in filenames:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(dirpath, f))
        if len(images) >= PHOTOS_PER_COW_REGISTER + 1:  # need reg photos + 1 test photo
            cow_images[cow_folder] = sorted(images)
    return cow_images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return

    if not os.path.exists(OPENCOWS_DIR):
        print(f"OpenCows2020 folder not found: {OPENCOWS_DIR}")
        print("Check the path matches your extracted folder structure.")
        return

    model = CowEmbeddingNet().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_cow_images = collect_images_by_cow(OPENCOWS_DIR)
    print(f"Found {len(all_cow_images)} usable cows in OpenCows2020 test split")

    random.seed(42)
    sample_cow_ids = random.sample(list(all_cow_images.keys()),
                                    min(NUM_COWS_TO_TEST, len(all_cow_images)))
    print(f"Testing on {len(sample_cow_ids)} cows: {sample_cow_ids}\n")

    # ---------- 1. Register each sampled cow ----------
    database = {}
    held_out_test_photo = {}

    for cow_id in sample_cow_ids:
        images = all_cow_images[cow_id]
        random.shuffle(images)
        register_imgs = images[:PHOTOS_PER_COW_REGISTER]
        test_img = images[PHOTOS_PER_COW_REGISTER]  # different photo, held out

        embeddings = [get_embedding(model, p, device) for p in register_imgs]
        database[cow_id] = embeddings
        held_out_test_photo[cow_id] = test_img

    print("Registered all sampled cows.\n")

    # ---------- 2. Identification test: does each cow match itself? ----------
    print("=" * 60)
    print("IDENTIFICATION TEST (unseen photo vs correct cow)")
    print("=" * 60)
    correct = 0
    for cow_id, test_img in held_out_test_photo.items():
        query_emb = get_embedding(model, test_img, device)
        scores = {}
        for other_id, ref_embeddings in database.items():
            sims = [F.cosine_similarity(query_emb.unsqueeze(0), ref.unsqueeze(0)).item()
                    for ref in ref_embeddings]
            scores[other_id] = sum(sims) / len(sims)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        best_cow, best_sim = ranked[0]
        is_correct = (best_cow == cow_id)
        correct += is_correct
        status = "CORRECT" if is_correct else f"WRONG (matched {best_cow})"
        print(f"  {cow_id}: best={best_cow} ({best_sim:.1%}) - {status}")

    accuracy = correct / len(sample_cow_ids)
    print(f"\nAccuracy: {correct}/{len(sample_cow_ids)} = {accuracy:.1%}")

    # ---------- 3. Duplicate detection test ----------
    print("\n" + "=" * 60)
    print(f"DUPLICATE DETECTION TEST (threshold: {DUPLICATE_THRESHOLD:.0%})")
    print("=" * 60)
    flagged = []
    for cow_a, cow_b in itertools.combinations(sample_cow_ids, 2):
        sims = [F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                for a in database[cow_a] for b in database[cow_b]]
        sim = sum(sims) / len(sims)
        if sim >= DUPLICATE_THRESHOLD:
            flagged.append((cow_a, cow_b, sim))

    if flagged:
        print(f"{len(flagged)} false positive(s) out of "
              f"{len(sample_cow_ids)*(len(sample_cow_ids)-1)//2} pairs checked:")
        for cow_a, cow_b, sim in sorted(flagged, key=lambda x: -x[2]):
            print(f"  {cow_a} <-> {cow_b}: {sim:.1%}")
    else:
        print(f"No false positives out of "
              f"{len(sample_cow_ids)*(len(sample_cow_ids)-1)//2} pairs checked - clean.")


if __name__ == "__main__":
    main()
