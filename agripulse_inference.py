#!/usr/bin/env python3
"""
AgriPulse cow re-identification inference.

Given a new photo of a cow, generates its embedding and compares it against
a database of registered cows to return the best match + confidence score.

This is meant to run outside Colab (e.g. on your server / locally) using the
checkpoint saved via save_checkpoint.py.

NOTE: The CowEmbeddingNet class below is reconstructed from your notebook's
printed model summary (ResNet50 backbone + Linear(2048->128) embedding head).
If your actual class differs, replace this class with the exact one from
your notebook so the state_dict loads correctly.
"""
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

CHECKPOINT_PATH = "cow_embedding_resnet50_v1.pt"
DATABASE_PATH = "cow_database.json"       # stores {cow_id: [embedding_list]}
MATCH_THRESHOLD = 0.5                     # below this, treat as "no match / unregistered"


class CowEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=None)  # weights loaded from checkpoint, not pretrained here
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


def load_model(device):
    model = CowEmbeddingNet().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_embedding(model, image_path, device):
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0).cpu()


def load_database():
    if not os.path.exists(DATABASE_PATH):
        return {}
    with open(DATABASE_PATH) as f:
        raw = json.load(f)
    return {cow_id: [torch.tensor(e) for e in embs] for cow_id, embs in raw.items()}


def save_database(database):
    raw = {cow_id: [e.tolist() for e in embs] for cow_id, embs in database.items()}
    with open(DATABASE_PATH, "w") as f:
        json.dump(raw, f)


def register_cow(model, cow_id, image_paths, device):
    """Add a new cow (or more reference photos for an existing cow) to the database."""
    database = load_database()
    embeddings = [get_embedding(model, p, device) for p in image_paths]
    database.setdefault(cow_id, []).extend(embeddings)
    save_database(database)
    print(f"Registered {len(image_paths)} image(s) for {cow_id}. "
          f"Total reference images for this cow: {len(database[cow_id])}")


def identify_cow(model, image_path, device):
    """Given a new photo, find the best-matching registered cow."""
    database = load_database()
    if not database:
        print("No cows registered yet. Use register_cow() first.")
        return None

    query_emb = get_embedding(model, image_path, device)

    best_cow, best_sim = None, -1.0
    all_scores = {}
    for cow_id, ref_embeddings in database.items():
        sims = [F.cosine_similarity(query_emb.unsqueeze(0), ref.unsqueeze(0)).item()
                for ref in ref_embeddings]
        avg_sim = sum(sims) / len(sims)
        all_scores[cow_id] = avg_sim
        if avg_sim > best_sim:
            best_cow, best_sim = cow_id, avg_sim

    print(f"\nQuery image: {image_path}")
    for cow_id, sim in sorted(all_scores.items(), key=lambda x: -x[1]):
        print(f"  {cow_id}: {sim:.3f}")

    if best_sim >= MATCH_THRESHOLD:
        print(f"\nBest match: {best_cow} (confidence: {best_sim:.1%})")
        return {"cow_id": best_cow, "confidence": best_sim, "matched": True}
    else:
        print(f"\nNo confident match (best was {best_cow} at {best_sim:.1%}, "
              f"below threshold {MATCH_THRESHOLD:.1%}). Flag for farmer confirmation.")
        return {"cow_id": best_cow, "confidence": best_sim, "matched": False}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(device)

    # --- Example usage ---
    # 1) Register a cow with a few reference photos:
    # register_cow(model, "AGP-00127", ["photos/cow127_front.jpg", "photos/cow127_side.jpg"], device)

    # 2) Identify a new, unlabeled photo:
    # result = identify_cow(model, "photos/new_photo.jpg", device)

    print("\nEdit the __main__ block above to register cows or run identification.")
