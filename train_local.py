#!/usr/bin/env python3
"""
AgriPulse cow re-identification - LOCAL training (no Colab needed).

Reads directly from ~/agripulse-ai/multicamcows_balanced (already on this machine).
CPU-optimized: freezes the ResNet50 backbone and only trains the embedding head,
since fine-tuning all of ResNet50 on CPU would be extremely slow.

Run: python3 train_local.py
"""
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_balanced")
CHECKPOINT_OUT = "cow_embedding_resnet50_v1.pt"
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-3
NUM_TRIPLETS_TRAIN = 500
NUM_TRIPLETS_VAL = 100

device = torch.device("cpu")
print(f"Using device: {device} (CPU-optimized: backbone frozen)")

# ---------- 1. Collect images by cow ----------
def collect_images_by_cow(root):
    cow_images = defaultdict(list)
    for cow_folder in sorted(os.listdir(root)):
        cow_path = os.path.join(root, cow_folder)
        if not os.path.isdir(cow_path):
            continue
        images = [os.path.join(cow_path, f) for f in os.listdir(cow_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if images:
            cow_images[cow_folder] = images
    return cow_images

cow_images = collect_images_by_cow(DATA_DIR)
print("\nImages per cow:")
for cow, imgs in cow_images.items():
    print(f"  {cow}: {len(imgs)} images")

# ---------- 2. Split by cow (train/val need 2+ cows each for triplet negatives) ----------
random.seed(42)
cow_ids = list(cow_images.keys())
random.shuffle(cow_ids)

n = len(cow_ids)
n_val = max(2, int(n * 0.25))
val_cows = cow_ids[:n_val]
train_cows = cow_ids[n_val:]

print(f"\nTrain cows: {train_cows}")
print(f"Val cows: {val_cows}")

# ---------- 3. Triplet dataset ----------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TripletDataset(Dataset):
    def __init__(self, cow_subset, cow_images_dict, length=500):
        self.cow_ids = [c for c in cow_subset if len(cow_images_dict[c]) >= 2]
        self.cow_images = cow_images_dict
        self.length = length
        if len(self.cow_ids) < 2:
            raise ValueError(f"Need at least 2 cows with >=2 images each. Got: {self.cow_ids}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        anchor_cow = random.choice(self.cow_ids)
        pos_imgs = random.sample(self.cow_images[anchor_cow], 2)
        negative_cow = random.choice([c for c in self.cow_ids if c != anchor_cow])
        neg_img = random.choice(self.cow_images[negative_cow])

        anchor = transform(Image.open(pos_imgs[0]).convert("RGB"))
        positive = transform(Image.open(pos_imgs[1]).convert("RGB"))
        negative = transform(Image.open(neg_img).convert("RGB"))
        return anchor, positive, negative

train_dataset = TripletDataset(train_cows, cow_images, length=NUM_TRIPLETS_TRAIN)
val_dataset = TripletDataset(val_cows, cow_images, length=NUM_TRIPLETS_VAL)
print(f"\nTrain triplets: {len(train_dataset)}, Val triplets: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- 4. Model: frozen ResNet50 backbone + trainable embedding head ----------
class CowEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128, freeze_backbone=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone_frozen = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        if self.backbone_frozen:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

print("\nDownloading pretrained ResNet50 weights (first run only, cached after)...")
model = CowEmbeddingNet(freeze_backbone=True).to(device)
print(model)

# ---------- 5. Training loop ----------
triplet_loss_fn = nn.TripletMarginLoss(margin=0.3, p=2)
optimizer = torch.optim.Adam(model.embedding.parameters(), lr=LR)  # only train the head

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    for anchor, positive, negative in loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        if train:
            optimizer.zero_grad()
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

print(f"\nTraining for {EPOCHS} epochs (backbone frozen, head only - fast on CPU)...")
for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    print(f"Epoch {epoch+1}/{EPOCHS} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

# ---------- 6. Save checkpoint locally - no download needed! ----------
torch.save({
    "model_state_dict": model.state_dict(),
    "epoch": EPOCHS,
    "embedding_dim": 128,
}, CHECKPOINT_OUT)
print(f"\nSaved checkpoint: {CHECKPOINT_OUT}")
print("You can now run gradio_app.py directly - no download from Colab needed.")
