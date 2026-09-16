#!/usr/bin/env python3
"""
AgriPulse cow re-identification - LOCAL training v2.

Improvement over train_local.py: unfreezes ResNet50's last block (layer4)
instead of the whole backbone being frozen. This lets the model actually
adapt some visual features to cow identity, not just re-weight fixed
ImageNet features - should fix the false-duplicate problem we saw.

Still CPU-feasible: layer4 + embedding head is a fraction of ResNet50's
total parameters, so this is much faster than full fine-tuning while
being meaningfully more powerful than a fully frozen backbone.

Run: python3 train_local_v2.py
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

DATA_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_subset")
CHECKPOINT_OUT = "cow_embedding_resnet50_v1.pt"  # same filename - drops into existing scripts
EPOCHS = 8
BATCH_SIZE = 8
LR = 1e-4
NUM_TRIPLETS_TRAIN = 600
NUM_TRIPLETS_VAL = 150

device = torch.device("cpu")
print(f"Using device: {device} (layer4 unfrozen, rest of backbone frozen)")

# ---------- 1. Collect images by cow (handles nested folders) ----------
def collect_images_by_cow(root):
    cow_images = defaultdict(list)
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
            cow_images[cow_folder] = images
    return cow_images

cow_images = collect_images_by_cow(DATA_DIR)
print("\nImages per cow:")
for cow, imgs in cow_images.items():
    print(f"  {cow}: {len(imgs)} images")

# ---------- 2. Split by cow ----------
random.seed(42)
cow_ids = list(cow_images.keys())
random.shuffle(cow_ids)

n = len(cow_ids)
n_val = max(2, int(n * 0.3))
val_cows = cow_ids[:n_val]
train_cows = cow_ids[n_val:]

print(f"\nTrain cows: {train_cows}")
print(f"Val cows: {val_cows}")

# ---------- 3. Triplet dataset ----------
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),  # light augmentation - helps with this little data
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = T.Compose([  # no augmentation for validation
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TripletDataset(Dataset):
    def __init__(self, cow_subset, cow_images_dict, length=500, use_augmentation=True):
        self.cow_ids = [c for c in cow_subset if len(cow_images_dict[c]) >= 2]
        self.cow_images = cow_images_dict
        self.length = length
        self.tfm = transform if use_augmentation else eval_transform
        if len(self.cow_ids) < 2:
            raise ValueError(f"Need at least 2 cows with >=2 images each. Got: {self.cow_ids}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retry a few times if a corrupt/truncated image is hit, instead of crashing
        for attempt in range(10):
            try:
                anchor_cow = random.choice(self.cow_ids)
                pos_imgs = random.sample(self.cow_images[anchor_cow], 2)
                negative_cow = random.choice([c for c in self.cow_ids if c != anchor_cow])
                neg_img = random.choice(self.cow_images[negative_cow])

                anchor = self.tfm(Image.open(pos_imgs[0]).convert("RGB"))
                positive = self.tfm(Image.open(pos_imgs[1]).convert("RGB"))
                negative = self.tfm(Image.open(neg_img).convert("RGB"))
                return anchor, positive, negative
            except (OSError, IOError) as e:
                if attempt == 9:
                    raise RuntimeError(f"Failed 10 times in a row trying to load images: {e}")
                continue

train_dataset = TripletDataset(train_cows, cow_images, length=NUM_TRIPLETS_TRAIN, use_augmentation=True)
val_dataset = TripletDataset(val_cows, cow_images, length=NUM_TRIPLETS_VAL, use_augmentation=False)
print(f"\nTrain triplets: {len(train_dataset)}, Val triplets: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- 4. Model: SAME structure as bulk_register.py/gradio_app.py expect,
# but freeze only the early backbone layers, leaving layer4 (index 7) trainable ----------
class CowEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # backbone children, in order: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        # freeze everything except layer4 (index 7)
        for i, module in enumerate(self.backbone):
            if i != 7:
                for param in module.parameters():
                    param.requires_grad = False
        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

print("\nBuilding model (downloading pretrained weights if not cached)...")
model = CowEmbeddingNet().to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
      f"({100*trainable_params/total_params:.1f}%)")

# ---------- 5. Training loop ----------
triplet_loss_fn = nn.TripletMarginLoss(margin=0.3, p=2)
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=LR)

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
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
        total_loss += loss.item()
    return total_loss / len(loader)

print(f"\nTraining for {EPOCHS} epochs (this will take longer than the frozen version - be patient)...")
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    marker = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        marker = "  <-- best so far"
    print(f"Epoch {epoch+1}/{EPOCHS} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}{marker}")

# ---------- 6. Save checkpoint ----------
torch.save({
    "model_state_dict": model.state_dict(),
    "epoch": EPOCHS,
    "embedding_dim": 128,
}, CHECKPOINT_OUT)
print(f"\nSaved checkpoint: {CHECKPOINT_OUT}")
print("Next: re-run bulk_register.py, then detect_duplicates.py to check if false positives are gone.")
