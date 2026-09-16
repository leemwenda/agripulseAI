#!/usr/bin/env python3
"""
AgriPulse cow re-identification - COMBINED training v2.

Fixes a real bug found in the v1 duplicate-detection results: the model was
separating cows by DATASET SOURCE (MultiCamCows vs OpenCows2020 - background,
camera, lighting) instead of by actual cow identity. Evidence: every MCC-vs-MCC
pair scored 36-78% similarity regardless of which cows, every MCC-vs-OC pair
scored strongly negative regardless of which cows. That's a domain shortcut,
not identity confusion.

Fix: negatives are now sampled mostly from the SAME dataset as the anchor
(hard negatives), so the model can't win by learning "which dataset is this"
- it has to actually learn coat pattern / marking differences between cows
from the same source.

Also fixes: v1 saved the LAST epoch's checkpoint even though epoch 7 had
better val loss than epoch 8 (mild overfitting). This version tracks and
saves the BEST val-loss checkpoint instead.

Run: python3 train_combined_v2.py
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

MULTICAMCOWS_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_subset")
OPENCOWS_ROOT = os.path.expanduser(
    "~/agripulse-ai/opencows2020/extracted/10m32xl88x2b61zlkkgz3fml17/identification/images"
)
CHECKPOINT_OUT = "cow_embedding_resnet50_v1.pt"
EPOCHS = 12                    # a few more, since best-checkpoint tracking means no risk of overfit at the end
BATCH_SIZE = 8
LR = 1e-4
NUM_TRIPLETS_TRAIN = 1500
NUM_TRIPLETS_VAL = 300
MIN_IMAGES_PER_COW = 4
SAME_DOMAIN_NEGATIVE_PROB = 0.85   # <-- the actual fix: 85% of negatives forced same-dataset

device = torch.device("cpu")
print(f"Using device: {device}")


# ---------- 1. Collect images from both datasets (unchanged) ----------
def collect_images_recursive(root, prefix=""):
    cow_images = {}
    if not os.path.exists(root):
        print(f"  (skipping - not found: {root})")
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
        if len(images) >= MIN_IMAGES_PER_COW:
            cow_images[f"{prefix}{cow_folder}"] = sorted(images)
    return cow_images

print("Collecting MultiCamCows2024...")
mcc_images = collect_images_recursive(MULTICAMCOWS_DIR, prefix="mcc_")
print(f"  {len(mcc_images)} cows")

print("Collecting OpenCows2020 (train split)...")
oc_train_images = collect_images_recursive(os.path.join(OPENCOWS_ROOT, "train"), prefix="oc_")
print(f"  {len(oc_train_images)} cows")

print("Collecting OpenCows2020 (test split)...")
oc_test_images = collect_images_recursive(os.path.join(OPENCOWS_ROOT, "test"), prefix="oc_")
print(f"  {len(oc_test_images)} cows")

cow_images = dict(mcc_images)
for cow_id, imgs in oc_train_images.items():
    cow_images.setdefault(cow_id, [])
    cow_images[cow_id].extend(imgs)
for cow_id, imgs in oc_test_images.items():
    cow_images.setdefault(cow_id, [])
    cow_images[cow_id].extend(imgs)

print(f"\nTotal combined: {len(cow_images)} cows, "
      f"{sum(len(v) for v in cow_images.values())} images")

if len(cow_images) < 4:
    print("Not enough cows found - check the OPENCOWS_ROOT path.")
    exit(1)

# domain lookup - which "source" each cow belongs to, based on its ID prefix
def domain_of(cow_id):
    return cow_id.split("_")[0]  # "mcc" or "oc"

cows_by_domain = defaultdict(list)
for cow_id in cow_images:
    cows_by_domain[domain_of(cow_id)].append(cow_id)

print(f"Domains found: { {d: len(v) for d, v in cows_by_domain.items()} }")

# ---------- 2. Split by cow, stratified by domain so val has both sources ----------
random.seed(42)
val_cows, train_cows = [], []
for domain, ids in cows_by_domain.items():
    ids = ids[:]
    random.shuffle(ids)
    n_val = max(1, int(len(ids) * 0.2))
    val_cows += ids[:n_val]
    train_cows += ids[n_val:]

print(f"\nTrain cows: {len(train_cows)}")
print(f"Val cows: {len(val_cows)}")

# ---------- 3. Triplet dataset — DOMAIN-AWARE hard negative sampling ----------
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # helps reduce reliance on lighting/color as a domain cue
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TripletDataset(Dataset):
    def __init__(self, cow_subset, cow_images_dict, length, use_augmentation):
        self.cow_ids = [c for c in cow_subset if len(cow_images_dict[c]) >= 2]
        self.cow_images = cow_images_dict
        self.length = length
        self.tfm = transform if use_augmentation else eval_transform
        if len(self.cow_ids) < 2:
            raise ValueError(f"Need at least 2 cows with >=2 images. Got: {len(self.cow_ids)}")

        self.by_domain = defaultdict(list)
        for c in self.cow_ids:
            self.by_domain[domain_of(c)].append(c)

    def _pick_negative(self, anchor_cow):
        anchor_domain = domain_of(anchor_cow)
        same_domain_pool = [c for c in self.by_domain[anchor_domain] if c != anchor_cow]

        # force the model to learn real identity features by mostly using
        # same-dataset negatives (hard) - only occasionally cross-domain (easy)
        if same_domain_pool and random.random() < SAME_DOMAIN_NEGATIVE_PROB:
            return random.choice(same_domain_pool)
        else:
            other_pool = [c for c in self.cow_ids if c != anchor_cow]
            return random.choice(other_pool)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                anchor_cow = random.choice(self.cow_ids)
                pos_imgs = random.sample(self.cow_images[anchor_cow], 2)
                negative_cow = self._pick_negative(anchor_cow)
                neg_img = random.choice(self.cow_images[negative_cow])

                anchor = self.tfm(Image.open(pos_imgs[0]).convert("RGB"))
                positive = self.tfm(Image.open(pos_imgs[1]).convert("RGB"))
                negative = self.tfm(Image.open(neg_img).convert("RGB"))
                return anchor, positive, negative
            except (OSError, IOError) as e:
                if attempt == 9:
                    raise RuntimeError(f"Failed 10 times loading images: {e}")
                continue

train_dataset = TripletDataset(train_cows, cow_images, NUM_TRIPLETS_TRAIN, use_augmentation=True)
val_dataset = TripletDataset(val_cows, cow_images, NUM_TRIPLETS_VAL, use_augmentation=False)
print(f"\nTrain triplets: {len(train_dataset)}, Val triplets: {len(val_dataset)}")
print(f"Same-domain negative rate: {SAME_DOMAIN_NEGATIVE_PROB:.0%}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- 4. Model (unchanged - same class your other scripts expect) ----------
class CowEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for i, module in enumerate(self.backbone):
            if i != 7:  # keep layer4 trainable, freeze the rest
                for param in module.parameters():
                    param.requires_grad = False
        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

print("\nBuilding model...")
model = CowEmbeddingNet().to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
      f"({100*trainable_params/total_params:.1f}%)")

# ---------- 5. Training loop — now tracks + saves BEST val loss, not last epoch ----------
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
            a, p, n = model(anchor), model(positive), model(negative)
            loss = triplet_loss_fn(a, p, n)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                a, p, n = model(anchor), model(positive), model(negative)
                loss = triplet_loss_fn(a, p, n)
        total_loss += loss.item()
    return total_loss / len(loader)

print(f"\nTraining for {EPOCHS} epochs on {len(cow_images)} cows "
      f"(domain-aware hard negatives - this will take a while)...")
best_val_loss = float("inf")
best_epoch = -1
for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    marker = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        marker = "  <-- best so far, saving checkpoint"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "embedding_dim": 128,
        }, CHECKPOINT_OUT)
    print(f"Epoch {epoch+1}/{EPOCHS} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}{marker}")

print(f"\nBest checkpoint was epoch {best_epoch+1} (val loss {best_val_loss:.4f}) — saved: {CHECKPOINT_OUT}")
print("Next: python3 bulk_register_combined.py && python3 detect_duplicates.py")
print("This time check whether MCC-vs-OC similarities look reasonable (not uniformly")
print("negative) and whether MCC cows still cross-flag each other as much as before.")
