# Honest generalization check: cow_006 was never seen during training.
# Compare its similarity scores in isolation from the "mixed" eval you already ran.
import torch
import torch.nn.functional as F
import numpy as np

model.eval()

def get_embedding(img_path):
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0).cpu()

# Pull test images for the unseen cow and a couple of seen cows to compare against
unseen_cow = "cow_006"
seen_cows_sample = [c for c in cow_images.keys() if c != unseen_cow][:2]

unseen_imgs = cow_images[unseen_cow][:20]  # cap for speed
unseen_embeddings = [get_embedding(p) for p in unseen_imgs]

same_cow_sims = []
for i in range(len(unseen_embeddings)):
    for j in range(i + 1, len(unseen_embeddings)):
        sim = F.cosine_similarity(unseen_embeddings[i].unsqueeze(0), unseen_embeddings[j].unsqueeze(0)).item()
        same_cow_sims.append(sim)

diff_cow_sims = []
for other_cow in seen_cows_sample:
    other_imgs = cow_images[other_cow][:10]
    other_embeddings = [get_embedding(p) for p in other_imgs]
    for e1 in unseen_embeddings[:10]:
        for e2 in other_embeddings:
            sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
            diff_cow_sims.append(sim)

print(f"UNSEEN cow ({unseen_cow}) — never in training data:")
print(f"  Same-cow similarity (cow_006 vs itself):  mean={np.mean(same_cow_sims):.3f}, std={np.std(same_cow_sims):.3f}")
print(f"  Diff-cow similarity (cow_006 vs seen cows): mean={np.mean(diff_cow_sims):.3f}, std={np.std(diff_cow_sims):.3f}")
print(f"\nSeparation gap: {np.mean(same_cow_sims) - np.mean(diff_cow_sims):.3f}")
print("(Compare this gap to your earlier mixed-eval gap of ~0.868 — a smaller gap here is expected and honest.)")
