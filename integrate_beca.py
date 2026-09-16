# Run this AFTER extracting the BECA val zip to a folder (e.g. data/beca_val)
# It merges BECA identities into the same cow_images dict used by MultiCamCows,
# with a prefix so IDs never collide between the two sources.

import os
from collections import defaultdict

BECA_ROOT = "data/beca_val"   # change if your extracted folder name differs
BECA_PREFIX = "beca_"         # keeps BECA cow IDs distinct from cow_001-006

def integrate_beca(cow_images_dict, beca_root=BECA_ROOT, prefix=BECA_PREFIX):
    """
    Adds BECA cattle into the existing cow_images dict (same structure as
    collect_images_by_cow output: {cow_id: [image_paths]}).
    Assumes beca_root contains one subfolder per individual cow ID.
    """
    if not os.path.exists(beca_root):
        print(f"BECA folder not found at {beca_root} - skipping integration.")
        print("Extract the BECA val zip first, or update BECA_ROOT to match.")
        return cow_images_dict

    added_cows = 0
    added_images = 0

    for cow_folder in sorted(os.listdir(beca_root)):
        cow_path = os.path.join(beca_root, cow_folder)
        if not os.path.isdir(cow_path):
            continue
        images = [
            os.path.join(cow_path, f)
            for f in os.listdir(cow_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not images:
            continue
        beca_id = f"{prefix}{cow_folder}"
        cow_images_dict[beca_id] = images
        added_cows += 1
        added_images += len(images)

    print(f"Integrated BECA: +{added_cows} cows, +{added_images} images")
    return cow_images_dict

# Usage once BECA is extracted:
cow_images = integrate_beca(cow_images)

print(f"\nCombined dataset:")
print(f"Total cows: {len(cow_images)}")
print(f"Total images: {sum(len(v) for v in cow_images.values())}")
print(f"\nBreakdown:")
for cow, imgs in sorted(cow_images.items()):
    print(f"  {cow}: {len(imgs)} images")
