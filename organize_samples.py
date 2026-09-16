#!/usr/bin/env python3
"""
Copies a handful of sample photos per cow into one flat folder,
so they're easy to find and upload through Gradio's file picker.

Creates:
  ~/agripulse-ai/gradio_samples/register/cow_001_1.jpg, cow_001_2.jpg, cow_001_3.jpg, ...
  ~/agripulse-ai/gradio_samples/test/cow_001_test.jpg, ...
"""
import os
import random
import shutil

SOURCE_DIR = os.path.expanduser("~/agripulse-ai/multicamcows_subset")
OUTPUT_DIR = os.path.expanduser("~/agripulse-ai/gradio_samples")
REGISTER_DIR = os.path.join(OUTPUT_DIR, "register")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

NUM_REGISTER_PHOTOS = 3
NUM_TEST_PHOTOS = 1
SEED = 42

def main():
    random.seed(SEED)
    src_root = SOURCE_DIR

    if not os.path.exists(src_root):
        print(f"Source not found: {src_root}")
        return

    os.makedirs(REGISTER_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    cow_dirs = sorted([d for d in os.listdir(src_root)
                        if os.path.isdir(os.path.join(src_root, d))])

    print(f"Organizing samples from {len(cow_dirs)} cows...\n")

    for cow in cow_dirs:
        cow_path = os.path.join(src_root, cow)
        images = []
        for dirpath, _, filenames in os.walk(cow_path):
            for f in filenames:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(dirpath, f))
        images.sort()

        if len(images) < NUM_REGISTER_PHOTOS + NUM_TEST_PHOTOS:
            print(f"  {cow}: only {len(images)} images, skipping (need at least "
                  f"{NUM_REGISTER_PHOTOS + NUM_TEST_PHOTOS})")
            continue

        random.shuffle(images)
        register_imgs = images[:NUM_REGISTER_PHOTOS]
        test_imgs = images[NUM_REGISTER_PHOTOS:NUM_REGISTER_PHOTOS + NUM_TEST_PHOTOS]

        for i, img_path in enumerate(register_imgs, 1):
            dst = os.path.join(REGISTER_DIR, f"{cow}_{i}.jpg")
            shutil.copy2(img_path, dst)

        for img_path in test_imgs:
            dst = os.path.join(TEST_DIR, f"{cow}_test.jpg")
            shutil.copy2(img_path, dst)

        print(f"  {cow}: {len(register_imgs)} register photos, {len(test_imgs)} test photo")

    print(f"\nDone.")
    print(f"Registration photos: {REGISTER_DIR}")
    print(f"Test photos:         {TEST_DIR}")
    print(f"\nOpen these two folders in your file manager alongside the Gradio browser tab.")

if __name__ == "__main__":
    main()
