#!/usr/bin/env python3
"""
Add a new disease class to the pipeline, end to end:

  1. Download the given Kaggle dataset (you must already know its slug --
     this script does NOT search Kaggle for you; find/verify the dataset
     with Claude first, same way we found the LSD dataset).
  2. Flatten all images from the download into
     data/raw/cattle_health_v1/<raw-subdir>/
  3. Append a new class entry to configs/config.yaml
  4. git add ONLY configs/config.yaml (never the data), commit, and push.

Data never gets staged or pushed -- only the one-line config change that
tells the training pipeline the new class exists. You still need to run
prepare_dataset.py -> train_baseline.py -> evaluate.py yourself afterward
(this script does not retrain automatically, since that can take a while
and you may want to review the data first).

Usage:
    python3 scripts/add_disease.py \\
        --kaggle-dataset ownername/dataset-slug \\
        --disease-name foot_and_mouth_disease \\
        --raw-subdir fmd_cows

Requires: kaggle CLI already authenticated (same setup used for the LSD
dataset), and being run from the disease-detection/ repo root.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def download_kaggle_dataset(slug: str, dest_zip_dir: Path):
    dest_zip_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_zip_dir)])
    zips = list(dest_zip_dir.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No zip file appeared in {dest_zip_dir} after download -- "
                          f"check the dataset slug is correct.")
    return zips[0]


def flatten_images_into(zip_path: Path, extract_tmp: Path, target_dir: Path):
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True)
    shutil.unpack_archive(str(zip_path), str(extract_tmp))

    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in extract_tmp.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTENSIONS:
            dest = target_dir / img_path.name
            # avoid silent overwrite if two source files share a filename
            if dest.exists():
                dest = target_dir / f"{img_path.stem}_{count}{img_path.suffix}"
            shutil.copy2(img_path, dest)
            count += 1

    shutil.rmtree(extract_tmp)
    zip_path.unlink()  # clean up the zip, keep only the flattened images
    return count


def update_config(config_path: Path, disease_name: str, raw_subdir: str):
    text = config_path.read_text()
    entry_marker = f'name: {disease_name}'
    if entry_marker in text:
        print(f"'{disease_name}' already present in {config_path} -- not adding a duplicate.")
        return False
    new_entry = f"  - name: {disease_name}\n    dir: {raw_subdir}\n"
    with config_path.open("a") as f:
        f.write(new_entry)
    print(f"Appended to {config_path}:\n{new_entry}")
    return True


def git_commit_config_only(repo_root: Path, config_rel_path: str, disease_name: str):
    run(["git", "add", config_rel_path], cwd=repo_root)
    # Confirm nothing else got staged before committing
    status = subprocess.run(["git", "diff", "--cached", "--name-only"],
                             cwd=repo_root, capture_output=True, text=True, check=True)
    staged = status.stdout.strip().splitlines()
    if staged != [config_rel_path]:
        raise SystemExit(f"Refusing to commit -- expected only {config_rel_path} staged, "
                          f"but found: {staged}. Check for unrelated pending changes.")
    run(["git", "commit", "-m", f"Add {disease_name} class to config"], cwd=repo_root)
    run(["git", "push"], cwd=repo_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-dataset", required=True, help="e.g. ownername/dataset-slug")
    ap.add_argument("--disease-name", required=True, help="label the model will output, e.g. mange")
    ap.add_argument("--raw-subdir", required=True,
                     help="folder name under data/raw/cattle_health_v1/ for this class's images")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--raw-dir", default="data/raw/cattle_health_v1")
    ap.add_argument("--skip-push", action="store_true",
                     help="update config and stage the commit but don't push (review first)")
    args = ap.parse_args()

    repo_root = Path.cwd()
    config_path = repo_root / args.config
    target_dir = repo_root / args.raw_dir / args.raw_subdir
    download_dir = repo_root / "data" / "_downloads_tmp"

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"{target_dir} already has files -- skipping download, "
              f"just ensuring config is up to date.")
    else:
        print(f"=== Downloading {args.kaggle_dataset} ===")
        zip_path = download_kaggle_dataset(args.kaggle_dataset, download_dir)
        print(f"=== Extracting into {target_dir} ===")
        n = flatten_images_into(zip_path, download_dir / "_extract", target_dir)
        print(f"Flattened {n} images into {target_dir}")
        if download_dir.exists() and not any(download_dir.iterdir()):
            download_dir.rmdir()

    print(f"\n=== Updating {config_path} ===")
    changed = update_config(config_path, args.disease_name, args.raw_subdir)

    if changed and not args.skip_push:
        print(f"\n=== Committing and pushing config change only ===")
        git_commit_config_only(repo_root, args.config, args.disease_name)
        print("\nDone. Data stayed local; only the config change was pushed.")
    elif changed:
        print("\n--skip-push set: config updated locally, not committed. "
              "Review it, then commit/push yourself when ready.")

    print(f"\nNext: rerun the pipeline to include the new class:")
    print(f"  python3 src/preprocessing/prepare_dataset.py --config {args.config}")
    print(f"  python3 src/training/train_baseline.py --config {args.config}")
    print(f"  python3 src/evaluation/evaluate.py --checkpoint models/cattle_health_classifier.pt "
          f"--data_dir data/processed")


if __name__ == "__main__":
    main()
