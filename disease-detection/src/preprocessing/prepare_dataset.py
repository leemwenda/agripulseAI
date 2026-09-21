#!/usr/bin/env python3
"""
Organize class folders into train/val/test splits WITHOUT leaking near-copies across splits.

Why this replaces the plain random split: the Kaggle data contains many near-identical photos
(re-saved, resized, several frames of the same animal). A per-file random split puts one copy in
train and its twin in test, so the test score looks great (88-89%) while real accuracy is much
lower. Here, images are first grouped (separately for the download and for your farm photos), and a
whole group always lands in ONE split:
  * near-duplicate images (perceptual hash distance <= data.near_dup_dist) form a group
  * for your own farm photos, every sub-folder is one group, so put one animal (or one visit) per
    sub-folder:  data/farm/healthy/cow_A/*.jpg   data/farm/lumpy_skin_disease/cow_B/*.jpg

Sources (both optional except the first):
  data.raw_dir  + classes[].dir     the downloaded dataset, split by data.split (default 70/15/15)
  data.farm_dir/<class name>/...    your own real photos, split by data.farm_split (default 50/25/25).
                                    They are copied with a "farm_" prefix so the trainer can give them
                                    extra weight and report on them separately.

Reads everything from configs/config.yaml. Adding a disease still only means adding a `classes:` entry.

Usage (from disease-detection/):
    python3 src/preprocessing/prepare_dataset.py --config configs/config.yaml
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def dhash(path, size=8):
    """64-bit perceptual difference hash (same recipe as check_leakage.py)."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    h = 0
    for b in (a[:, 1:] > a[:, :-1]).flatten():
        h = (h << 1) | int(b)
    return h


def list_top_level(folder: Path):
    return [(p, None) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]


def list_farm(folder: Path):
    """Images directly in the class folder, plus one atomic group per sub-folder."""
    items = list_top_level(folder)
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        for q in sorted(sub.rglob("*")):
            if q.is_file() and q.suffix.lower() in IMG_EXTENSIONS:
                items.append((q, f"dir:{sub.name}"))
    return items


def build_groups(items, max_dist):
    """items: [(path, forced_group_or_None)] -> list of index lists (union-find)."""
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    first = {}
    for i, (_, g) in enumerate(items):
        if g is None:
            continue
        if g in first:
            union(i, first[g])
        else:
            first[g] = i

    hashes = np.zeros(n, dtype=np.uint64)
    ok = np.ones(n, dtype=bool)
    for i, (p, _) in enumerate(items):
        try:
            hashes[i] = dhash(p)
        except Exception as exc:
            print(f"  WARNING: cannot read {p} ({exc}) - kept as its own group")
            ok[i] = False
    for i in range(n - 1):
        if not ok[i]:
            continue
        x = hashes[i + 1:] ^ hashes[i]
        bits = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero((bits <= max_dist) & ok[i + 1:])[0]:
            union(i, i + 1 + int(j))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def split_groups(groups, ratios, seed):
    """Assign whole groups to splits so the image counts come out close to the ratios."""
    rng = random.Random(seed)
    groups = [g[:] for g in groups]
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)          # big groups first; stable, so shuffle order breaks ties
    total = sum(len(g) for g in groups)
    target = {s: total * r for s, r in zip(SPLITS, ratios)}
    got = {s: 0 for s in SPLITS}
    out = {s: [] for s in SPLITS}
    for g in groups:
        s = max(SPLITS, key=lambda name: target[name] - got[name])
        out[s].extend(g)
        got[s] += len(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    raw_dir = Path(data["raw_dir"])
    out_dir = Path(data["processed_dir"])
    ratios = data["split"]
    farm_dir = Path(data["farm_dir"]) if data.get("farm_dir") else None
    farm_ratios = data.get("farm_split", [0.5, 0.25, 0.25])
    max_dist = int(data.get("near_dup_dist", 6))
    seed = data.get("seed", 42)
    classes = cfg["classes"]
    if not classes:
        raise SystemExit("No classes defined in config.yaml under `classes:` - nothing to do.")

    # (class name) -> list of (path, forced_group, source)
    found = {}
    print("Discovered images per class:")
    for c in classes:
        name, subdir = c["name"], c["dir"]
        folder = raw_dir / subdir
        rows = []
        if not folder.is_dir():
            print(f"  WARNING: {folder} does not exist - no dataset images for class '{name}'")
        else:
            rows += [(p, g, "dataset") for p, g in list_top_level(folder)]
        farm_n = 0
        if farm_dir is not None and (farm_dir / name).is_dir():
            farm_rows = [(p, g, "farm") for p, g in list_farm(farm_dir / name)]
            farm_n = len(farm_rows)
            rows += farm_rows
        if not rows:
            print(f"  WARNING: no images for class '{name}' - skipping it")
            continue
        found[name] = rows
        print(f"  {name}: {len(rows) - farm_n} dataset + {farm_n} farm")
    if len(found) < 2:
        raise SystemExit("Need at least 2 classes with images. Check raw_dir, farm_dir and the class `dir` values.")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for s in SPLITS:
        for name in found:
            (out_dir / s / name).mkdir(parents=True, exist_ok=True)

    manifest = ["split\tclass\tsource\tgroup\tfile"]
    totals = {}
    print(f"\nGrouping near-duplicates (hash distance <= {max_dist}) and splitting:")
    for name, rows in found.items():
        for source, ratio in (("dataset", ratios), ("farm", farm_ratios)):
            items = [r for r in rows if r[2] == source]
            if not items:
                continue
            groups = build_groups([(p, g) for p, g, _ in items], max_dist)
            assign = split_groups(groups, ratio, seed)
            gid = {}
            for k, g in enumerate(groups):
                for i in g:
                    gid[i] = k
            counts = {}
            for s in SPLITS:
                counts[s] = len(assign[s])
                for i in assign[s]:
                    p = items[i][0]
                    fname = f"farm_{p.parent.name}_{p.name}" if source == "farm" else p.name
                    shutil.copy2(p, out_dir / s / name / fname)
                    manifest.append(f"{s}\t{name}\t{source}\t{gid[i]}\t{p}")
            multi = sum(1 for g in groups if len(g) > 1)
            biggest = max(len(g) for g in groups)
            if len(items) >= 20 and biggest > 0.25 * len(items):
                print(f"  WARNING: {name}/{source}: one group holds {biggest} of {len(items)} images. Near-duplicate "
                      f"chaining is lumping unrelated photos together, so the splits will be lopsided. Lower "
                      f"data.near_dup_dist (now {max_dist}) or check that these photos really are different.")
            for s_name in SPLITS:
                totals[(name, s_name)] = totals.get((name, s_name), 0) + counts[s_name]
            print(f"  {name:<24}{source:<9}{len(items):>5} images in {len(groups):>5} groups "
                  f"({multi} with near-copies)  -> {counts}")

    (out_dir / "_manifest.tsv").write_text("\n".join(manifest) + "\n")
    empty = [f"{c}/{s_name}" for c in found for s_name in ("val", "test") if totals.get((c, s_name), 0) == 0]
    if empty:
        raise SystemExit(f"\nERROR: no images ended up in: {', '.join(empty)}. Training needs every class in every split. "
                         f"Either the class has too few images, or near-duplicate grouping lumped them together "
                         f"(see the WARNING above) - lower data.near_dup_dist or add more photos.")
    print(f"\nClasses in this dataset: {list(found)}")
    print(f"Processed dataset written to: {out_dir}   (audit trail: {out_dir / '_manifest.tsv'})")
    print("Next: python3 check_leakage.py   # should now report 0 near-duplicates between train and val/test")


if __name__ == "__main__":
    main()
