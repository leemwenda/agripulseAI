#!/usr/bin/env python3
"""
AgriPulse duplicate-cow detector.

Scans the whole cow_database.json and flags any pair of DIFFERENT cow IDs
whose reference photos are suspiciously similar - a sign the same physical
animal may have been registered twice (accidentally, or fraudulently under
two different farmer accounts).

Run: python3 detect_duplicates.py
"""
import json
import os
import itertools

import torch
import torch.nn.functional as F

DATABASE_PATH = "cow_database.json"

# Similarity above this between two DIFFERENT cow IDs is flagged as a possible duplicate.
# Your own eval showed genuinely different cows scoring near 0.0 (mixed) or slightly
# negative (unseen), and the same cow scoring ~0.83+. 0.65 leaves margin below real
# same-cow scores while catching anything clearly higher than normal cross-cow noise.
DUPLICATE_THRESHOLD = 0.65


def load_database():
    if not os.path.exists(DATABASE_PATH):
        print(f"No database found at {DATABASE_PATH}")
        return {}
    with open(DATABASE_PATH) as f:
        raw = json.load(f)
    return {cow_id: [torch.tensor(e) for e in embs] for cow_id, embs in raw.items()}


def average_similarity(embs_a, embs_b):
    """Mean cosine similarity across all photo pairs between two cows."""
    sims = []
    for a in embs_a:
        for b in embs_b:
            sims.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return sum(sims) / len(sims)


def main():
    database = load_database()
    if len(database) < 2:
        print("Need at least 2 registered cows to check for duplicates.")
        return

    cow_ids = sorted(database.keys())
    print(f"Checking {len(cow_ids)} cows for possible duplicates "
          f"(threshold: {DUPLICATE_THRESHOLD:.0%} similarity)\n")

    flagged = []
    all_pairs = list(itertools.combinations(cow_ids, 2))

    for cow_a, cow_b in all_pairs:
        sim = average_similarity(database[cow_a], database[cow_b])
        marker = ""
        if sim >= DUPLICATE_THRESHOLD:
            flagged.append((cow_a, cow_b, sim))
            marker = "  <-- POSSIBLE DUPLICATE"
        print(f"  {cow_a} vs {cow_b}: {sim:.1%}{marker}")

    print(f"\n{'='*50}")
    if flagged:
        print(f"⚠ {len(flagged)} possible duplicate pair(s) found:")
        for cow_a, cow_b, sim in sorted(flagged, key=lambda x: -x[2]):
            print(f"  {cow_a} <-> {cow_b}: {sim:.1%} similarity - review manually")
        print("\nThis does NOT confirm fraud - could also mean two cows with very "
              "similar markings, or photos of the same cow mistakenly registered "
              "under two IDs. Always review flagged pairs manually before acting.")
    else:
        print("No possible duplicates found - all registered cows look distinct.")


if __name__ == "__main__":
    main()
