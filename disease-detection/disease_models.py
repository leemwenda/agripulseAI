#!/usr/bin/env python3
"""
disease_models.py - evidence-fusion layer for AgriPulse's photo triage.

Registers one "head" per disease that can, even in principle, be read off an
ordinary photo (the ones marked "symptoms+image" in your triage/diseases/*.yaml
files: lumpy skin disease, foot-and-mouth disease, mastitis). Each head is
either:

  - trained:  a checkpoint exists and has been validated against held-out data
  - stubbed:  registered so the architecture and farmer-facing contract are in
              place, but with no checkpoint yet, so it always and honestly
              returns "insufficient visual evidence" rather than a guess.

Every other disease in the knowledge base is symptom-only by design - there
is no reliable photographic signature to train a head against - and is
deliberately NOT registered here. SYMPTOM_ONLY_DISEASE_IDS lists them
explicitly (rather than "everything not in REGISTRY") so the self-check at
the bottom can catch a disease that's neither registered NOR accounted for,
which would mean this file has drifted from the knowledge base.

Place this file at the top level of disease-detection/, next to triage/.
Run it directly to sanity-check the registry against your real YAML files:

    cd ~/agripulse-ai/disease-detection
    python3 disease_models.py
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import yaml

HERE = Path(__file__).resolve().parent
DISEASES_DIR = HERE / "triage" / "diseases"
CHECKPOINTS_DIR = HERE / "triage" / "checkpoints"

DEFAULT_MARGIN = 0.15   # top prob must beat the runner-up by at least this much
DEFAULT_FLOOR = 0.80    # top prob must clear this to call a disease, not just "leading"


@dataclass(frozen=True)
class HeadSpec:
    disease_id: str        # matches the YAML filename stem, e.g. "lumpy_skin_disease"
    label: str              # display name, e.g. "Lumpy skin disease"
    checkpoint_path: Path   # model file this head's probability comes from
    class_name: str         # the softmax class name this head reads from that checkpoint
    trained: bool           # False = stub; always reports "insufficient evidence"
    triage_sign: str        # photo-evidence sign key, used to pull the matching KB entry
    margin: float = DEFAULT_MARGIN
    floor: float = DEFAULT_FLOOR


REGISTRY: List[HeadSpec] = [
    HeadSpec(
        disease_id="lumpy_skin_disease",
        label="Lumpy skin disease",
        checkpoint_path=CHECKPOINTS_DIR / "cattle_health_classifier.pt",
        class_name="lumpy_skin_disease",
        trained=True,
        triage_sign="skin_nodules",
    ),
    HeadSpec(
        disease_id="foot_and_mouth_disease",
        label="Foot-and-mouth disease",
        checkpoint_path=CHECKPOINTS_DIR / "fmd_classifier.pt",   # not trained yet
        class_name="foot_and_mouth_disease",
        trained=False,
        triage_sign="mouth_hoof_lesions",
    ),
    HeadSpec(
        disease_id="mastitis",
        label="Mastitis",
        checkpoint_path=CHECKPOINTS_DIR / "mastitis_classifier.pt",   # not trained yet
        class_name="mastitis",
        trained=False,
        triage_sign="udder_abnormality",
    ),
]

SYMPTOM_ONLY_DISEASE_IDS = {
    "anaplasmosis", "anthrax", "babesiosis", "blackleg", "bovine_tuberculosis",
    "bovine_viral_diarrhoea", "brucellosis", "cbpp", "east_coast_fever",
    "heartwater", "johnes_disease", "rift_valley_fever", "salmonellosis",
    "theileriosis", "trypanosomiasis",
}


def run_heads(classify_fn: Callable[[Path], Dict[str, float]]) -> List[dict]:
    """Call classify_fn once per DISTINCT checkpoint, then read each head's own
    class probability out of that result.

    classify_fn(checkpoint_path) -> {class_name: probability} for the CURRENT
    photo, or {} / missing key if that checkpoint produced nothing (a stub
    head, or a checkpoint that failed to load) - always treated as "no
    evidence" rather than an error.

    Returns one dict per head:
      {"head": HeadSpec, "outcome": "disease"|"healthy"|"uncertain"|"no_evidence", "prob": float|None}
    """
    cache: Dict[Path, Dict[str, float]] = {}
    results = []
    for head in REGISTRY:
        if not head.trained:
            results.append({"head": head, "outcome": "no_evidence", "prob": None})
            continue

        if head.checkpoint_path not in cache:
            cache[head.checkpoint_path] = classify_fn(head.checkpoint_path) or {}
        probs = cache[head.checkpoint_path]

        if head.class_name not in probs:
            results.append({"head": head, "outcome": "no_evidence", "prob": None})
            continue

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_class, top_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0

        if (top_prob - second_prob) < head.margin:
            results.append({"head": head, "outcome": "uncertain", "prob": probs[head.class_name]})
        elif top_class != head.class_name:
            results.append({"head": head, "outcome": "healthy", "prob": probs[head.class_name]})
        elif probs[head.class_name] >= head.floor:
            results.append({"head": head, "outcome": "disease", "prob": probs[head.class_name]})
        else:
            results.append({"head": head, "outcome": "uncertain", "prob": probs[head.class_name]})
    return results


def fuse(head_results: List[dict], cow_found: bool = True) -> dict:
    """Collapse per-head results into ONE farmer-facing outcome.

    Returns either:
      {"outcome": "result", "headline": str, "triage_sign": str, "disease_id": str, "prob": float}
    or:
      {"outcome": "no_evidence", "headline": str, "detail": str, "offer": str}
    """
    diseases = [r for r in head_results if r["outcome"] == "disease"]
    if diseases:
        best = max(diseases, key=lambda r: r["prob"])
        head = best["head"]
        return {
            "outcome": "result",
            "headline": f"Possible {head.label}",
            "triage_sign": head.triage_sign,
            "disease_id": head.disease_id,
            "prob": best["prob"],
        }

    if not cow_found:
        return {
            "outcome": "no_evidence",
            "headline": "No cow clearly visible in this photo.",
            "detail": "Try a clearer, closer shot with the cow filling more of the frame.",
            "offer": "Describe what you see instead",
        }

    trained_heads = [r["head"] for r in head_results if r["head"].trained]
    if not trained_heads:
        lead_note = "This build has no trained photo model available."
    else:
        names = ", ".join(h.label.lower() for h in trained_heads)
        lead_note = (
            f"The photo does not show enough evidence for {names}, "
            f"the only disease{'s' if len(trained_heads) > 1 else ''} this build can "
            f"currently screen from a photo. This doesn't rule out the other diseases "
            f"this app tracks - most of those don't show in a photo at all."
        )

    return {
        "outcome": "no_evidence",
        "headline": "No reliable visual evidence of a detectable disease.",
        "detail": lead_note,
        "offer": "Add symptoms you can see",
    }


def _self_check() -> None:
    yaml_files = sorted(glob.glob(str(DISEASES_DIR / "*.yaml")))
    if not yaml_files:
        print(f"WARNING: no disease YAML files found under {DISEASES_DIR}")
        return

    kb_ids = set()
    for path in yaml_files:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        kb_ids.add(data.get("id") or Path(path).stem)

    registered_ids = {h.disease_id for h in REGISTRY}
    trained_count = sum(1 for h in REGISTRY if h.trained)
    accounted = registered_ids | SYMPTOM_ONLY_DISEASE_IDS

    print(f"Registered heads: {len(REGISTRY)} (trained: {trained_count})")
    print(f"Symptom-only diseases: {len(SYMPTOM_ONLY_DISEASE_IDS)}")
    print(f"Total: {len(accounted)}")

    missing = kb_ids - accounted
    extra = accounted - kb_ids
    if missing:
        print(f"FAIL: diseases in the KB but not accounted for here: {sorted(missing)}")
    elif extra:
        print(f"FAIL: diseases accounted for here but not in the KB: {sorted(extra)}")
    else:
        print("OK - registry matches the 18-disease triage knowledge base.")


if __name__ == "__main__":
    _self_check()
