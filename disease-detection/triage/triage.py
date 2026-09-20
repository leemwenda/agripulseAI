#!/usr/bin/env python3
"""
AgriPulse cattle disease triage (symptom-based).

Ranks possible diseases from the signs a farmer or vet officer observes. It is a
screening aid, not a diagnosis: every result tells the user to involve a vet.

Each disease lives in its own file under diseases/<id>.yaml, so adding a disease
means adding one file (and any new signs to symptoms.yaml). Files are validated
on load, so a typo fails loudly instead of silently scoring wrong.

Usage (from the triage/ folder or anywhere):
    python3 triage.py --list-signs
    python3 triage.py --list-diseases
    python3 triage.py --signs fever,enlarged_lymph_nodes,ticks_present
    python3 triage.py --signs fever,enlarged_lymph_nodes --json     # for the API / Node backend
"""
import argparse
import json
from pathlib import Path

import yaml

BASE = Path(__file__).parent
DETECTION = {"image", "symptoms", "symptoms+image", "lab_only"}
URGENCY = ["routine", "soon", "urgent", "emergency"]
REQUIRED = ["id", "name", "detection", "urgency", "signs", "advice"]
DISCLAIMER = ("This is a screening aid, not a diagnosis. Many cattle diseases look alike; "
              "only a vet (with lab tests where needed) can confirm what the animal has.")


class KBError(Exception):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys (PyYAML would silently keep the last one)."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise KBError(f"duplicate key '{key}' (line {key_node.start_mark.line + 1})")
            seen.add(key)
        return super().construct_mapping(node, deep)


def _load_yaml(path):
    with open(path) as fh:
        try:
            return yaml.load(fh, Loader=_UniqueKeyLoader) or {}
        except KBError as exc:
            raise KBError(f"{path.name}: {exc}")


def load_kb(base=BASE):
    vocab = _load_yaml(base / "symptoms.yaml").get("signs") or {}
    diseases, seen = [], set()
    for f in sorted((base / "diseases").glob("*.yaml")):
        d = _load_yaml(f)
        where = f"diseases/{f.name}"
        for k in REQUIRED:
            if k not in d:
                raise KBError(f"{where}: missing required field '{k}'")
        if d["id"] != f.stem:
            raise KBError(f"{where}: id '{d['id']}' must match the file name '{f.stem}'")
        if d["id"] in seen:
            raise KBError(f"{where}: duplicate id '{d['id']}'")
        if d["detection"] not in DETECTION:
            raise KBError(f"{where}: detection must be one of {sorted(DETECTION)}")
        if d["urgency"] not in URGENCY:
            raise KBError(f"{where}: urgency must be one of {URGENCY}")
        if not d["signs"]:
            raise KBError(f"{where}: needs at least one sign")
        for s, w in d["signs"].items():
            if s not in vocab:
                raise KBError(f"{where}: sign '{s}' is not in symptoms.yaml")
            if w not in (1, 2, 3):
                raise KBError(f"{where}: weight for '{s}' must be 1, 2 or 3 (got {w})")
        if 3 not in d["signs"].values():
            raise KBError(f"{where}: needs at least one characteristic sign (weight 3)")
        la = d.get("likely_at", 0.45)
        if not isinstance(la, (int, float)) or not (0.25 < la <= 1):
            raise KBError(f"{where}: likely_at must be a number above 0.25 and at most 1 (got {la!r})")
        if not isinstance(d.get("possible_needs_key", False), bool):
            raise KBError(f"{where}: possible_needs_key must be true or false")
        for s_ in d.get("solo_signs", []):
            if s_ not in d["signs"]:
                raise KBError(f"{where}: solo_signs entry '{s_}' must also be listed under signs")
        seen.add(d["id"])
        diseases.append(d)
    return vocab, diseases


def rank(kb, observed):
    vocab, diseases = kb
    obs = set(observed)
    unknown = obs - set(vocab)
    if unknown:
        raise KBError(f"Unknown sign(s): {sorted(unknown)}. Run with --list-signs to see valid ids.")
    results = []
    for d in diseases:
        w = d["signs"]
        hits = [s for s in w if s in obs]
        solo = [s for s in d.get("solo_signs", []) if s in obs]
        if len(hits) < d.get("min_signs", 2) and not solo:
            continue
        frac = sum(w[s] for s in hits) / sum(w.values())
        key_hit = any(w[s] == 3 for s in hits)
        if frac >= d.get("likely_at", 0.45) and key_hit:
            band = "likely"
        elif (frac >= 0.25 and (key_hit or not d.get("possible_needs_key", False))) or solo:
            band = "possible"
        else:
            continue
        missing = sorted((s for s in w if s not in obs), key=lambda s: -w[s])[:3]
        results.append({
            "id": d["id"],
            "name": d["name"],
            "band": band,
            "score": round(frac, 2),
            "matched_signs": hits,
            "check_next": [{"sign": s, "ask": vocab[s]["ask"]} for s in missing],
            "urgency": d["urgency"],
            "zoonotic": bool(d.get("zoonotic", False)),
            "notifiable": bool(d.get("notifiable", False)),
            "detection": d["detection"],
            "summary": " ".join(str(d.get("summary", "")).split()),
            "differentials": d.get("differentials", []),
            "confirm_with": d.get("confirm_with", ""),
            "advice": d["advice"],
        })
    results.sort(key=lambda r: (-r["score"], -URGENCY.index(r["urgency"])))
    return results


def print_results(results):
    if not results:
        print("No disease in the knowledge base matches these signs strongly enough.")
        print("That does NOT mean the animal is healthy - if it is unwell, call a vet.")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['name']}  [{r['band'].upper()}, score {r['score']}]  urgency: {r['urgency']}")
        if r["notifiable"]:
            print("   !! Suspected REPORTABLE disease - notify the county veterinary officer / "
                  "veterinary authority before moving any animals or products.")
        if r["zoonotic"]:
            print("   !! Can infect people - handle with protective gear and wash thoroughly.")
        print(f"   {r['summary']}")
        print(f"   Matched signs: {', '.join(r['matched_signs'])}")
        if r["check_next"]:
            print("   Check next:")
            for c in r["check_next"]:
                print(f"     - {c['ask']}")
        if r["differentials"]:
            print(f"   Also consider: {'; '.join(r['differentials'])}")
        if r["confirm_with"]:
            print(f"   Confirmed by: {r['confirm_with']}")
        print("   What to do:")
        for a in r["advice"]:
            print(f"     - {a}")
    print(f"\n{DISCLAIMER}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", help="comma-separated sign ids")
    ap.add_argument("--list-signs", action="store_true")
    ap.add_argument("--list-diseases", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    try:
        kb = load_kb()
    except KBError as exc:
        raise SystemExit(f"Knowledge base error: {exc}")
    vocab, diseases = kb

    if args.list_signs:
        for s, v in vocab.items():
            print(f"{s:<28}{v['ask']}")
        return
    if args.list_diseases:
        for d in diseases:
            print(f"{d['id']:<28}{d['name']}  ({d['detection']}, urgency {d['urgency']})")
        return
    if not args.signs:
        ap.error("give --signs, --list-signs or --list-diseases")

    try:
        results = rank(kb, [s.strip() for s in args.signs.split(",") if s.strip()])
    except KBError as exc:
        raise SystemExit(str(exc))
    if args.json:
        print(json.dumps({"results": results, "disclaimer": DISCLAIMER}, indent=2))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
