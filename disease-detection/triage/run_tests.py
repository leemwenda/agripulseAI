#!/usr/bin/env python3
"""
Regression tests for the symptom triage knowledge base.

Every disease gets a file in tests/<disease_id>.yaml. Each case lists the signs a farmer
ticks and what the ranking MUST look like. Run this after EVERY change to triage.py,
symptoms.yaml or any diseases/*.yaml, so adding disease #10 cannot silently break disease #1.

Case format:
    - name: what this case is checking
      signs: [fever, enlarged_lymph_nodes]
      expect:
        east_coast_fever: {band: likely, rank: 1}
        lumpy_skin_disease: {band: not_likely}

`band` values:
    likely | possible   exactly that band
    listed             likely or possible
    not_likely         possible or absent (never likely)
    absent             not listed at all
`rank` (optional): 1-based position in the ranked list.

Usage (from triage/):
    python3 run_tests.py            # all test files
    python3 run_tests.py -v         # also print each ranking
    python3 run_tests.py east_coast_fever
"""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import triage as engine  # noqa: E402

BAND_OK = {
    "likely": lambda b: b == "likely",
    "possible": lambda b: b == "possible",
    "listed": lambda b: b in ("likely", "possible"),
    "not_likely": lambda b: b in (None, "possible"),
    "absent": lambda b: b is None,
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    kb = engine.load_kb(HERE)
    known = {d["id"] for d in kb[1]}
    files = sorted((HERE / "tests").glob("*.yaml"))
    if args:
        files = [f for f in files if f.stem in args]
    if not files:
        raise SystemExit("No test files found in tests/")

    total = failed = 0
    for f in files:
        cases = yaml.safe_load(f.read_text()) or []
        print(f"\n{f.stem}: {len(cases)} cases")
        for c in cases:
            total += 1
            results = engine.rank(kb, c["signs"])
            order = [r["id"] for r in results]
            band = {r["id"]: r["band"] for r in results}
            problems = []
            for did, want in c["expect"].items():
                if did not in known:
                    problems.append(f"{did}: not in the knowledge base")
                    continue
                if want["band"] not in BAND_OK:
                    problems.append(f"{did}: unknown band '{want['band']}'")
                    continue
                if not BAND_OK[want["band"]](band.get(did)):
                    problems.append(f"{did}: wanted {want['band']}, got {band.get(did) or 'absent'}")
                if "rank" in want and (did not in order or order.index(did) + 1 != want["rank"]):
                    got = order.index(did) + 1 if did in order else None
                    problems.append(f"{did}: wanted rank {want['rank']}, got {got}")
            mark = "PASS" if not problems else "FAIL"
            print(f"  [{mark}] {c['name']}")
            if verbose or problems:
                shown = ", ".join(f"{r['name']} {r['band']} {r['score']}" for r in results) or "nothing listed"
                print(f"         signs: {', '.join(c['signs'])}\n         result: {shown}")
            for p in problems:
                print(f"         -> {p}")
            failed += bool(problems)

    print(f"\n{total - failed}/{total} cases passed" + ("" if not failed else f"  ({failed} FAILED)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
