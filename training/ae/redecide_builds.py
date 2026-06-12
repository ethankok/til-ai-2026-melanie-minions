"""Re-decide every relevant prior AE build under the 2026-06-07 finals-aligned gate.

Re-ranks the best/representative checkpoint of each distinct strategic line against
the deployed incumbent (`confpol-semis2b-u75`), on the TUNE brackets
(`MELEE_BRACKETS`) + the frozen real-competitor HELDOUT_COMPOSITIONS, and applies the
new `melee_eval._promotion_verdict` (placement PRIMARY; absolute reward = non-crater
floors; paired effect + Probability-of-Improvement noise gate; held-out composition
GAP overfit alarm). Supersedes the ad-hoc `data/_cem_transfer_test.py`.

Every candidate is built DEPLOY-FAITHFUL via `tune_planner_weights.build_candidate_spec`
(CBOMB7 weights + AE_CONTENTION=1 + confpol gate 5/10/1); only the consultant
checkpoint varies, except `cem-g00-fixed-03` which swaps the 10 planner weights over
the SAME semis2b-u75 consultant.

Config via env: AE_REDECIDE_HASH (default "0 1 2"), AE_REDECIDE_SIM (default "42"),
AE_REDECIDE_ROUNDS (default "8"), AE_REDECIDE_SMOKE=1 (tiny: 2 cands, 1+1 bracket,
1 hash/sim/round). Output -> data/redecide-builds.{json,log}; touch _DONE on finish.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from melee_eval import (  # noqa: E402
    _evaluate_candidate,
    _promotion_verdict,
    _print_report,
    HELDOUT_COMPOSITIONS,
)
from opponents import MELEE_BRACKETS  # noqa: E402
from tune_planner_weights import build_candidate_spec, incumbent_values  # noqa: E402

DATA = THIS_DIR / "data"


def _ckpt(name: str) -> str:
    return str((THIS_DIR / "checkpoints" / name).resolve())


def _cem_values(label: str = "g00-fixed-03") -> dict:
    rows = [json.loads(l) for l in
            (DATA / "planner-weight-cem-final" / "candidates.jsonl").read_text().splitlines()
            if l.strip()]
    return [r for r in rows if r["candidate"] == label][0]["values"]


INC = incumbent_values()
SEMIS2B = _ckpt("confpol-semis2b-u75.pt")

# (label, planner-weight values, consultant checkpoint). incumbent MUST be first.
CANDIDATES = [
    ("incumbent-semis2b-u75", INC,               SEMIS2B),
    ("cem-g00-fixed-03",      _cem_values(),     SEMIS2B),
    ("native-u100",           INC,               _ckpt("confpol-native-u100.pt")),
    ("semis2c-u75",           INC,               _ckpt("confpol-semis2c-u75.pt")),
    ("pandemonium-v1-u860",   INC,               _ckpt("pandemonium-v1-best-u860.pt")),
    ("pandemonium-v2-u300",   INC,               _ckpt("pandemonium-v2-respawnfix-phase1-u300.pt")),
    ("dir2-v1-latest",        INC,               _ckpt("dir2-v1-latest.pt")),
]


def main() -> int:
    smoke = os.environ.get("AE_REDECIDE_SMOKE") == "1"
    if smoke:
        hash_seeds, sim_seeds, rounds = [0], [42], 1
        cands = CANDIDATES[:2]
        tune = list(MELEE_BRACKETS)[:1]
        held = dict(list(HELDOUT_COMPOSITIONS.items())[:1])
    else:
        hash_seeds = [int(x) for x in os.environ.get("AE_REDECIDE_HASH", "0 1 2").split()]
        sim_seeds = [int(x) for x in os.environ.get("AE_REDECIDE_SIM", "42").split()]
        rounds = int(os.environ.get("AE_REDECIDE_ROUNDS", "8"))
        cands = CANDIDATES
        only = os.environ.get("AE_REDECIDE_ONLY", "").split()
        if only:  # subset by label, keeping the incumbent (CANDIDATES[0]) first
            keep = {CANDIDATES[0][0], *only}
            cands = [c for c in CANDIDATES if c[0] in keep]
        tune = list(MELEE_BRACKETS)
        held = dict(HELDOUT_COMPOSITIONS)

    out_stem = os.environ.get("AE_REDECIDE_OUT", "redecide-builds")

    inc_label = cands[0][0]
    print(f"redecide_builds: {len(cands)} candidates | tune={tune} | heldout={list(held)} | "
          f"hash={hash_seeds} sim={sim_seeds} rounds={rounds} | incumbent={inc_label}",
          flush=True)

    t0 = time.monotonic()
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for label, vals, ckpt in cands:
        print(f"\n[candidate] {label}  (ckpt={Path(ckpt).name})", flush=True)
        try:
            spec = build_candidate_spec(vals, policy_ckpt=ckpt)
            res = _evaluate_candidate(label, spec, tune, hash_seeds, sim_seeds, rounds, False)
            print(f"  [heldout compositions] {label}", flush=True)
            ho = _evaluate_candidate(label, spec, list(held.values()),
                                     hash_seeds, sim_seeds, rounds, False)
            res["heldout_per_bracket"] = ho["per_bracket"]
            res["heldout_mean_placement"] = ho["tune_mean_placement"]
            res["heldout_worst_placement"] = ho["worst_bracket_placement"]
            res["composition_gap"] = ho["tune_mean_placement"] - res["tune_mean_placement"]
            results[label] = res
        except Exception as exc:  # one bad checkpoint must not kill the overnight run
            errors[label] = repr(exc)
            print(f"  !! FAILED: {exc!r}", flush=True)

    elapsed = time.monotonic() - t0

    if inc_label in results:
        _print_report(results, inc_label, tune)

        inc = results[inc_label]
        rows = []
        for label, res in results.items():
            if label == inc_label:
                v = {"promotable": False}
            else:
                v = _promotion_verdict(res, inc)
            rows.append((label, res, v))
        rows.sort(key=lambda r: (not r[2]["promotable"], r[1]["worst_bracket_placement"]))

        print("\n" + "=" * 78)
        print("  FINAL RE-RANK (finals-aligned gate; incumbent = " + inc_label + ")")
        print("=" * 78)
        print(f"  {'build':24s} {'worst':>6s} {'tuneμ':>6s} {'heldμ':>6s} {'gap':>6s} {'verdict':>14s}")
        for label, res, v in rows:
            verdict = "INCUMBENT" if label == inc_label else (
                "PROMOTABLE ✓" if v["promotable"] else "not promotable")
            print(f"  {label:24s} {res['worst_bracket_placement']:6.2f} "
                  f"{res['tune_mean_placement']:6.2f} {res.get('heldout_mean_placement', float('nan')):6.2f} "
                  f"{res.get('composition_gap', float('nan')):+6.2f} {verdict:>14s}")
        if errors:
            print("\n  ERRORS:")
            for label, e in errors.items():
                print(f"    {label}: {e}")
        promotable = [r[0] for r in rows if r[2]["promotable"]]
        print("\n  PROMOTABLE over incumbent: " + (", ".join(promotable) if promotable else "NONE — deploy stays " + inc_label))
        print("=" * 78)

    out = {
        "incumbent": inc_label,
        "hash_seeds": hash_seeds, "sim_seeds": sim_seeds, "rounds": rounds,
        "tune_brackets": tune, "heldout_compositions": held,
        "elapsed_s": elapsed,
        "results": results,
        "errors": errors,
        "verdicts": {label: _promotion_verdict(results[label], results[inc_label])
                     for label in results if label != inc_label} if inc_label in results else {},
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{out_stem}.json").write_text(json.dumps(out, indent=2))
    (DATA / f"_{out_stem}_DONE").write_text(json.dumps({"elapsed_s": elapsed, "errors": list(errors)}))
    print(f"\nelapsed_s={elapsed:.1f}  summary -> data/{out_stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
