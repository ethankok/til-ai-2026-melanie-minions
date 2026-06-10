"""Finalist validation for the raw_ae CEM campaign's balanced candidate.

Validates `g02-sample-08` (the defense-retaining, best-placement candidate the
campaign produced — NOT the dominated glass-cannon auto-best `g00-sample-01`)
against the incumbent on:
  1. High-N FRESH-seed re-eval on the 5 tune brackets (noise-tail check: does the
     raw_ae edge + placement reproduce off the 42/137 training seeds?).
  2. Held-out composition gap (curry/peroxide) — overfit alarm.

Resilient: each (suite, seed) is evaluated in its own subprocess with a retry, so
a single transient OOM-killed worker (rc=-9) does NOT abort the whole run; it is
recorded as a miss and the rest continues.

Run SOLO (8GB Mac): do not launch alongside another melee gate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from opponents import MELEE_BRACKETS, HELDOUT_COMPOSITIONS  # noqa: E402
from tune_planner_weights import (  # noqa: E402
    DEFAULT_POLICY_CKPT,
    evaluate_values,
    incumbent_values,
)

CAND_NAME = "g02-sample-08"
SEEDS = [42, 137, 7, 99]          # 42/137 = training seeds; 7/99 = FRESH
ROUNDS = 6
GAP_TOL = 0.5                     # AE_GATE_GAP_TOL default
OUT = THIS_DIR / "data" / "g02-finalist-validation.json"


def _candidate_values(name: str) -> dict[str, float]:
    rows = [json.loads(l) for l in
            (THIS_DIR / "data" / "planner-weight-cem-rawae" / "candidates.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        if r["candidate"] == name:
            return {k: float(v) for k, v in r["values"].items()}
    raise SystemExit(f"candidate {name!r} not found in candidates.jsonl")


def _eval_one(label: str, values: dict, suite: str, seed: int, attempts: int = 2):
    """Eval a single (suite, seed); return the per-bracket dict or None on failure."""
    for attempt in range(attempts):
        try:
            r = evaluate_values(label, values, policy_ckpt=DEFAULT_POLICY_CKPT,
                                brackets=[suite], hash_seeds=[0], sim_seeds=[seed],
                                rounds=ROUNDS, non_novice=False)
            return r["per_bracket"][suite]
        except Exception as e:  # noqa: BLE001 — resilience: log + continue
            print(f"    [retry {attempt+1}/{attempts}] {label} {suite} seed={seed} failed: "
                  f"{str(e).splitlines()[-1][:80]}", flush=True)
            time.sleep(5)
    return None


def _aggregate(per_seed: dict[str, list[dict]]) -> dict:
    """Aggregate {suite: [per_bracket dict per seed]} into raw_ae / placement / wrp."""
    import statistics as st
    raws, places = [], []
    per_bracket = {}
    wrp_candidates = []
    for suite, entries in per_seed.items():
        ok = [e for e in entries if e is not None]
        if not ok:
            per_bracket[suite] = {"raw_ae": None, "mean_placement": None, "n": 0}
            continue
        raw = st.mean(e["raw_ae"] for e in ok)
        place = st.mean(e["mean_placement"] for e in ok)
        per_bracket[suite] = {"raw_ae": raw, "mean_placement": place, "n": len(ok)}
        raws.append(raw)
        places.append(place)
        # worst_robust: per opp_mult mean over seeds, then max
        oms = ok[0].get("weighted_placement", {})
        for om in oms:
            wrp_candidates.append(st.mean(e["weighted_placement"][om] for e in ok))
    return {
        "raw_ae": st.mean(raws) if raws else None,
        "mean_placement": st.mean(places) if places else None,
        "worst_robust_placement": max(wrp_candidates) if wrp_candidates else None,
        "per_bracket": per_bracket,
    }


def _run_candidate(label: str, values: dict) -> dict:
    tune, held = {}, {}
    for suite in MELEE_BRACKETS:
        tune[suite] = [_eval_one(label, values, suite, s) for s in SEEDS]
    for name, spec in HELDOUT_COMPOSITIONS.items():
        held[name] = [_eval_one(label, values, spec, s) for s in SEEDS]
    agg_tune = _aggregate(tune)
    agg_held = _aggregate(held)
    gap = None
    if agg_tune["mean_placement"] is not None and agg_held["mean_placement"] is not None:
        gap = agg_held["mean_placement"] - agg_tune["mean_placement"]
    print(f"\n=== {label}: tune raw_ae={agg_tune['raw_ae']} wrp={agg_tune['worst_robust_placement']} "
          f"tune_place={agg_tune['mean_placement']} held_place={agg_held['mean_placement']} gap={gap}", flush=True)
    return {"tune": agg_tune, "heldout": agg_held, "composition_gap": gap, "values": values}


def main() -> int:
    cands = {"incumbent": incumbent_values(), CAND_NAME: _candidate_values(CAND_NAME)}
    out = {}
    for label, vals in cands.items():
        t = time.time()
        out[label] = _run_candidate(label, vals)
        print(f"  ({label} done in {time.time()-t:.0f}s)", flush=True)

    inc, cand = out["incumbent"], out[CAND_NAME]
    dr = (cand["tune"]["raw_ae"] - inc["tune"]["raw_ae"]) if (cand["tune"]["raw_ae"] and inc["tune"]["raw_ae"]) else None
    wrp_ok = (cand["tune"]["worst_robust_placement"] is not None and
              cand["tune"]["worst_robust_placement"] <= inc["tune"]["worst_robust_placement"] + 0.25 + 1e-9)
    gap_ok = (cand["composition_gap"] is not None and inc["composition_gap"] is not None and
              cand["composition_gap"] <= inc["composition_gap"] + GAP_TOL + 1e-9)
    rawae_ok = dr is not None and dr > 1e-9
    print("\n========== FINALIST VERDICT (high-N, fresh seeds 7/99 + train 42/137) ==========")
    print(f"incumbent : tune raw_ae={inc['tune']['raw_ae']:.1f} wrp={inc['tune']['worst_robust_placement']:.2f} "
          f"gap={inc['composition_gap']}")
    print(f"{CAND_NAME}: tune raw_ae={cand['tune']['raw_ae']:.1f} wrp={cand['tune']['worst_robust_placement']:.2f} "
          f"gap={cand['composition_gap']}  Δraw_ae={dr:+.1f}")
    print(f"  raw_ae_gain_ok={rawae_ok}  placement_floor_ok={wrp_ok}  gap_not_widening_ok={gap_ok}")
    verdict = "PROMOTABLE -> hardware A/B" if (rawae_ok and wrp_ok and gap_ok) else "NOT PROMOTABLE -> deploy stays semis2b-u75"
    print(f"  VERDICT: {verdict}")
    out["verdict"] = {"raw_ae_gain_ok": rawae_ok, "placement_floor_ok": wrp_ok,
                      "gap_not_widening_ok": gap_ok, "delta_raw_ae": dr, "summary": verdict}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"WROTE {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
