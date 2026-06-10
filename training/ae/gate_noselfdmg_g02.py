"""Re-gate AE_NO_SELF_DAMAGE under the g02-sample-08 deploy weights.

The 9-Jun noselfdmg gate (+125 real_field / +212 adversarial / -275 semis_mixed)
was measured on the OLD deploy profile (mission 80 / resource 40 / base 100 /
tether 0.5). The deploy changed 10 Jun to g02-sample-08 (tether 1.31, base 80,
resource 57 — commit 68164b3), so the flag x g02 payoff matrix is unmeasured.
This is INTEL for the flip-for-finals decision, not a promotion gate: the flag
stays OFF for Semis either way; we need the per-regime deltas under the weights
we actually deploy.

Resumable: every (suite, seed, label) result is appended to a JSONL checkpoint
and skipped on restart, and OFF/ON are evaluated back-to-back per (suite, seed)
so even a killed run yields paired deltas. Decision-critical brackets run first.

Run SOLO (8GB Mac): do not launch alongside another melee gate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from melee_eval import _evaluate_candidate  # noqa: E402
from opponents import MELEE_BRACKETS, HELDOUT_COMPOSITIONS  # noqa: E402
from tune_planner_weights import (  # noqa: E402
    DEFAULT_POLICY_CKPT,
    build_candidate_spec,
)

SEEDS = [42, 137, 7, 99]
ROUNDS = 6
CKPT = THIS_DIR / "data" / "noselfdmg-g02-gate.jsonl"
OUT = THIS_DIR / "data" / "noselfdmg-g02-gate.json"

# Decision-critical first: the three regime brackets, then the rest, then held-out.
_ORDERED = ["real_field", "adversarial", "semis_mixed"]
SUITES: list[tuple[str, object]] = (
    [(b, b) for b in _ORDERED]
    + [(b, b) for b in MELEE_BRACKETS if b not in _ORDERED]
    + [(f"heldout:{n}", spec) for n, spec in HELDOUT_COMPOSITIONS.items()]
)


def _g02_values() -> dict[str, float]:
    rows = [json.loads(l) for l in
            (THIS_DIR / "data" / "planner-weight-cem-rawae" / "candidates.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        if r["candidate"] == "g02-sample-08":
            return {k: float(v) for k, v in r["values"].items()}
    raise SystemExit("g02-sample-08 not found in candidates.jsonl")


def _spec(flag_on: bool) -> dict:
    spec = build_candidate_spec(_g02_values(), policy_ckpt=DEFAULT_POLICY_CKPT)
    if flag_on:
        spec["env"]["AE_NO_SELF_DAMAGE"] = "1"
    return spec


def _load_done() -> dict[tuple[str, int, str], dict]:
    done = {}
    if CKPT.exists():
        for line in CKPT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["suite"], r["seed"], r["label"])] = r["result"]
    return done


def _eval_one(label: str, spec: dict, bracket, seed: int, attempts: int = 2):
    for attempt in range(attempts):
        try:
            r = _evaluate_candidate(label, spec, [bracket], [0], [seed], ROUNDS, False)
            return r["per_bracket"][bracket]
        except Exception as e:  # noqa: BLE001 — resilience: log + continue
            print(f"    [retry {attempt+1}/{attempts}] {label} {bracket} seed={seed} failed: "
                  f"{str(e).splitlines()[-1][:80]}", flush=True)
            time.sleep(5)
    return None


def _aggregate(done: dict) -> dict:
    import statistics as st
    out = {"g02-off": {}, "g02-noselfdmg": {}, "deltas": {}}
    for suite_name, _ in SUITES:
        for label in ("g02-off", "g02-noselfdmg"):
            ok = [v for (s, _seed, l), v in done.items()
                  if s == suite_name and l == label and v is not None]
            if not ok:
                continue
            out[label][suite_name] = {
                "raw_ae": st.mean(e["raw_ae"] for e in ok),
                "mean_placement": st.mean(e["mean_placement"] for e in ok),
                "n": len(ok),
            }
        off = out["g02-off"].get(suite_name)
        on = out["g02-noselfdmg"].get(suite_name)
        if off and on:
            out["deltas"][suite_name] = {
                "d_raw_ae": on["raw_ae"] - off["raw_ae"],
                "d_place": on["mean_placement"] - off["mean_placement"],
                "place_off": off["mean_placement"],
                "place_on": on["mean_placement"],
            }
    return out


def main() -> int:
    specs = {"g02-off": _spec(False), "g02-noselfdmg": _spec(True)}
    done = _load_done()
    if done:
        print(f"resuming: {len(done)} evals already checkpointed", flush=True)
    for suite_name, bracket in SUITES:
        for seed in SEEDS:
            for label, spec in specs.items():
                if (suite_name, seed, label) in done:
                    continue
                res = _eval_one(label, spec, bracket, seed)
                done[(suite_name, seed, label)] = res
                with CKPT.open("a") as f:
                    f.write(json.dumps({"suite": suite_name, "seed": seed,
                                        "label": label, "result": res}) + "\n")
                if res:
                    print(f"  {label:14s} {suite_name:28s} seed={seed} "
                          f"raw_ae={res['raw_ae']:+.1f} place={res['mean_placement']:.2f}", flush=True)

    agg = _aggregate(done)
    print("\n========== FLAG x g02 PAYOFF MATRIX (deltas = ON minus OFF) ==========")
    for suite, d in agg["deltas"].items():
        print(f"  {suite:28s} d_raw_ae {d['d_raw_ae']:+7.1f}  "
              f"place {d['place_off']:.2f} -> {d['place_on']:.2f}", flush=True)
    OUT.write_text(json.dumps(agg, indent=2))
    print(f"WROTE {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
