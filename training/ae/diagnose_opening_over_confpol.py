"""Diagnostic: does the locked opening gate help OVER CONFPOL (not the bare
heuristic) locally?

The gate was validated with the bare C+bomb7 heuristic as the planner. The
shipped build wraps confpol-u860 instead, and the cloud farm regressed
(opening+confpol 0.584 < confpol 0.634). Hypothesis #1: confpol is a *better
opener* than the heuristic, so the fixed opening overwrites confpol's good early
play (regression-to-mean relative to confpol). This tests that locally:

  baseline  = confpol-u860 alone
  treatment = locked opening prefix + confpol-u860 (the shipped agent)

paired full games per spawn vs cloudsuite. If the per-spawn deltas are ~0 or
negative over confpol (unlike the +0.05..+0.20 we saw over the heuristic), the
gate simply doesn't transfer to a confpol planner -> revert.

Run:
    PYTHONHASHSEED=0 .venv/bin/python training/ae/diagnose_opening_over_confpol.py \
        --rounds 8 --slots 0,1,2,3,4,5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

ROOT = Path(__file__).resolve().parents[2]

# Shipped C+bomb7 profile + point confpol at the u860 checkpoint BEFORE any
# manager is constructed.
os.environ.setdefault("AE_ITEM_MISSION_VALUE", "80")
os.environ.setdefault("AE_ITEM_RESOURCE_VALUE", "40")
os.environ.setdefault("AE_ENEMY_BASE_VALUE", "100")
os.environ.setdefault("AE_DIJKSTRA_BOMB_COST", "7.0")
os.environ.setdefault("AE_LEAD_BASE_TETHER", "1")
os.environ.setdefault("AE_LEAD_TETHER_HEALTH", "60.0")
os.environ.setdefault("AE_LEAD_TETHER_WEIGHT", "0.5")
os.environ.setdefault(
    "AE_POLICY_CHECKPOINT",
    str(ROOT / "training" / "ae" / "checkpoints" / "pandemonium-v1-best-u860.pt"),
)

for p in (str(ROOT / "ae" / "src"), str(ROOT / "til-26-ae"), str(ROOT / "training" / "ae")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from confidence_policy_hybrid_manager import ConfidencePolicyHybridAEManager  # noqa: E402
from opening_eval_manager import OpeningHybridManager  # noqa: E402
from opening_sim import BASE_LOCATIONS  # noqa: E402
from opponents import make_opponent, resolve_opponent_spec  # noqa: E402
from simulate import run_one_round  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402

GATE = json.loads((ROOT / "ae" / "src" / "openings_gate.json").read_text())


def _make_opps(spec: str, seed: int):
    names = resolve_opponent_spec(spec)
    return [make_opponent(n, seed=seed + 1000 + i) for i, n in enumerate(names)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="cloudsuite")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--slots", default="0,1,2,3,4,5")
    ap.add_argument("--seed-start", type=int, default=5000)
    args = ap.parse_args()
    slots = [int(s) for s in args.slots.split(",") if s.strip()]

    cfg = default_config(); cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)

    # confpol baseline and the opening+confpol treatment (separate instances so
    # their belief/frame-stacks don't cross-contaminate; share the cached model).
    baseline = ConfidencePolicyHybridAEManager()
    treatment = OpeningHybridManager(heuristic=ConfidencePolicyHybridAEManager(), openings=GATE)

    print(f"planner=confpol-u860  suite={args.suite}  rounds/slot={args.rounds}  slots={slots}\n")
    header = f"{'slot':>5} {'spawn':>7} {'gate':>7} {'confpol':>9} {'open+cp':>9} {'delta':>9} {'z':>6} {'compl%':>7}"
    print(header); print("-" * len(header))

    enabled_deltas: list[float] = []
    for slot in slots:
        base = tuple(BASE_LOCATIONS[slot])
        key = f"{base[0]},{base[1]}"
        on = bool(GATE.get(key))
        b_scores, t_scores, deltas, compl = [], [], [], []
        for r in range(args.rounds):
            seed = args.seed_start + r
            rb = run_one_round(env, baseline, _make_opps(args.suite, seed), False, seed=seed, us_slot=slot)
            rt = run_one_round(env, treatment, _make_opps(args.suite, seed), False, seed=seed, us_slot=slot)
            b_scores.append(rb["score"]); t_scores.append(rt["score"])
            deltas.append(rt["score"] - rb["score"])
            compl.append(1 if (len(treatment.seq) and treatment.idx == len(treatment.seq)) else 0)
        d = np.array(deltas)
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        z = float(d.mean() / se) if se > 0 else 0.0
        if on:
            enabled_deltas.append(float(d.mean()))
        print(f"{slot:>5} {key:>7} {'ON' if on else 'off':>7} {np.mean(b_scores):>9.4f} "
              f"{np.mean(t_scores):>9.4f} {d.mean():>+9.4f} {z:>6.2f} {100*np.mean(compl):>6.0f}%")

    env.close()
    print("-" * len(header))
    if enabled_deltas:
        agg = float(np.mean(enabled_deltas))
        print(f"mean delta over ENABLED spawns (opening helps confpol by): {agg:+.4f}")
        print("READ: if ~0 or negative, the gate doesn't transfer to a confpol planner "
              "(explains the cloud regression) -> revert to confidence_policy_hybrid.")


if __name__ == "__main__":
    main()
