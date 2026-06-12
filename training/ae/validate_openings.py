"""Outer validation: do the generated openings improve full-game score?

For each spawn slot, plays full 200-tick games with (a) the plain C+bomb7
AEManager and (b) the same heuristic with the divergence-gated opening prefix
(OpeningHybridManager), against the same opponents and env seeds, and reports
the paired score delta. Reuses simulate.run_one_round (score = cum_reward/1000,
the same metric as the calibrated gate) with per-slot rotation via us_slot.

Run:
    PYTHONHASHSEED=0 .venv/bin/python training/ae/validate_openings.py \
        --suite cloudsuite --rounds 12 --slots 0,1,2,3,4,5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

# Set BEFORE any AEManager is constructed so both the baseline and the
# wrapper's internal heuristic match the deployed agent.
os.environ.setdefault("AE_ITEM_MISSION_VALUE", "80")
os.environ.setdefault("AE_ITEM_RESOURCE_VALUE", "40")
os.environ.setdefault("AE_ENEMY_BASE_VALUE", "100")
os.environ.setdefault("AE_DIJKSTRA_BOMB_COST", "7.0")
os.environ.setdefault("AE_LEAD_BASE_TETHER", "1")
os.environ.setdefault("AE_LEAD_TETHER_HEALTH", "60.0")
os.environ.setdefault("AE_LEAD_TETHER_WEIGHT", "0.5")

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "ae" / "src"), str(ROOT / "til-26-ae"), str(ROOT / "training" / "ae")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from ae_manager import AEManager  # noqa: E402
from opening_eval_manager import OpeningHybridManager  # noqa: E402
from opponents import make_opponent, resolve_opponent_spec  # noqa: E402
from simulate import run_one_round  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


def _make_opps(spec: str, seed: int):
    names = resolve_opponent_spec(spec)
    return [make_opponent(n, seed=seed + 1000 + i) for i, n in enumerate(names)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="cloudsuite")
    ap.add_argument("--rounds", type=int, default=12, help="seeds per slot")
    ap.add_argument("--slots", default="0,1,2,3,4,5")
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--candidate-index", type=int, default=0,
                    help="which ranked opening per slot to test (0 = best reward)")
    args = ap.parse_args()

    slots = [int(s) for s in args.slots.split(",") if s.strip()]

    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)

    baseline = AEManager()
    treatment = OpeningHybridManager(heuristic=AEManager(), candidate_index=args.candidate_index)

    print(f"suite={args.suite}  rounds/slot={args.rounds}  slots={slots}  "
          f"candidate={args.candidate_index}\n")
    header = f"{'slot':>5} {'base':>8} {'open':>8} {'delta':>9} {'paired_se':>10} {'open_ticks':>11} {'compl%':>7}"
    print(header)
    print("-" * len(header))

    all_deltas: list[float] = []
    for slot in slots:
        base_scores: list[float] = []
        treat_scores: list[float] = []
        deltas: list[float] = []
        opening_ticks: list[int] = []
        completions: list[int] = []
        for r in range(args.rounds):
            seed = args.seed_start + r
            opps_b = _make_opps(args.suite, seed)
            rb = run_one_round(env, baseline, opps_b, False, seed=seed, us_slot=slot)
            opps_t = _make_opps(args.suite, seed)
            rt = run_one_round(env, treatment, opps_t, False, seed=seed, us_slot=slot)
            base_scores.append(rb["score"])
            treat_scores.append(rt["score"])
            deltas.append(rt["score"] - rb["score"])
            opening_ticks.append(treatment.idx)
            completions.append(1 if (len(treatment.seq) and treatment.idx == len(treatment.seq)) else 0)

        d = np.array(deltas)
        paired_se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        all_deltas.extend(deltas)
        print(f"{slot:>5} {np.mean(base_scores):>8.4f} {np.mean(treat_scores):>8.4f} "
              f"{np.mean(deltas):>+9.4f} {paired_se:>10.4f} "
              f"{np.mean(opening_ticks):>11.1f} {100*np.mean(completions):>6.0f}%")

    env.close()
    d = np.array(all_deltas)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    z = float(np.mean(d) / se) if se > 0 else 0.0
    print("-" * len(header))
    print(f"AGGREGATE paired delta = {np.mean(d):+.4f}  ± {se:.4f} (SE)  z={z:+.2f}  n={len(d)}")
    verdict = ("OPENINGS HELP" if z > 2 else
               "OPENINGS HURT" if z < -2 else
               "INCONCLUSIVE (|z|<2) — within noise at this n")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
