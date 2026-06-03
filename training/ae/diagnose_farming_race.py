#!/usr/bin/env python
"""Phase-0 farming-race diagnosis: why do we lose the all_farmer melee bracket?

Runs our shipped heuristic against a melee bracket and dumps the diagnostics
simulate already collects. Discriminates three hypotheses:
  (i)   high mean_freeze_ticks + negative margin -> we burn ticks frozen/fighting
        -> Phase A stun tax + Phase B threat-aversion is the lever.
  (ii)  low mean_final_base_health / base_failure_classes -> we get base-rushed
        -> Phase B base-defense + tether is the lever.
  (iii) normal freeze + healthy base + low item collection (reward_component_sum)
        + negative margin -> we are out-collected on raw pathing (a speed/LUT gap;
        this spec's levers will NOT fix it -> STOP and report).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from simulate import run_simulation
import opponents as _opp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--our", default="heuristic",
                    help="simulate._make_our_agent name (e.g. heuristic)")
    ap.add_argument("--bracket", default="all_farmer",
                    help="key in opponents.OPPONENT_SUITES (melee bracket)")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # OPPONENT_SUITES maps a bracket name -> list of opponent names;
    # run_simulation resolves names via make_opponent (foreign factory for
    # non-builtins). MELEE_BRACKETS is just the ordered list of suite keys.
    spec = ",".join(_opp.OPPONENT_SUITES[args.bracket])
    out = run_simulation(
        rounds=args.rounds, opponents_spec=spec, our_name=args.our,
        log_traj=False, seed_start=args.seed,
    )
    s = out["summary"]
    d = s["diagnostics"]
    report = {
        "our": args.our,
        "bracket": args.bracket,
        "rounds": args.rounds,
        "mean_score": s["mean_score"],
        "mean_placement": s["mean_placement"],
        "mean_margin": s["mean_margin"],
        "win_rate": s["win_rate"],
        "mean_freeze_ticks": d.get("mean_freeze_ticks"),
        "mean_final_base_health": d["mean_final_base_health"],
        "reward_component_sum": d["reward_component_sum"],
        "decision_counts": d["decision_counts"],
        "base_failure_classes": d["base_failure_classes"],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
