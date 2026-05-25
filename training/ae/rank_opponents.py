"""Rank scripted AE opponents for furnished local evaluation.

The goal is not to predict the hidden policy exactly. It is to pick local
opponents that are both competent under the slot-0 AE score and hard for our
current agent to play against. Those two signals make the furnished suite less
dependent on hand-picked intuition.

Usage:
  PYTHONHASHSEED=0 .venv/bin/python training/ae/rank_opponents.py \\
      --rounds 8 \\
      --summary-out training/ae/data/opponent-rank-furnished.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from simulate import run_simulation


DEFAULT_CANDIDATES = [
    "rusher_fast",
    "rusher_safe",
    "base_bomber",
    "scripted_base_attack",
    "spawn_rusher",
    "our_base_sieger",
    "safe_base_bomber",
    "cluster_hunter",
    "counter_defender",
    "hybrid_collector",
]

DEFAULT_AGENT_SUITES = [
    "cloudsuite",
    "pressure2",
    "strong_realistic",
]


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _run_agent_strength(name: str, args: argparse.Namespace) -> dict:
    suite_scores: dict[str, float] = {}
    for suite in args.agent_suites:
        out = run_simulation(
            rounds=args.rounds,
            opponents_spec=suite,
            our_name=f"opponent:{name}",
            log_traj=False,
            seed_start=args.seed,
            novice=not args.non_novice,
        )
        suite_scores[suite] = float(out["summary"]["mean_score"])
    return {
        "suites": suite_scores,
        "mean": _mean(list(suite_scores.values())),
        "worst": min(suite_scores.values()) if suite_scores else 0.0,
    }


def _run_pressure(name: str, args: argparse.Namespace) -> dict:
    out = run_simulation(
        rounds=args.rounds,
        opponents_spec=name,
        our_name=args.our,
        log_traj=False,
        seed_start=args.seed,
        novice=not args.non_novice,
    )
    summary = out["summary"]
    diagnostics = summary.get("diagnostics", {})
    return {
        "our_mean": float(summary["mean_score"]),
        "our_p25": float(summary["p25"]),
        "our_worst": float(summary["min_score"]),
        "our_base_health": float(diagnostics.get("mean_final_base_health", 0.0)),
        "our_early_end_rate": float(diagnostics.get("early_end_rate", 0.0)),
    }


def rank_opponents(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != "0":
        print(
            f"[warn] PYTHONHASHSEED={hash_seed!r}; launch as PYTHONHASHSEED=0 for comparable rankings.",
            flush=True,
        )

    rows: list[dict] = []
    for idx, name in enumerate(args.candidates, start=1):
        print(f"\n=== {idx}/{len(args.candidates)} {name} ===", flush=True)
        agent = _run_agent_strength(name, args)
        pressure = _run_pressure(name, args)
        pressure_hardness = 1.0 - pressure["our_mean"]
        rank_score = args.agent_weight * agent["mean"] + args.pressure_weight * pressure_hardness
        row = {
            "name": name,
            "rank_score": rank_score,
            "agent_strength": agent,
            "pressure": pressure,
        }
        rows.append(row)
        print(
            f"{name}: rank={rank_score:.4f} "
            f"agent_mean={agent['mean']:.4f} "
            f"our_vs_5x={pressure['our_mean']:.4f} "
            f"base={pressure['our_base_health']:.1f}",
            flush=True,
        )

    rows.sort(key=lambda row: -float(row["rank_score"]))
    report = {
        "rounds": args.rounds,
        "our_agent": args.our,
        "seed": args.seed,
        "novice": not args.non_novice,
        "agent_suites": args.agent_suites,
        "agent_weight": args.agent_weight,
        "pressure_weight": args.pressure_weight,
        "pythonhashseed": hash_seed,
        "elapsed_s": time.monotonic() - started,
        "ranked": rows,
    }

    print("\n=== ranked opponents ===")
    print(f"{'rank':>4}  {'name':<22} {'score':>8} {'agent':>8} {'our_vs_5x':>10} {'base':>7}")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>4}  {row['name']:<22} "
            f"{row['rank_score']:>8.4f} "
            f"{row['agent_strength']['mean']:>8.4f} "
            f"{row['pressure']['our_mean']:>10.4f} "
            f"{row['pressure']['our_base_health']:>7.1f}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--our", default="heuristic",
                        help="our agent for pressure checks")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--agent-suites", nargs="+", default=DEFAULT_AGENT_SUITES,
                        help="suites used to score each opponent as slot 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-weight", type=float, default=0.55)
    parser.add_argument("--pressure-weight", type=float, default=0.45)
    parser.add_argument("--non-novice", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = rank_opponents(args)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report, indent=2))
        print(f"summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
