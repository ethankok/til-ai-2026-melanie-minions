"""Run a cloud-like local AE validation suite.

`til test ae` is useful for container smoke testing, but its random NPCs have
mis-ranked our AE candidates. This script keeps validation local while testing
the planner against several opponent styles: random, scripted library, and a
pressure-heavy `cloudsuite` with base rushers/hunters.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# Match the shipping Dockerfile unless the caller deliberately overrides these.
os.environ.setdefault("AE_USE_PLAYBOOK", "0")
os.environ.setdefault("AE_USE_OPPONENT_MODEL", "0")

from simulate import run_simulation


DEFAULT_SUITES = [
    "random",
    "library",
    "cloudsuite",
    "mixed",
    "rusher",
    "hunter",
]


def _aggregate(results: list[dict]) -> dict:
    means = [float(r["mean_score"]) for r in results]
    medians = [float(r["p50"]) for r in results]
    return {
        "suite_count": len(results),
        "mean_of_means": sum(means) / max(1, len(means)),
        "median_of_medians": statistics.median(medians) if medians else 0.0,
        "worst_mean": min(means) if means else 0.0,
        "best_mean": max(means) if means else 0.0,
    }


def _format_components(summary: dict) -> str:
    diagnostics = summary.get("diagnostics", {})
    components = diagnostics.get("reward_component_sum", {})
    if not components:
        return "components={}"
    ordered = sorted(components.items(), key=lambda kv: -abs(float(kv[1])))
    return "components={" + ", ".join(f"{k}:{float(v):+.1f}" for k, v in ordered[:5]) + "}"


def run_suite(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    per_suite: list[dict] = []
    for spec in args.suites:
        print(f"\n=== {spec} ({args.rounds} rounds, our={args.our}) ===", flush=True)
        out = run_simulation(
            rounds=args.rounds,
            opponents_spec=spec,
            our_name=args.our,
            log_traj=False,
            seed_start=args.seed,
            novice=not args.non_novice,
        )
        summary = out["summary"]
        per_suite.append(summary)
        print(
            f"{spec}: mean={summary['mean_score']:.4f} "
            f"p50={summary['p50']:.4f} min={summary['min_score']:.4f} "
            f"max={summary['max_score']:.4f}",
            flush=True,
        )
        diagnostics = summary.get("diagnostics", {})
        print(
            "  diag: "
            f"bombs={diagnostics.get('mean_bombs_placed', 0.0):.1f} "
            f"cells={diagnostics.get('mean_unique_cells_visited', 0.0):.1f} "
            f"hp={diagnostics.get('mean_final_health', 0.0):.1f} "
            f"base={diagnostics.get('mean_final_base_health', 0.0):.1f} "
            f"early_end={diagnostics.get('early_end_rate', 0.0):.2f} "
            f"{_format_components(summary)}",
            flush=True,
        )

    aggregate = _aggregate(per_suite)
    elapsed = time.monotonic() - started
    report = {
        "rounds_per_suite": args.rounds,
        "our_agent": args.our,
        "novice": not args.non_novice,
        "elapsed_s": elapsed,
        "aggregate": aggregate,
        "per_suite": per_suite,
    }
    print("\n=== aggregate ===")
    print(json.dumps(aggregate, indent=2))
    print(f"elapsed_s={elapsed:.1f}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--our", choices=["heuristic", "hybrid"], default="heuristic")
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-novice", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_suite(args)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report, indent=2))
        print(f"summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
