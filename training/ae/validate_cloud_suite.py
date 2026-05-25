"""Run a cloud-like local AE validation suite.

`til test ae` is useful for container smoke testing, but its random NPCs have
mis-ranked our AE candidates. This script keeps validation local while testing
the planner against non-random scripted libraries and pressure-heavy suites.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

# Match the shipping Dockerfile unless the caller deliberately overrides these.
os.environ.setdefault("AE_USE_PLAYBOOK", "0")
os.environ.setdefault("AE_USE_OPPONENT_MODEL", "0")

from simulate import run_simulation


LEGACY_SUITES = [
    "library",
    "cloudsuite",
    "pressure2",
    "mixed",
    "rusher",
    "rusher_fast",
    "rusher_safe",
    "hunter",
    "hunter_sticky",
    "bomber",
    "bomber_fast",
    "base_bomber",
    "defender",
    "greedy",
]

FURNISHED_SUITES = [
    "cloudsuite",
    "pressure2",
    "strong_realistic",
    "base_rush_exploit",
    "bracket_proxy",
    "top_seed_proxy",
    "defense_trap",
    "mixed",
]

BRACKET_SUITES = [
    "cloudsuite",
    "pressure2",
    "strong_realistic",
    "base_rush_exploit",
    "bracket_proxy",
    "top_seed_proxy",
]

STRESS_SUITES = [
    "base_rush_exploit",
    "top_seed_proxy",
    "our_base_sieger",
    "scripted_base_attack",
    "safe_base_bomber",
    "cluster_hunter",
]

SUITE_PRESETS = {
    "legacy": LEGACY_SUITES,
    "furnished": FURNISHED_SUITES,
    "bracket": BRACKET_SUITES,
    "stress": STRESS_SUITES,
}

DEFAULT_SUITES = FURNISHED_SUITES

FURNISHED_WEIGHTS = {
    "cloudsuite": 0.18,
    "pressure2": 0.18,
    "strong_realistic": 0.20,
    "base_rush_exploit": 0.18,
    "bracket_proxy": 0.14,
    "top_seed_proxy": 0.08,
    "defense_trap": 0.04,
    "mixed": 0.04,
}


def _aggregate(results: list[dict]) -> dict:
    means = [float(r["mean_score"]) for r in results]
    medians = [float(r["p50"]) for r in results]
    by_suite: dict[str, list[float]] = defaultdict(list)
    for result in results:
        by_suite[str(result.get("suite", ",".join(result.get("opponents", []))))].append(float(result["mean_score"]))
    suite_means = {suite: statistics.mean(values) for suite, values in by_suite.items()}
    weighted_total = 0.0
    weight_sum = 0.0
    for suite, value in suite_means.items():
        weight = FURNISHED_WEIGHTS.get(suite, 1.0)
        weighted_total += weight * value
        weight_sum += weight
    return {
        "run_count": len(results),
        "suite_count": len(by_suite),
        "seed_count": len({int(r.get("seed", 0)) for r in results}),
        "mean_of_means": sum(means) / max(1, len(means)),
        "weighted_mean": weighted_total / weight_sum if weight_sum else 0.0,
        "median_of_medians": statistics.median(medians) if medians else 0.0,
        "worst_mean": min(means) if means else 0.0,
        "best_mean": max(means) if means else 0.0,
        "suite_means": suite_means,
        "worst_suite_mean": min(suite_means.values()) if suite_means else 0.0,
    }


def _format_components(summary: dict) -> str:
    diagnostics = summary.get("diagnostics", {})
    components = diagnostics.get("reward_component_sum", {})
    if not components:
        return "components={}"
    ordered = sorted(components.items(), key=lambda kv: -abs(float(kv[1])))
    return "components={" + ", ".join(f"{k}:{float(v):+.1f}" for k, v in ordered[:5]) + "}"


def _format_base_failures(summary: dict) -> str:
    diagnostics = summary.get("diagnostics", {})
    failures = diagnostics.get("base_failure_classes", {})
    if not failures:
        return "base_failures={}"
    ordered = sorted(failures.items(), key=lambda kv: -int(kv[1]))
    return "base_failures={" + ", ".join(f"{k}:{int(v)}" for k, v in ordered[:4]) + "}"


def _format_decisions(summary: dict) -> str:
    diagnostics = summary.get("diagnostics", {})
    decisions = diagnostics.get("decision_counts", {})
    if not decisions:
        return "decisions={}"
    ordered = sorted(decisions.items(), key=lambda kv: -int(kv[1]))
    return "decisions={" + ", ".join(f"{k}:{int(v)}" for k, v in ordered[:6]) + "}"


def _resolve_suites(args: argparse.Namespace) -> list[str]:
    if args.suites:
        return args.suites
    return list(SUITE_PRESETS[args.preset])


def run_suite(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    per_suite: list[dict] = []
    suites = _resolve_suites(args)
    seeds = args.seeds if args.seeds else [args.seed]
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != "0":
        print(
            f"[warn] PYTHONHASHSEED={hash_seed!r}; launch as PYTHONHASHSEED=0 for comparable AEManager-bearing evals.",
            flush=True,
        )
    for seed in seeds:
        for spec in suites:
            print(f"\n=== {spec} ({args.rounds} rounds, seed={seed}, our={args.our}) ===", flush=True)
            out = run_simulation(
                rounds=args.rounds,
                opponents_spec=spec,
                our_name=args.our,
                log_traj=False,
                seed_start=seed,
                novice=not args.non_novice,
            )
            summary = out["summary"]
            summary["suite"] = spec
            summary["seed"] = seed
            per_suite.append(summary)
            print(
                f"{spec}@{seed}: mean={summary['mean_score']:.4f} "
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
                f"ttd={diagnostics.get('mean_base_pressure_overrides', 0.0):.1f} "
                f"early_end={diagnostics.get('early_end_rate', 0.0):.2f} "
                f"{_format_components(summary)} "
                f"{_format_base_failures(summary)} "
                f"{_format_decisions(summary)}",
                flush=True,
            )

    aggregate = _aggregate(per_suite)
    elapsed = time.monotonic() - started
    report = {
        "rounds_per_suite": args.rounds,
        "our_agent": args.our,
        "novice": not args.non_novice,
        "preset": args.preset,
        "suites": suites,
        "seeds": seeds,
        "pythonhashseed": hash_seed,
        "suite_weights": FURNISHED_WEIGHTS,
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
    parser.add_argument("--our", default="heuristic",
                        help="agent to score: heuristic, option_v2, hybrid, option_hybrid, or opponent:<name>")
    parser.add_argument("--preset", choices=sorted(SUITE_PRESETS), default="furnished",
                        help="suite preset used when --suites is omitted")
    parser.add_argument("--suites", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="optional list of seed starts; runs every suite for every seed")
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
