"""Multi-hash-seed wrapper around validate_cloud_suite.py.

The 24 May calibration finding showed that single-hash-seed local evals are
not a reliable cloud proxy: AEManager has hash-order-dependent branches in
candidate scoring, and cloud's hash distribution is effectively random across
submissions. Pinning PYTHONHASHSEED=0 makes one realization reproducible, but
that single realization can still be ~0.05 off the underlying expected value.

This script approximates the hash-distribution average locally by spawning
validate_cloud_suite.py once per (hash_seed, sim_seed) pair and aggregating
per-suite results across all runs. Output is a single JSON report with:

  per_suite[<spec>] = {
      "runs": K * len(sim_seeds),
      "mean": float,                # mean of per-run means
      "se":   float,                # standard error of the mean (across runs)
      "by_run": [ ... ],
  }
  aggregate = {
      "weighted_mean": float,       # using FURNISHED_WEIGHTS in
                                    # validate_cloud_suite.py
      "weighted_mean_se": float,
      "worst_suite_mean": float,
      ...
  }

Typical use as the new gold-standard local gate (replaces single-seed runs):

    PYTHONHASHSEED is intentionally NOT set when invoking — this script
    sets it per-subprocess so we get genuine multi-hash sampling. If the
    parent shell has PYTHONHASHSEED set, that value is ignored for
    the children.

    .venv/bin/python training/ae/multi_seed_eval.py \\
        --rounds 12 --our tactical_hybrid \\
        --preset furnished \\
        --hash-seeds 0 1 2 3 4 --sim-seeds 42 137 \\
        --summary-out training/ae/data/multi-seed-tactical.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
VALIDATE_SCRIPT = THIS_DIR / "validate_cloud_suite.py"

# Re-exported so callers can introspect the aggregate formula without
# importing validate_cloud_suite (which auto-relaunches for hash-seed pinning).
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


def _se(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def _run_validate(
    hash_seed: int,
    sim_seed: int,
    args: argparse.Namespace,
    summary_path: Path,
) -> dict:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(hash_seed)
    env.setdefault("PYTHONUNBUFFERED", "1")
    for kv in args.extra_env or []:
        key, _, value = kv.partition("=")
        if key:
            env[key] = value

    cmd = [
        sys.executable,
        "-u",
        str(VALIDATE_SCRIPT),
        "--rounds", str(args.rounds),
        "--our", args.our,
        "--preset", args.preset,
        "--seed", str(sim_seed),
        "--summary-out", str(summary_path),
    ]
    if args.suites:
        cmd += ["--suites", *args.suites]
    if args.non_novice:
        cmd.append("--non-novice")

    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise RuntimeError(
            f"validate_cloud_suite failed (hash_seed={hash_seed}, sim_seed={sim_seed}, "
            f"returncode={proc.returncode}):\n{tail}"
        )
    if not summary_path.exists():
        raise RuntimeError(
            f"validate_cloud_suite did not write summary at {summary_path} "
            f"(hash_seed={hash_seed}, sim_seed={sim_seed})"
        )
    report = json.loads(summary_path.read_text())
    report["_hash_seed"] = int(hash_seed)
    report["_sim_seed"] = int(sim_seed)
    report["_elapsed_s"] = float(elapsed)
    return report


def _aggregate(reports: list[dict]) -> dict:
    by_suite: dict[str, list[float]] = defaultdict(list)
    weighted_per_run: list[float] = []
    worst_per_run: list[float] = []
    for report in reports:
        suite_means: dict[str, float] = {}
        for entry in report.get("per_suite", []):
            suite = str(entry.get("suite"))
            mean = float(entry.get("mean_score", 0.0))
            by_suite[suite].append(mean)
            suite_means[suite] = mean
        if suite_means:
            weighted_total = 0.0
            weight_sum = 0.0
            for suite, value in suite_means.items():
                w = FURNISHED_WEIGHTS.get(suite, 1.0)
                weighted_total += w * value
                weight_sum += w
            weighted_per_run.append(weighted_total / weight_sum if weight_sum else 0.0)
            worst_per_run.append(min(suite_means.values()))

    per_suite = {}
    for suite, values in by_suite.items():
        per_suite[suite] = {
            "runs": len(values),
            "mean": statistics.mean(values) if values else 0.0,
            "se": _se(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "by_run": values,
        }
    aggregate = {
        "runs": len(reports),
        "suite_count": len(by_suite),
        "weighted_mean": statistics.mean(weighted_per_run) if weighted_per_run else 0.0,
        "weighted_mean_se": _se(weighted_per_run),
        "worst_suite_mean": min((v["mean"] for v in per_suite.values()), default=0.0),
        "worst_per_run_mean": statistics.mean(worst_per_run) if worst_per_run else 0.0,
        "worst_per_run_se": _se(worst_per_run),
    }
    return {"per_suite": per_suite, "aggregate": aggregate}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--our", default="heuristic",
                        help="agent to score; passed through to validate_cloud_suite.py")
    parser.add_argument("--preset", default="furnished",
                        choices=["legacy", "furnished", "bracket", "stress"])
    parser.add_argument("--suites", nargs="+", default=None,
                        help="override preset suites; passed through")
    parser.add_argument("--hash-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="PYTHONHASHSEED values to sample across")
    parser.add_argument("--sim-seeds", nargs="+", type=int, default=[42],
                        help="sim-seed values to pass to validate_cloud_suite.py")
    parser.add_argument("--non-novice", action="store_true")
    parser.add_argument("--extra-env", nargs="+", default=None,
                        help="KEY=VALUE pairs propagated to every subprocess (e.g. "
                             "AE_TACTICAL_PROFILE=bracket AE_TACTICAL_DELTA_CONF=0.85)")
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=None,
                        help="where to write per-run summary jsons (default: same dir "
                             "as --summary-out, with '.runs/' suffix)")
    args = parser.parse_args(argv)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    workdir = args.workdir or args.summary_out.with_suffix(".runs")
    workdir.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    started = time.monotonic()
    plan = [(h, s) for h in args.hash_seeds for s in args.sim_seeds]
    print(
        f"multi_seed_eval: rounds={args.rounds} our={args.our} preset={args.preset} "
        f"hash_seeds={args.hash_seeds} sim_seeds={args.sim_seeds} "
        f"runs={len(plan)} workdir={workdir}",
        flush=True,
    )
    for i, (hash_seed, sim_seed) in enumerate(plan, start=1):
        summary_path = workdir / f"h{hash_seed}_s{sim_seed}.json"
        print(f"  [{i}/{len(plan)}] hash={hash_seed} sim={sim_seed} -> {summary_path.name}", flush=True)
        report = _run_validate(hash_seed, sim_seed, args, summary_path)
        reports.append(report)

    aggregated = _aggregate(reports)
    elapsed = time.monotonic() - started
    out = {
        "rounds_per_suite": args.rounds,
        "our": args.our,
        "preset": args.preset,
        "suites": args.suites,
        "hash_seeds": list(args.hash_seeds),
        "sim_seeds": list(args.sim_seeds),
        "non_novice": bool(args.non_novice),
        "extra_env": args.extra_env or [],
        "elapsed_s": elapsed,
        "per_suite": aggregated["per_suite"],
        "aggregate": aggregated["aggregate"],
    }
    args.summary_out.write_text(json.dumps(out, indent=2))

    print("\n=== multi-seed aggregate ===", flush=True)
    print(json.dumps(aggregated["aggregate"], indent=2), flush=True)
    print("\nper-suite mean ± SE:", flush=True)
    for suite, info in sorted(aggregated["per_suite"].items()):
        print(f"  {suite:24s} {info['mean']:+.4f} ± {info['se']:.4f}   "
              f"(min {info['min']:.4f}, max {info['max']:.4f}, n={info['runs']})", flush=True)
    print(f"\nelapsed_s={elapsed:.1f}", flush=True)
    print(f"summary -> {args.summary_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
