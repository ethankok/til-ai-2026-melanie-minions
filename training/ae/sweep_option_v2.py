"""Controlled sweep for AE option-v2 planner settings.

This is intentionally smaller than the heuristic knob sweeps. Option-v2 lost
badly in its first gate, so this script tests the few settings most likely to
explain that miss:

- base-vs-mission priority
- commitment/hysteresis strength
- base-defense radius/health
- commitment disabled entirely

The baseline candidate is the current shipping heuristic. Every option-v2
candidate is evaluated against the same local suites and receives baseline
deltas plus rejection reasons.
"""

from __future__ import annotations

# Pin PYTHONHASHSEED=0 before any other import. See sweep_heuristic_knobs.py
# for the full rationale.
import os
import sys

if os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

import argparse
import contextlib
import io
import json
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
AE_SRC = REPO_ROOT / "ae" / "src"
TIL_AE = REPO_ROOT / "til-26-ae"
for path in (str(THIS_DIR), str(AE_SRC), str(TIL_AE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from simulate import run_simulation  # noqa: E402


SAFETY_ENV = {
    "AE_USE_PLAYBOOK": "0",
    "AE_USE_OPPONENT_MODEL": "0",
    "AE_MCTS": "0",
    "AE_MCTS_LOG_TIMING": "0",
    "AE_TIER1_REPEAT_KILL": "1",
    "AE_TIER1_NO_STAY_PENALTY": "1",
    "AE_TIER1_PREDICTIVE_WALK": "1",
}

BASELINE_ENV = {
    "AE_PLANNER": "legacy",
    "AE_ENEMY_BASE_VALUE": "130",
    "AE_DIST_PENALTY": "1.15",
    "AE_DIJKSTRA_BOMB_COST": "5",
    "AE_ITEM_MISSION_VALUE": "50",
    "AE_ITEM_RESOURCE_VALUE": "25",
    "AE_ITEM_RECON_VALUE": "10",
    "AE_BASE_DEFENSE_HEALTH": "60",
    "AE_BASE_DEFENSE_RADIUS": "6",
    "AE_BASE_DEFENSE_VALUE": "60",
    "AE_BASE_DEFENSE_EMERGENCY_VALUE": "150",
    "AE_PATH_THREAT_PENALTY": "2",
    "AE_CELL_THREAT_PENALTY": "5",
    "AE_ENEMY_CHASE_VALUE": "0",
    "AE_ENEMY_CHASE_RADIUS": "4",
    "AE_TIER1_DEFENSE": "0",
    "AE_TIER1_SHARED_CREDIT": "0",
}

OPTION_BASE_ENV = {
    **BASELINE_ENV,
    "AE_PLANNER": "option_v2",
    "AE_OPTION_BASE_BIAS": "120",
    "AE_OPTION_MISSION_BIAS": "70",
    "AE_OPTION_COMMIT_MARGIN": "10",
    "AE_OPTION_DISABLE_COMMIT": "0",
}

CONTROLLED_GRID = {
    "AE_OPTION_BASE_BIAS": ["90", "120", "150", "180"],
    "AE_OPTION_MISSION_BIAS": ["35", "50", "70"],
    "AE_OPTION_COMMIT_MARGIN": ["0", "5", "10", "20"],
    "AE_OPTION_DISABLE_COMMIT": ["0", "1"],
    "AE_BASE_DEFENSE_RADIUS": ["6", "8", "10"],
    "AE_BASE_DEFENSE_HEALTH": ["60", "80", "101"],
}

DEFAULT_OBJECTIVE_WEIGHTS = {
    "library": 0.15,
    "cloudsuite": 0.30,
    "pressure2": 0.25,
    "mixed": 0.15,
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    env: dict[str, str]
    our_name: str
    source: str


def _env_signature(env: dict[str, str], our_name: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    return our_name, tuple(sorted((k, str(v)) for k, v in env.items()))


def _with_option_overrides(overrides: dict[str, str]) -> dict[str, str]:
    env = dict(OPTION_BASE_ENV)
    env.update({key: str(value) for key, value in overrides.items()})
    return env


def _anchor_candidates() -> list[Candidate]:
    return [
        Candidate("baseline", dict(BASELINE_ENV), "heuristic", "baseline"),
        Candidate("option_v2_default", dict(OPTION_BASE_ENV), "option_v2", "anchor"),
        Candidate(
            "option_no_commit",
            _with_option_overrides({"AE_OPTION_DISABLE_COMMIT": "1", "AE_OPTION_COMMIT_MARGIN": "0"}),
            "option_v2",
            "anchor",
        ),
        Candidate(
            "option_base_soft",
            _with_option_overrides({"AE_OPTION_BASE_BIAS": "90", "AE_OPTION_MISSION_BIAS": "70"}),
            "option_v2",
            "anchor",
        ),
        Candidate(
            "option_base_strong",
            _with_option_overrides({"AE_OPTION_BASE_BIAS": "180", "AE_OPTION_MISSION_BIAS": "35"}),
            "option_v2",
            "anchor",
        ),
        Candidate(
            "option_defense_wide",
            _with_option_overrides({"AE_BASE_DEFENSE_RADIUS": "10", "AE_BASE_DEFENSE_HEALTH": "101"}),
            "option_v2",
            "anchor",
        ),
    ]


def generate_candidates(total: int, seed: int) -> list[Candidate]:
    if total < 1:
        raise ValueError("--candidates must be >= 1")
    rng = random.Random(seed)
    candidates: list[Candidate] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    def append(candidate: Candidate) -> None:
        sig = _env_signature(candidate.env, candidate.our_name)
        if sig in seen:
            return
        seen.add(sig)
        candidates.append(candidate)

    for candidate in _anchor_candidates():
        append(candidate)
        if len(candidates) >= total:
            return candidates

    keys = list(CONTROLLED_GRID)
    combos = list(product(*(CONTROLLED_GRID[key] for key in keys)))
    rng.shuffle(combos)
    for idx, values in enumerate(combos):
        if len(candidates) >= total:
            break
        overrides = {key: value for key, value in zip(keys, values)}
        append(Candidate(f"option_grid_{idx:04d}", _with_option_overrides(overrides), "option_v2", "grid"))
    return candidates


def _select_candidates(all_candidates: list[Candidate], ids: list[str] | None) -> list[Candidate]:
    if not ids:
        return all_candidates
    by_id = {candidate.candidate_id: candidate for candidate in all_candidates}
    missing = [candidate_id for candidate_id in ids if candidate_id not in by_id]
    if missing:
        known = ", ".join(sorted(by_id)[:40])
        raise SystemExit(f"unknown candidate ids {missing}; first known ids: {known}")
    return [by_id[candidate_id] for candidate_id in ids]


def _apply_env(candidate_env: dict[str, str]) -> dict[str, str | None]:
    merged = {**SAFETY_ENV, **candidate_env}
    old = {key: os.environ.get(key) for key in merged}
    for key, value in merged.items():
        os.environ[key] = str(value)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _objective(per_suite: dict[str, dict[str, Any]], suites: list[str], worst_weight: float) -> float:
    means = {suite: float(per_suite[suite]["mean_score"]) for suite in suites}
    worst = min(means.values()) if means else 0.0
    weighted_total = 0.0
    used_weight = 0.0
    for suite, weight in DEFAULT_OBJECTIVE_WEIGHTS.items():
        if suite in means:
            weighted_total += weight * means[suite]
            used_weight += weight
    primary = weighted_total / used_weight if used_weight > 0 else _mean(list(means.values()))
    return (1.0 - worst_weight) * primary + worst_weight * worst


def _run_candidate(payload: tuple[Candidate, int, list[str], int, bool, float]) -> dict[str, Any]:
    candidate, rounds, suites, seed, novice, worst_weight = payload
    old_env = _apply_env(candidate.env)
    captured = io.StringIO()
    try:
        per_suite: dict[str, dict[str, Any]] = {}
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            for suite_index, suite in enumerate(suites):
                out = run_simulation(
                    rounds=rounds,
                    opponents_spec=suite,
                    our_name=candidate.our_name,
                    log_traj=False,
                    seed_start=seed + 10_000 * suite_index,
                    novice=novice,
                    our_kwargs=None,
                )
                summary = out["summary"]
                per_suite[suite] = {
                    "mean_score": float(summary["mean_score"]),
                    "p50": float(summary["p50"]),
                    "min_score": float(summary["min_score"]),
                    "max_score": float(summary["max_score"]),
                    "diagnostics": summary.get("diagnostics", {}),
                }
        suite_means = [float(per_suite[suite]["mean_score"]) for suite in suites]
        return {
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "our_name": candidate.our_name,
            "env": candidate.env,
            "aggregate": {
                "mean_of_means": _mean(suite_means),
                "median_of_medians": statistics.median(float(per_suite[suite]["p50"]) for suite in suites),
                "worst_mean": min(suite_means),
                "best_mean": max(suite_means),
                "objective": _objective(per_suite, suites, worst_weight),
            },
            "per_suite": per_suite,
            "ok": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "our_name": candidate.our_name,
            "env": candidate.env,
            "ok": False,
            "error": repr(exc),
            "captured_tail": captured.getvalue()[-4000:],
        }
    finally:
        _restore_env(old_env)


def _reject_reasons(result: dict[str, Any]) -> list[str]:
    delta = result.get("delta_vs_baseline", {})
    suite_delta = delta.get("suites", {})
    reasons = []
    if suite_delta.get("library", 0.0) < -0.05:
        reasons.append("library_drop_gt_0.05")
    if suite_delta.get("cloudsuite", 0.0) < -0.03:
        reasons.append("cloudsuite_drop_gt_0.03")
    if float(delta.get("worst_mean", 0.0)) < -0.03:
        reasons.append("worst_suite_drop_gt_0.03")
    if float(delta.get("objective", 0.0)) < 0.0:
        reasons.append("objective_below_baseline")
    return reasons


def _add_baseline_deltas(results: list[dict[str, Any]], suites: list[str]) -> None:
    baseline = next((r for r in results if r.get("candidate_id") == "baseline" and r.get("ok")), None)
    if baseline is None:
        return
    baseline_obj = float(baseline["aggregate"]["objective"])
    baseline_worst = float(baseline["aggregate"]["worst_mean"])
    baseline_suite = {
        suite: float(baseline["per_suite"][suite]["mean_score"])
        for suite in suites
        if suite in baseline["per_suite"]
    }
    for result in results:
        if not result.get("ok"):
            continue
        result["delta_vs_baseline"] = {
            "objective": float(result["aggregate"]["objective"]) - baseline_obj,
            "worst_mean": float(result["aggregate"]["worst_mean"]) - baseline_worst,
            "suites": {
                suite: float(result["per_suite"][suite]["mean_score"]) - baseline_suite[suite]
                for suite in suites
                if suite in baseline_suite and suite in result["per_suite"]
            },
        }
        result["reject_reasons"] = _reject_reasons(result)


def _rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [r for r in results if r.get("ok")],
        key=lambda r: (
            float(r["aggregate"]["objective"]),
            float(r["aggregate"]["worst_mean"]),
            float(r["aggregate"]["mean_of_means"]),
        ),
        reverse=True,
    )


def _format_suite_scores(result: dict[str, Any], suites: list[str]) -> str:
    return " ".join(
        f"{suite}={float(result['per_suite'][suite]['mean_score']):.4f}"
        for suite in suites
        if suite in result.get("per_suite", {})
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    all_candidates = generate_candidates(args.candidates, args.seed)
    candidates = _select_candidates(all_candidates, args.candidate_ids)
    if "baseline" not in {c.candidate_id for c in candidates}:
        candidates = [all_candidates[0], *candidates]

    jobs = args.jobs or max(1, min(8, (os.cpu_count() or 2) - 1))
    novice = not args.non_novice
    started = time.monotonic()
    payloads = [
        (candidate, args.rounds, args.suites, args.seed, novice, args.worst_weight)
        for candidate in candidates
    ]

    print(
        f"option-v2 sweep: candidates={len(candidates)} rounds={args.rounds} "
        f"suites={','.join(args.suites)} jobs={jobs} seed={args.seed}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_run_candidate, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if result.get("ok"):
                print(
                    f"[{completed:>4}/{len(candidates)}] {result['candidate_id']:<24} "
                    f"obj={float(result['aggregate']['objective']):.4f} "
                    f"worst={float(result['aggregate']['worst_mean']):.4f} "
                    f"{_format_suite_scores(result, args.suites)}",
                    flush=True,
                )
            else:
                print(f"[{completed:>4}/{len(candidates)}] {result['candidate_id']:<24} ERROR {result['error']}", flush=True)

    _add_baseline_deltas(results, args.suites)
    ranked = _rank(results)
    elapsed = time.monotonic() - started
    report = {
        "rounds_per_suite": args.rounds,
        "suites": args.suites,
        "candidate_count": len(candidates),
        "seed": args.seed,
        "novice": novice,
        "jobs": jobs,
        "worst_weight": args.worst_weight,
        "elapsed_s": elapsed,
        "safety_env": SAFETY_ENV,
        "baseline_env": BASELINE_ENV,
        "option_base_env": OPTION_BASE_ENV,
        "controlled_grid": CONTROLLED_GRID,
        "results": sorted(results, key=lambda r: r["candidate_id"]),
        "ranked_candidate_ids": [r["candidate_id"] for r in ranked],
    }

    print("\n=== top candidates ===", flush=True)
    print(f"{'rank':>4}  {'candidate':<24}  {'obj':<8}  {'delta':<8}  {'worst':<8}  suites", flush=True)
    for i, result in enumerate(ranked[: args.top], start=1):
        delta = result.get("delta_vs_baseline", {}).get("objective", 0.0)
        rejected = ",".join(result.get("reject_reasons", []))
        suffix = f" reject={rejected}" if rejected else ""
        print(
            f"{i:>4}  {result['candidate_id']:<24}  "
            f"{float(result['aggregate']['objective']):<8.4f}  "
            f"{float(delta):<+8.4f}  "
            f"{float(result['aggregate']['worst_mean']):<8.4f}  "
            f"{_format_suite_scores(result, args.suites)}{suffix}",
            flush=True,
        )
    print(f"elapsed_s={elapsed:.1f}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=int, default=96, help="total generated candidates, including baseline")
    parser.add_argument("--candidate-ids", nargs="*", default=None, help="optional subset of generated ids to evaluate")
    parser.add_argument("--rounds", type=int, default=2, help="rounds per suite per candidate")
    parser.add_argument("--suites", nargs="+", default=["library", "cloudsuite", "pressure2"],
                        help="opponent suites to screen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=0, help="parallel workers; default=min(8, cpu-1)")
    parser.add_argument("--worst-weight", type=float, default=0.20, help="objective weight assigned to worst suite")
    parser.add_argument("--top", type=int, default=20, help="number of ranked candidates to print")
    parser.add_argument("--non-novice", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not 0.0 <= args.worst_weight <= 0.8:
        raise SystemExit("--worst-weight must be between 0.0 and 0.8")

    report = run_sweep(args)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report, indent=2))
        print(f"summary -> {args.summary_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
