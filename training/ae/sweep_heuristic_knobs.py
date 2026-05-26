"""Screen current AEManager heuristic knobs against non-random local suites.

This is the Mac-first knob sweeper for the current heuristic runtime. It does
not use stale constructor kwargs; each candidate is expressed as the exact
``AE_*`` environment variables that ``ae/src/ae_manager.py`` reads on manager
construction. The baseline candidate is explicit and matches the current fixed
Novice-map defaults after AEManager's fixed-map auto-promotion.

Typical flow:

    # Tiny sanity check
    .venv/bin/python training/ae/sweep_heuristic_knobs.py \
      --candidates 8 --rounds 1 --suites cloudsuite pressure2 \
      --jobs 4 --summary-out training/ae/data/knob-smoke.json

    # Broad first screen. Includes baseline plus sampled candidates.
    .venv/bin/python training/ae/sweep_heuristic_knobs.py \
      --candidates 224 --rounds 2 --suites cloudsuite pressure2 \
      --jobs 8 --summary-out training/ae/data/knob-screen-224.json

    # Focused interaction screen around the strongest broad-screen region.
    .venv/bin/python training/ae/sweep_heuristic_knobs.py \
      --mode focused --candidates 288 --rounds 2 --suites cloudsuite pressure2 \
      --jobs 8 --summary-out training/ae/data/knob-focused-288.json

    # Promotion gate for the top few candidate ids from the screen.
    .venv/bin/python training/ae/sweep_heuristic_knobs.py \
      --candidate-ids baseline rand_0042 rand_0107 --rounds 24 \
      --suites library cloudsuite pressure2 mixed \
      --summary-out training/ae/data/knob-gate.json

    # Bridge sweep between pressure-strong and cloudsuite-strong families.
    .venv/bin/python training/ae/sweep_heuristic_knobs.py \
      --mode bridge --candidates 240 --rounds 3 --suites cloudsuite pressure2 \
      --jobs 8 --summary-out training/ae/data/knob-bridge-240.json
"""

from __future__ import annotations

# Pin PYTHONHASHSEED=0 before any other import so AEManager's hash-dependent
# branches are deterministic in this process AND in ProcessPoolExecutor
# children (which inherit os.environ at spawn time).
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


# Explicit current fixed-Novice baseline. AEManager would auto-promote
# AE_ENEMY_BASE_VALUE from 80 to 130 and AE_BASE_DEFENSE_RADIUS from 4 to 6 on
# the fixed map when the env vars are absent; setting them here makes every
# candidate fully comparable and keeps worker processes stateless.
BASELINE_ENV = {
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


KNOB_GRID = {
    "AE_ENEMY_BASE_VALUE": ["90", "110", "130", "150", "170", "190"],
    "AE_DIST_PENALTY": ["0.75", "0.95", "1.15", "1.35", "1.60"],
    "AE_DIJKSTRA_BOMB_COST": ["3", "5", "8", "12", "20"],
    "AE_ITEM_MISSION_VALUE": ["35", "50", "65", "80"],
    "AE_ITEM_RESOURCE_VALUE": ["15", "25", "35", "45"],
    "AE_ITEM_RECON_VALUE": ["5", "10", "15"],
    "AE_BASE_DEFENSE_HEALTH": ["40", "60", "75", "90"],
    "AE_BASE_DEFENSE_RADIUS": ["4", "6", "8", "10"],
    "AE_BASE_DEFENSE_VALUE": ["30", "60", "90", "120", "180"],
    "AE_BASE_DEFENSE_EMERGENCY_VALUE": ["100", "150", "220", "300"],
    "AE_PATH_THREAT_PENALTY": ["0.75", "1.5", "2", "3", "4"],
    "AE_CELL_THREAT_PENALTY": ["2", "5", "8", "12"],
    "AE_ENEMY_CHASE_VALUE": ["0", "10", "20", "35"],
    "AE_ENEMY_CHASE_RADIUS": ["3", "4", "6", "8"],
    "AE_TIER1_DEFENSE": ["0", "1"],
    "AE_TIER1_SHARED_CREDIT": ["0", "1"],
}


# Focused combination grid around the strongest region from the first broad
# screen. Keep this deliberately smaller than the broad grid: it tests
# interactions among knobs that showed signal without dragging every item /
# chase / shared-credit setting back into a noisy full factorial.
FOCUSED_GRID = {
    "AE_ENEMY_BASE_VALUE": ["130", "150"],
    "AE_DIST_PENALTY": ["0.95", "1.15", "1.35", "1.60"],
    "AE_DIJKSTRA_BOMB_COST": ["8", "10", "12"],
    "AE_BASE_DEFENSE_RADIUS": ["6", "8", "10"],
    "AE_PATH_THREAT_PENALTY": ["2", "3", "4"],
    "AE_CELL_THREAT_PENALTY": ["5", "8", "12"],
    "AE_TIER1_DEFENSE": ["0", "1"],
}

# Rescue grid between the best focused-gate tradeoffs:
# - focus_0124 improved pressure but hurt cloudsuite.
# - focus_0266 improved cloudsuite but hurt pressure2.
# The goal here is not another broad sweep; it is to find a narrow bridge that
# keeps cell threat high while backing off the pressure/cloudsuite tradeoff.
BRIDGE_GRID = {
    "AE_ENEMY_BASE_VALUE": ["130", "150"],
    "AE_DIST_PENALTY": ["1.15", "1.25", "1.35", "1.45", "1.60"],
    "AE_DIJKSTRA_BOMB_COST": ["5", "8", "10"],
    "AE_BASE_DEFENSE_RADIUS": ["6", "8"],
    "AE_PATH_THREAT_PENALTY": ["2", "3", "4"],
    "AE_CELL_THREAT_PENALTY": ["5", "8", "10", "12"],
    "AE_TIER1_DEFENSE": ["0", "1"],
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
    source: str


def _with_overrides(overrides: dict[str, str]) -> dict[str, str]:
    env = dict(BASELINE_ENV)
    env.update({key: str(value) for key, value in overrides.items()})
    return env


def _focused_anchor_candidates() -> list[Candidate]:
    """Carry forward the strongest broad-screen regions into focused mode."""

    return [
        Candidate(
            "anchor_dijkstra_8",
            _with_overrides({"AE_DIJKSTRA_BOMB_COST": "8"}),
            "focused_anchor",
        ),
        Candidate(
            "anchor_rand_0023",
            _with_overrides({
                "AE_ENEMY_BASE_VALUE": "150",
                "AE_DIST_PENALTY": "0.95",
                "AE_DIJKSTRA_BOMB_COST": "8",
                "AE_ITEM_RESOURCE_VALUE": "15",
                "AE_BASE_DEFENSE_HEALTH": "40",
                "AE_BASE_DEFENSE_RADIUS": "10",
                "AE_PATH_THREAT_PENALTY": "3",
                "AE_CELL_THREAT_PENALTY": "8",
                "AE_ENEMY_CHASE_VALUE": "20",
                "AE_TIER1_DEFENSE": "1",
            }),
            "focused_anchor",
        ),
        Candidate(
            "anchor_rand_0066",
            _with_overrides({
                "AE_DIST_PENALTY": "1.60",
                "AE_DIJKSTRA_BOMB_COST": "12",
                "AE_ITEM_MISSION_VALUE": "65",
                "AE_ITEM_RESOURCE_VALUE": "15",
                "AE_ITEM_RECON_VALUE": "15",
                "AE_BASE_DEFENSE_RADIUS": "10",
                "AE_BASE_DEFENSE_VALUE": "30",
                "AE_BASE_DEFENSE_EMERGENCY_VALUE": "220",
                "AE_PATH_THREAT_PENALTY": "4",
                "AE_ENEMY_CHASE_RADIUS": "8",
            }),
            "focused_anchor",
        ),
        Candidate(
            "anchor_rand_0128",
            _with_overrides({
                "AE_DIJKSTRA_BOMB_COST": "8",
                "AE_ITEM_MISSION_VALUE": "35",
                "AE_ITEM_RESOURCE_VALUE": "35",
                "AE_ITEM_RECON_VALUE": "15",
                "AE_BASE_DEFENSE_HEALTH": "75",
                "AE_BASE_DEFENSE_VALUE": "30",
                "AE_BASE_DEFENSE_EMERGENCY_VALUE": "100",
                "AE_PATH_THREAT_PENALTY": "3",
                "AE_CELL_THREAT_PENALTY": "8",
                "AE_ENEMY_CHASE_VALUE": "10",
                "AE_ENEMY_CHASE_RADIUS": "8",
            }),
            "focused_anchor",
        ),
        Candidate(
            "anchor_rand_0114",
            _with_overrides({
                "AE_ENEMY_BASE_VALUE": "150",
                "AE_DIST_PENALTY": "1.60",
                "AE_DIJKSTRA_BOMB_COST": "12",
                "AE_ITEM_RECON_VALUE": "15",
                "AE_BASE_DEFENSE_HEALTH": "75",
                "AE_BASE_DEFENSE_RADIUS": "4",
                "AE_BASE_DEFENSE_VALUE": "30",
                "AE_BASE_DEFENSE_EMERGENCY_VALUE": "220",
                "AE_PATH_THREAT_PENALTY": "3",
                "AE_CELL_THREAT_PENALTY": "12",
                "AE_ENEMY_CHASE_VALUE": "20",
                "AE_ENEMY_CHASE_RADIUS": "3",
            }),
            "focused_anchor",
        ),
        Candidate(
            "anchor_rand_0154",
            _with_overrides({
                "AE_ENEMY_BASE_VALUE": "150",
                "AE_DIST_PENALTY": "0.75",
                "AE_DIJKSTRA_BOMB_COST": "8",
                "AE_ITEM_MISSION_VALUE": "65",
                "AE_ITEM_RECON_VALUE": "15",
                "AE_BASE_DEFENSE_HEALTH": "90",
                "AE_BASE_DEFENSE_VALUE": "120",
                "AE_BASE_DEFENSE_EMERGENCY_VALUE": "100",
                "AE_PATH_THREAT_PENALTY": "1.5",
                "AE_CELL_THREAT_PENALTY": "8",
                "AE_ENEMY_CHASE_RADIUS": "8",
                "AE_TIER1_DEFENSE": "1",
            }),
            "focused_anchor",
        ),
    ]


def _bridge_anchor_candidates() -> list[Candidate]:
    """Known focused-gate tradeoffs plus small manual midpoint probes."""

    pressure_family = {
        "AE_ENEMY_BASE_VALUE": "150",
        "AE_DIST_PENALTY": "1.35",
        "AE_DIJKSTRA_BOMB_COST": "8",
        "AE_BASE_DEFENSE_RADIUS": "6",
        "AE_PATH_THREAT_PENALTY": "4",
        "AE_CELL_THREAT_PENALTY": "12",
        "AE_TIER1_DEFENSE": "1",
    }
    cloud_family = {
        "AE_ENEMY_BASE_VALUE": "130",
        "AE_DIST_PENALTY": "1.60",
        "AE_DIJKSTRA_BOMB_COST": "10",
        "AE_BASE_DEFENSE_RADIUS": "8",
        "AE_PATH_THREAT_PENALTY": "2",
        "AE_CELL_THREAT_PENALTY": "12",
        "AE_TIER1_DEFENSE": "0",
    }
    midpoint_soft = {
        "AE_ENEMY_BASE_VALUE": "130",
        "AE_DIST_PENALTY": "1.35",
        "AE_DIJKSTRA_BOMB_COST": "8",
        "AE_BASE_DEFENSE_RADIUS": "6",
        "AE_PATH_THREAT_PENALTY": "3",
        "AE_CELL_THREAT_PENALTY": "10",
        "AE_TIER1_DEFENSE": "0",
    }
    midpoint_defense = {
        "AE_ENEMY_BASE_VALUE": "150",
        "AE_DIST_PENALTY": "1.45",
        "AE_DIJKSTRA_BOMB_COST": "8",
        "AE_BASE_DEFENSE_RADIUS": "8",
        "AE_PATH_THREAT_PENALTY": "3",
        "AE_CELL_THREAT_PENALTY": "10",
        "AE_TIER1_DEFENSE": "1",
    }
    return [
        Candidate("bridge_pressure_family", _with_overrides(pressure_family), "bridge_anchor"),
        Candidate("bridge_cloud_family", _with_overrides(cloud_family), "bridge_anchor"),
        Candidate("bridge_midpoint_soft", _with_overrides(midpoint_soft), "bridge_anchor"),
        Candidate("bridge_midpoint_defense", _with_overrides(midpoint_defense), "bridge_anchor"),
    ]


def _env_signature(env: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, str(v)) for k, v in env.items()))


def _one_factor_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for knob, values in KNOB_GRID.items():
        baseline_value = BASELINE_ENV[knob]
        for value in values:
            if value == baseline_value:
                continue
            env = dict(BASELINE_ENV)
            env[knob] = value
            safe_knob = knob.removeprefix("AE_").lower()
            candidates.append(Candidate(
                candidate_id=f"one_{safe_knob}_{value.replace('.', 'p')}",
                env=env,
                source="one_factor",
            ))
    return candidates


def _sample_candidate(rng: random.Random, idx: int) -> Candidate:
    env = dict(BASELINE_ENV)

    # Random search, but with a little structure so most candidates stay
    # interpretable: offense/items/threat/defense are all sampled, while risky
    # historical toggles are rare unless the RNG deliberately flips them.
    for knob, values in KNOB_GRID.items():
        if knob == "AE_TIER1_SHARED_CREDIT":
            env[knob] = "1" if rng.random() < 0.12 else "0"
        elif knob == "AE_TIER1_DEFENSE":
            env[knob] = "1" if rng.random() < 0.35 else "0"
        else:
            env[knob] = rng.choice(values)

    return Candidate(candidate_id=f"rand_{idx:04d}", env=env, source="random")


def _append_unique(candidates: list[Candidate], seen: set[tuple[tuple[str, str], ...]],
                   candidate: Candidate) -> bool:
    sig = _env_signature(candidate.env)
    if sig in seen:
        return False
    candidates.append(candidate)
    seen.add(sig)
    return True


def generate_broad_candidates(total: int, seed: int) -> list[Candidate]:
    if total < 1:
        raise ValueError("--candidates must be >= 1")

    rng = random.Random(seed)
    candidates = [Candidate("baseline", dict(BASELINE_ENV), "baseline")]
    seen = {_env_signature(BASELINE_ENV)}

    for candidate in _one_factor_candidates():
        if len(candidates) >= total:
            return candidates
        _append_unique(candidates, seen, candidate)

    idx = 0
    while len(candidates) < total:
        candidate = _sample_candidate(rng, idx)
        idx += 1
        _append_unique(candidates, seen, candidate)
    return candidates


def generate_focused_candidates(total: int, seed: int) -> list[Candidate]:
    if total < 1:
        raise ValueError("--candidates must be >= 1")

    rng = random.Random(seed)
    candidates = [Candidate("baseline", dict(BASELINE_ENV), "baseline")]
    seen = {_env_signature(BASELINE_ENV)}

    for candidate in _focused_anchor_candidates():
        if len(candidates) >= total:
            return candidates
        _append_unique(candidates, seen, candidate)

    keys = list(FOCUSED_GRID)
    combos = list(product(*(FOCUSED_GRID[key] for key in keys)))
    rng.shuffle(combos)
    for idx, values in enumerate(combos):
        if len(candidates) >= total:
            break
        env = dict(BASELINE_ENV)
        env.update({key: value for key, value in zip(keys, values)})
        _append_unique(candidates, seen, Candidate(f"focus_{idx:04d}", env, "focused_grid"))
    return candidates


def generate_bridge_candidates(total: int, seed: int) -> list[Candidate]:
    if total < 1:
        raise ValueError("--candidates must be >= 1")

    rng = random.Random(seed)
    candidates = [Candidate("baseline", dict(BASELINE_ENV), "baseline")]
    seen = {_env_signature(BASELINE_ENV)}

    for candidate in _bridge_anchor_candidates():
        if len(candidates) >= total:
            return candidates
        _append_unique(candidates, seen, candidate)

    keys = list(BRIDGE_GRID)
    combos = list(product(*(BRIDGE_GRID[key] for key in keys)))
    rng.shuffle(combos)
    for idx, values in enumerate(combos):
        if len(candidates) >= total:
            break
        env = dict(BASELINE_ENV)
        env.update({key: value for key, value in zip(keys, values)})
        _append_unique(candidates, seen, Candidate(f"bridge_{idx:04d}", env, "bridge_grid"))
    return candidates


def generate_candidates(total: int, seed: int, mode: str) -> list[Candidate]:
    if mode == "broad":
        return generate_broad_candidates(total, seed)
    if mode == "focused":
        return generate_focused_candidates(total, seed)
    if mode == "bridge":
        return generate_bridge_candidates(total, seed)
    raise ValueError(f"unknown mode: {mode!r}")


def _select_candidates(all_candidates: list[Candidate], ids: list[str] | None) -> list[Candidate]:
    if not ids:
        return all_candidates
    by_id = {candidate.candidate_id: candidate for candidate in all_candidates}
    missing = [candidate_id for candidate_id in ids if candidate_id not in by_id]
    if missing:
        known = ", ".join(sorted(by_id)[:30])
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
                    our_name="heuristic",
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
        aggregate = {
            "mean_of_means": _mean(suite_means),
            "median_of_medians": statistics.median(float(per_suite[suite]["p50"]) for suite in suites),
            "worst_mean": min(suite_means),
            "best_mean": max(suite_means),
            "objective": _objective(per_suite, suites, worst_weight),
        }
        return {
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "env": candidate.env,
            "aggregate": aggregate,
            "per_suite": per_suite,
            "ok": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "env": candidate.env,
            "ok": False,
            "error": repr(exc),
            "captured_tail": captured.getvalue()[-4000:],
        }
    finally:
        _restore_env(old_env)


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


def _reject_reasons(result: dict[str, Any]) -> list[str]:
    delta = result.get("delta_vs_baseline", {})
    suite_delta = delta.get("suites", {})
    reasons = []
    if suite_delta.get("cloudsuite", 0.0) < -0.03:
        reasons.append("cloudsuite_drop_gt_0.03")
    if float(delta.get("worst_mean", 0.0)) < -0.03:
        reasons.append("worst_suite_drop_gt_0.03")
    if float(delta.get("objective", 0.0)) < 0.0:
        reasons.append("objective_below_baseline")
    return reasons


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
    all_candidates = generate_candidates(args.candidates, args.seed, args.mode)
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
        f"heuristic knob sweep: mode={args.mode} candidates={len(candidates)} rounds={args.rounds} "
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
                    f"[{completed:>4}/{len(candidates)}] {result['candidate_id']:<26} "
                    f"obj={float(result['aggregate']['objective']):.4f} "
                    f"worst={float(result['aggregate']['worst_mean']):.4f} "
                    f"{_format_suite_scores(result, args.suites)}",
                    flush=True,
                )
            else:
                print(f"[{completed:>4}/{len(candidates)}] {result['candidate_id']:<26} ERROR {result['error']}", flush=True)

    _add_baseline_deltas(results, args.suites)
    ranked = _rank(results)
    elapsed = time.monotonic() - started
    report = {
        "rounds_per_suite": args.rounds,
        "suites": args.suites,
        "candidate_count": len(candidates),
        "mode": args.mode,
        "seed": args.seed,
        "novice": novice,
        "jobs": jobs,
        "worst_weight": args.worst_weight,
        "elapsed_s": elapsed,
        "baseline_env": BASELINE_ENV,
        "safety_env": SAFETY_ENV,
        "results": sorted(results, key=lambda r: r["candidate_id"]),
        "ranked_candidate_ids": [r["candidate_id"] for r in ranked],
    }

    print("\n=== top candidates ===", flush=True)
    print(f"{'rank':>4}  {'candidate':<26}  {'obj':<8}  {'delta':<8}  {'worst':<8}  suites", flush=True)
    for i, result in enumerate(ranked[: args.top], start=1):
        delta = result.get("delta_vs_baseline", {}).get("objective", 0.0)
        rejected = ",".join(result.get("reject_reasons", []))
        suffix = f" reject={rejected}" if rejected else ""
        print(
            f"{i:>4}  {result['candidate_id']:<26}  "
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
    parser.add_argument("--mode", choices=["broad", "focused", "bridge"], default="broad",
                        help="candidate generator: broad, focused interaction grid, or bridge rescue grid")
    parser.add_argument("--candidates", type=int, default=224, help="total generated candidates, including baseline")
    parser.add_argument("--candidate-ids", nargs="*", default=None, help="optional subset of generated ids to evaluate")
    parser.add_argument("--rounds", type=int, default=2, help="rounds per suite per candidate")
    parser.add_argument("--suites", nargs="+", default=["cloudsuite", "pressure2"], help="opponent suites to screen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=0, help="parallel workers; default=min(8, cpu-1)")
    parser.add_argument("--worst-weight", type=float, default=0.15, help="objective weight assigned to worst suite")
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
