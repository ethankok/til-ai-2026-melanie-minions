"""Sweep fixed-map AE route profiles locally.

This is the Mac-side gate before any Workbench build/submit. It does not
change the shipped Docker image. Instead it instantiates ``AEManager`` with
different fixed-Novice-map strategy profiles and ranks them against the same
local opponent suites used by ``validate_cloud_suite.py``.

Quick smoke:

    .venv/bin/python training/ae/sweep_fixed_routes.py --profiles smoke --rounds 4

Cloudsuite-only core sweep:

    .venv/bin/python training/ae/sweep_fixed_routes.py --profiles core --rounds 8 \
      --suites cloudsuite --summary-out training/ae/data/fixed-route-suite-core.json

Promotion gate before any push/build/submit:

    .venv/bin/python training/ae/sweep_fixed_routes.py --profiles core --rounds 24 \
      --suites random library cloudsuite \
      --summary-out training/ae/data/fixed-route-suite-gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from simulate import run_simulation


# Matches the Docker image's safety baseline; profiles can override via kwargs below.
BASE_ENV = {
    "AE_USE_PLAYBOOK": "0",
    "AE_USE_OPPONENT_MODEL": "0",
    "AE_ALLY_BOMB_SAFE": "0",
    "AE_TTD_DEFENSE": "0",
}


@dataclass(frozen=True)
class Profile:
    name: str
    note: str
    kwargs: dict[str, float | int | bool]


PROFILES: list[Profile] = [
    Profile(
        name="baseline_fixed_v3",
        note="Current shipped fixed-map heuristic knobs.",
        kwargs={},
    ),
    Profile(
        name="attack_cells_close",
        note="Bias toward nearby enemy-base bombing cells.",
        kwargs={
            "fixed_macro_enabled": True,
            "fixed_center_enabled": False,
            "fixed_attack_cells_enabled": True,
            "fixed_attack_cell_radius": 5.0,
            "fixed_attack_cell_bonus": 50.0,
            "ENEMY_BASE_VALUE": 145.0,
            "DIST_PENALTY": 1.05,
            "low_ammo_enemy_base_value": 135.0,
            "low_ammo_resource_value": 20.0,
        },
    ),
    Profile(
        name="attack_cells_wide",
        note="Allow longer fixed-map base-hit routes before falling back to items.",
        kwargs={
            "fixed_macro_enabled": True,
            "fixed_center_enabled": False,
            "fixed_attack_cells_enabled": True,
            "fixed_attack_cell_radius": 10.0,
            "fixed_attack_cell_bonus": 40.0,
            "ENEMY_BASE_VALUE": 135.0,
            "DIST_PENALTY": 1.0,
            "low_ammo_enemy_base_value": 125.0,
            "low_ammo_resource_value": 25.0,
        },
    ),
    Profile(
        name="center_then_attack",
        note="Open through the dense center cluster, then prefer base-hit cells.",
        kwargs={
            "fixed_macro_enabled": True,
            "fixed_center_enabled": True,
            "fixed_attack_cells_enabled": True,
            "fixed_center_value": 105.0,
            "fixed_attack_cell_radius": 8.0,
            "fixed_attack_cell_bonus": 35.0,
            "ENEMY_BASE_VALUE": 130.0,
            "item_mission_value": 55.0,
            "item_resource_value": 20.0,
        },
    ),
    Profile(
        name="center_greedy",
        note="Very strong center/item opening; tests whether cloud rewards farming more.",
        kwargs={
            "fixed_macro_enabled": True,
            "fixed_center_enabled": True,
            "fixed_attack_cells_enabled": True,
            "fixed_center_value": 155.0,
            "fixed_attack_cell_radius": 6.0,
            "fixed_attack_cell_bonus": 20.0,
            "ENEMY_BASE_VALUE": 110.0,
            "item_mission_value": 75.0,
            "item_resource_value": 18.0,
            "item_prior_confidence": 0.85,
            "item_prior_floor": 0.35,
        },
    ),
    Profile(
        name="base_leash",
        note="Fixed-map defensive leash while preserving attack-cell pressure.",
        kwargs={
            "fixed_macro_enabled": True,
            "fixed_center_enabled": False,
            "fixed_attack_cells_enabled": True,
            "fixed_attack_cell_radius": 7.0,
            "fixed_attack_cell_bonus": 30.0,
            "BASE_DEFENSE_RADIUS": 8,
            "BASE_DEFENSE_HEALTH": 80.0,
            "base_defense_panic_value": 230.0,
            "base_defense_panic_radius": 10,
            "ENEMY_BASE_VALUE": 115.0,
            "low_ammo_enemy_base_value": 90.0,
            "low_ammo_resource_value": 40.0,
        },
    ),
    Profile(
        name="chase_home_threat",
        note="Kill/chase enemies seen near home instead of pure base racing.",
        kwargs={
            "fixed_macro_enabled": True,
            "fixed_center_enabled": False,
            "fixed_attack_cells_enabled": True,
            "fixed_attack_cell_radius": 6.0,
            "fixed_attack_cell_bonus": 25.0,
            "BASE_DEFENSE_RADIUS": 8,
            "BASE_DEFENSE_HEALTH": 70.0,
            "ENEMY_CHASE_VALUE": 35.0,
            "ENEMY_CHASE_RADIUS": 4,
            "ENEMY_BASE_VALUE": 120.0,
        },
    ),
    Profile(
        name="low_ammo_base_race",
        note="Keep racing bases even after bomb stock drops.",
        kwargs={
            "fixed_macro_enabled": True,
            "fixed_center_enabled": False,
            "fixed_attack_cells_enabled": True,
            "fixed_attack_cell_radius": 9.0,
            "fixed_attack_cell_bonus": 45.0,
            "ENEMY_BASE_VALUE": 145.0,
            "low_ammo_enemy_base_value": 160.0,
            "low_ammo_resource_value": 12.0,
            "dijkstra_no_bomb_cost": 8.0,
            "DIST_PENALTY": 0.95,
        },
    ),
    Profile(
        name="ttd_shallow_guard",
        note="Experimental visible-pressure override; included because base failures are visible-bomb heavy.",
        kwargs={
            "fixed_macro_enabled": True,
            "fixed_center_enabled": False,
            "fixed_attack_cells_enabled": True,
            "fixed_attack_cell_radius": 6.0,
            "fixed_attack_cell_bonus": 30.0,
            "ttd_defense_enabled": True,
            "ttd_defense_immediate": 0,
            "ttd_defense_damaged": 1,
            "ttd_defense_max_tti": 3,
            "BASE_DEFENSE_RADIUS": 7,
            "ENEMY_BASE_VALUE": 120.0,
        },
    ),
    Profile(
        name="farm_low_threat",
        note="Reduce path threat fear and farm fixed item priors aggressively.",
        kwargs={
            "fixed_macro_enabled": True,
            "fixed_center_enabled": True,
            "fixed_attack_cells_enabled": True,
            "fixed_center_value": 95.0,
            "fixed_attack_cell_radius": 7.0,
            "fixed_attack_cell_bonus": 25.0,
            "PATH_THREAT_PENALTY": 0.75,
            "item_mission_value": 70.0,
            "item_resource_value": 30.0,
            "item_prior_confidence": 0.9,
            "item_seen_floor": 0.5,
            "RECENT_LOCATION_PENALTY": 0.75,
        },
    ),
]

PROFILE_SETS = {
    "smoke": ["baseline_fixed_v3", "attack_cells_close", "base_leash"],
    "core": [
        "baseline_fixed_v3",
        "attack_cells_close",
        "attack_cells_wide",
        "center_then_attack",
        "center_greedy",
        "base_leash",
        "chase_home_threat",
        "low_ammo_base_race",
    ],
    "wide": [p.name for p in PROFILES],
}


def _profile_map() -> dict[str, Profile]:
    return {profile.name: profile for profile in PROFILES}


def _select_profiles(specs: list[str]) -> list[Profile]:
    profiles = _profile_map()
    selected: list[str] = []
    for spec in specs:
        if spec in PROFILE_SETS:
            selected.extend(PROFILE_SETS[spec])
        elif spec in profiles:
            selected.append(spec)
        else:
            known = ", ".join(sorted([*PROFILE_SETS, *profiles]))
            raise SystemExit(f"unknown profile/profile-set {spec!r}; known: {known}")

    deduped: list[Profile] = []
    seen: set[str] = set()
    for name in selected:
        if name not in seen:
            deduped.append(profiles[name])
            seen.add(name)
    return deduped


def _format_failures(summary: dict) -> str:
    failures = summary.get("diagnostics", {}).get("base_failure_classes", {})
    if not failures:
        return "{}"
    ordered = sorted(failures.items(), key=lambda kv: -int(kv[1]))
    return "{" + ", ".join(f"{k}:{int(v)}" for k, v in ordered[:3]) + "}"


def _format_components(summary: dict) -> str:
    components = summary.get("diagnostics", {}).get("reward_component_sum", {})
    if not components:
        return "{}"
    ordered = sorted(components.items(), key=lambda kv: -abs(float(kv[1])))
    return "{" + ", ".join(f"{k}:{float(v):+.0f}" for k, v in ordered[:4]) + "}"


def _aggregate(per_suite: list[dict]) -> dict:
    means = [float(s["mean_score"]) for s in per_suite]
    medians = [float(s["p50"]) for s in per_suite]
    return {
        "mean_of_means": sum(means) / max(1, len(means)),
        "median_of_medians": statistics.median(medians) if medians else 0.0,
        "worst_mean": min(means) if means else 0.0,
        "best_mean": max(means) if means else 0.0,
    }


def _rank_key(result: dict, primary_suite: str) -> tuple[float, float]:
    by_suite = {s["suite"]: s["summary"] for s in result["per_suite"]}
    primary = by_suite.get(primary_suite)
    primary_mean = float(primary["mean_score"]) if primary else float(result["aggregate"]["worst_mean"])
    return primary_mean, float(result["aggregate"]["worst_mean"])


def run_sweep(args: argparse.Namespace) -> dict:
    for key, value in BASE_ENV.items():
        os.environ[key] = value

    profiles = _select_profiles(args.profiles)
    started = time.monotonic()
    results: list[dict] = []

    print(
        f"fixed-route sweep: profiles={len(profiles)} rounds={args.rounds} "
        f"suites={','.join(args.suites)} seed={args.seed}",
        flush=True,
    )
    print("baseline env: " + " ".join(f"{k}={v}" for k, v in sorted(BASE_ENV.items())), flush=True)

    for idx, profile in enumerate(profiles, start=1):
        print(f"\n=== [{idx}/{len(profiles)}] {profile.name} ===", flush=True)
        print(f"note: {profile.note}", flush=True)
        if profile.kwargs:
            print("kwargs: " + json.dumps(profile.kwargs, sort_keys=True), flush=True)

        per_suite: list[dict] = []
        for suite in args.suites:
            print(f"\n-- {profile.name} / {suite} --", flush=True)
            out = run_simulation(
                rounds=args.rounds,
                opponents_spec=suite,
                our_name=args.our,
                log_traj=False,
                seed_start=args.seed,
                novice=not args.non_novice,
                our_kwargs=dict(profile.kwargs),
            )
            summary = out["summary"]
            per_suite.append({"suite": suite, "summary": summary})
            diagnostics = summary.get("diagnostics", {})
            print(
                f"{suite}: mean={summary['mean_score']:.4f} p50={summary['p50']:.4f} "
                f"min={summary['min_score']:.4f} max={summary['max_score']:.4f} "
                f"base={diagnostics.get('mean_final_base_health', 0.0):.1f} "
                f"bombs={diagnostics.get('mean_bombs_placed', 0.0):.1f} "
                f"ttd={diagnostics.get('mean_base_pressure_overrides', 0.0):.1f} "
                f"fail={_format_failures(summary)} comp={_format_components(summary)}",
                flush=True,
            )

        aggregate = _aggregate([entry["summary"] for entry in per_suite])
        result = {
            "profile": profile.name,
            "note": profile.note,
            "kwargs": profile.kwargs,
            "aggregate": aggregate,
            "per_suite": per_suite,
        }
        results.append(result)
        print(
            f"\n{profile.name}: mean_of_means={aggregate['mean_of_means']:.4f} "
            f"worst={aggregate['worst_mean']:.4f}",
            flush=True,
        )

    ranked = sorted(results, key=lambda r: _rank_key(r, args.primary_suite), reverse=True)
    elapsed = time.monotonic() - started
    report = {
        "rounds_per_suite": args.rounds,
        "suites": args.suites,
        "primary_suite": args.primary_suite,
        "our_agent": args.our,
        "novice": not args.non_novice,
        "elapsed_s": elapsed,
        "results": results,
        "ranked_profiles": [r["profile"] for r in ranked],
    }

    print("\n=== leaderboard ===", flush=True)
    print(f"{'rank':>4}  {'profile':<22}  {args.primary_suite:<10}  {'worst':<8}  {'mean':<8}", flush=True)
    for rank, result in enumerate(ranked, start=1):
        by_suite = {s["suite"]: s["summary"] for s in result["per_suite"]}
        primary = by_suite.get(args.primary_suite)
        primary_mean = float(primary["mean_score"]) if primary else float(result["aggregate"]["worst_mean"])
        print(
            f"{rank:>4}  {result['profile']:<22}  {primary_mean:<10.4f}  "
            f"{result['aggregate']['worst_mean']:<8.4f}  {result['aggregate']['mean_of_means']:<8.4f}",
            flush=True,
        )
    print(f"elapsed_s={elapsed:.1f}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", nargs="+", default=["core"], help="profile sets or names; use --list to inspect")
    parser.add_argument("--rounds", type=int, default=6, help="rounds per suite per profile")
    parser.add_argument("--suites", nargs="+", default=["cloudsuite"], help="opponent suites to run")
    parser.add_argument("--primary-suite", default="cloudsuite", help="suite used for leaderboard ranking")
    parser.add_argument("--our", choices=["heuristic", "hybrid"], default="heuristic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-novice", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="print available profiles and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("profile sets:")
        for name, members in PROFILE_SETS.items():
            print(f"  {name}: {', '.join(members)}")
        print("\nprofiles:")
        for profile in PROFILES:
            print(f"  {profile.name}: {profile.note}")
        return 0

    report = run_sweep(args)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report, indent=2))
        print(f"summary -> {args.summary_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
