"""Head-to-head melee eval + finals-aligned (relative-rank) selection gate (AE Semis).

This is the realistic Semis gate: it scores candidate agents in the 6-team melee
brackets (``opponents.MELEE_BRACKETS``), against the FOREIGN (non-mirror) opponent
pool, and ranks by **relative placement** — the shape finals actually scores. (Finals
AE = teams sorted by ``ae_reward * mission_multiplier``; placement, not absolute
reward, is what's paid. See design spec 2026-06-07-ae-finals-aligned-eval-redesign,
private archive.)

Determinism: AEManager has hash-order-dependent branches, so — exactly like
``multi_seed_eval.py`` — we spawn a fresh interpreter per (hash_seed, sim_seed)
with ``PYTHONHASHSEED`` pinned, run ONE bracket per worker, and aggregate across
runs. One bracket per subprocess also isolates the per-process model cache
(``policy_manager._MODEL_CACHE``) to a single candidate + that bracket's
opponents.

Promotion rule (2026-06-09 finals-aligned overlay — placement is PRIMARY, raw_ae
is the finals-proportional DISCRIMINATOR, absolute reward is a non-crater FLOOR).
The gate applies the finals ``raw_ae * mission_multiplier`` shape via an opponent
mission-multiplier SWEEP (``--opp-mults``, our mult fixed at ``--our-mult``): a
stronger field cannot improve our placement, so placement saturates into a floor
and raw_ae (our agent's cumulative reward; proportional to final score since our
mult is fixed) carries the discrimination. A candidate is promotable over the
incumbent only if ALL hold:
  1. ``placement_robust_ok`` — ``worst_robust_placement`` (worst over brackets ×
     opp-mults) is better-or-equal to the incumbent's. PRIMARY selector.
  2. ``rawae_floor_ok`` — raw_ae has not CRATERED vs the incumbent
     (``>= RAWAE_FLOOR_FRAC * inc``). The DISCRIMINATOR: a real raw_ae gain (paired
     effect + Probability-of-Improvement on raw_ae) breaks placement ties.
  3. ``margin_noncrater_ok`` — worst-bracket margin has not cratered vs the
     incumbent (``>= inc.min_margin - MARGIN_SLACK``); a non-crater FLOOR.
  4. ``score_noncrater_ok`` — ``semis_mixed`` absolute reward has not cratered
     (``>= SCORE_FLOOR_FRAC * inc``); a non-crater FLOOR, not a selector. (Cloud
     single-agent reward is likewise a functionality check, never a revert trigger.)
  5. ``axis_ok`` (only with ``--target-axis {mission,base_defense,opening}``) — the
     per-change paired effect + PoI floor on that axis: the targeted axis must
     actually move (paired delta >= ``MIN_EFFECT_AXIS`` with PoI Wilson lower bound
     > 0.5). Gates each planned downstream change on its own axis.
  6. ``gap_not_widening_ok`` (only with ``--heldout``) — the train-vs-heldout
     composition placement GAP does not widen vs the incumbent (overfit alarm).

Thresholds are env-overridable: AE_EVAL_OUR_MULT (0.93), AE_EVAL_OPP_MULTS
("0.24,0.7"), AE_GATE_RAWAE_FLOOR_FRAC (0.75), AE_GATE_MIN_EFFECT_AXIS (0.0),
AE_GATE_SCORE_FLOOR_FRAC (0.85), AE_GATE_MARGIN_SLACK_FRAC (0.5),
AE_GATE_MIN_EFFECT_RANK (0.3), AE_GATE_GAP_TOL (0.5).

Usage
-----
    # Re-rank the default candidates (incumbent + heuristic + raw policy):
    .venv/bin/python training/ae/melee_eval.py --rounds 12 \\
        --hash-seeds 0 1 2 --sim-seeds 42 137 \\
        --summary-out training/ae/data/melee-rerank.json

    # Add Pandemonium / confpol-native ladder rungs (each a confpol wrapper over
    # a different raw-policy checkpoint):
    .venv/bin/python training/ae/melee_eval.py \\
        --confpol-ckpt native-u200=training/ae/checkpoints/confpol-native-u200.pt \\
        --confpol-ckpt native-u360=training/ae/checkpoints/confpol-native-u360.pt
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
for _p in (str(THIS_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opponents import MELEE_BRACKETS, HELDOUT_COMPOSITIONS  # noqa: E402

U860 = str((THIS_DIR / "checkpoints" / "pandemonium-v1-best-u860.pt").resolve())

# --- Finals-aligned gate thresholds (2026-06-07 redesign; env-overridable) -------
# Off-construct (absolute-reward) signals are non-crater FLOORS, not selectors; the
# placement gain must clear a real effect size and a Probability-of-Improvement bar.
SCORE_FLOOR_FRAC = float(os.environ.get("AE_GATE_SCORE_FLOOR_FRAC", "0.85"))
MARGIN_SLACK_FRAC = float(os.environ.get("AE_GATE_MARGIN_SLACK_FRAC", "0.5"))
MIN_EFFECT_RANK = float(os.environ.get("AE_GATE_MIN_EFFECT_RANK", "0.3"))
GAP_TOL = float(os.environ.get("AE_GATE_GAP_TOL", "0.5"))
OUR_MULT = float(os.environ.get("AE_EVAL_OUR_MULT", "0.93"))
OPP_MULTS = [float(x) for x in os.environ.get("AE_EVAL_OPP_MULTS", "0.24,0.7").split(",")]
RAWAE_FLOOR_FRAC = float(os.environ.get("AE_GATE_RAWAE_FLOOR_FRAC", "0.75"))
MIN_EFFECT_AXIS = float(os.environ.get("AE_GATE_MIN_EFFECT_AXIS", "0.0"))


def _cbomb7_env() -> dict[str, str]:
    from foreign_opponents import CBOMB7_ENV
    return dict(CBOMB7_ENV)


# label=path[@eps=E,floor=F,ovr=O] -> (label, env). The optional @-suffix sets the
# confpol gate knobs for THIS candidate so one sweep registers the same checkpoint
# at several consult thresholds. No @-suffix == checkpoint only (old behavior).
_GATE_KEYS = {
    "eps": "AE_CONFPOL_MARGIN_EPSILON",
    "floor": "AE_CONFPOL_TOP_FLOOR",
    "ovr": "AE_CONFPOL_OVERRIDE_TARGET_NONE",
    "override": "AE_CONFPOL_OVERRIDE_TARGET_NONE",
    "rollback": "AE_CONFPOL_ROLLBACK_PHANTOM_BOMB",
    "phantom": "AE_CONFPOL_ROLLBACK_PHANTOM_BOMB",
    "detonate": "AE_BOMB_DETONATE_STEPS",
    # planner item/base weights (deploy: mission=80, resource=40, base=100) — lets a
    # sweep pin the full deploy profile per candidate and vary one weight (e.g. farming).
    "mission": "AE_ITEM_MISSION_VALUE",
    "resource": "AE_ITEM_RESOURCE_VALUE",
    "recon": "AE_ITEM_RECON_VALUE",
    "base": "AE_ENEMY_BASE_VALUE",
    # deploy-faithful env + off-by-default behavioural flags under test.
    "contention": "AE_CONTENTION",
    "time_danger": "AE_TIME_DANGER",
    "no_self_damage": "AE_NO_SELF_DAMAGE",
    "basekill_noescape": "AE_BASEKILL_NOESCAPE",
}


def _parse_confpol_spec(spec_str: str) -> tuple[str, dict[str, str]]:
    label, _, rest = spec_str.partition("=")
    path_part, _, gate_part = rest.partition("@")
    env: dict[str, str] = {"AE_POLICY_CHECKPOINT": str(Path(path_part).resolve())}
    for kv in filter(None, gate_part.split(",")):
        key, _, val = kv.partition("=")
        env_name = _GATE_KEYS.get(key.strip().lower())
        if env_name is None:
            raise ValueError(f"unknown gate knob {key!r} in {spec_str!r}")
        env[env_name] = val.strip()
    return label, env


# label -> {"our": <simulate _make_our_agent name>, "env": {extra env vars}}
def _base_registry() -> dict[str, dict]:
    return {
        "confpol-u860":     {"our": "confidence_policy_hybrid", "env": {"AE_POLICY_CHECKPOINT": U860}},
        "heuristic-cbomb7": {"our": "heuristic",                "env": _cbomb7_env()},
        "self-policy-u860": {"our": "policy",                   "env": {"AE_POLICY_CHECKPOINT": U860}},
    }


DEFAULT_CANDIDATES = ["confpol-u860", "heuristic-cbomb7", "self-policy-u860"]
DEFAULT_INCUMBENT = "confpol-u860"
RESULT_SENTINEL = "MELEE_RESULT "


# ---------------------------------------------------------------------------
# Worker: run ONE (candidate, bracket, seed) and emit a one-line JSON summary.
# ---------------------------------------------------------------------------
def _compute_worker_payload(summary: dict, suite: str, our: str, seed: int) -> dict:
    """Build the worker result payload (paired-friendly) from a run_simulation
    summary, applying the multiplier overlay and per-axis aggregation."""
    pr = summary["per_round"]
    usid = pr["us_agent_id"]
    rounds = max(1, len(pr["cumulative_all"]))
    wplace: dict[str, float] = {}
    for om in OPP_MULTS:
        places = [_weighted_placement(ca, usid, OUR_MULT, om) for ca in pr["cumulative_all"]]
        wplace[f"{om:g}"] = float(sum(places) / rounds) if places else 0.0
    raw_ae = float(sum(ca[usid] for ca in pr["cumulative_all"]) / rounds)
    axes = [_axis_totals(c) for c in pr["reward_components"]]
    mission_axis = float(sum(a["mission_axis"] for a in axes) / rounds) if axes else 0.0
    base_defense_axis = float(sum(a["base_defense_axis"] for a in axes) / rounds) if axes else 0.0
    opening_axis = float(sum(pr["opening_reward"]) / rounds) if pr["opening_reward"] else 0.0
    return {
        "suite": suite, "our": our, "seed": seed,
        "mean_score": float(summary["mean_score"]),
        "mean_placement": float(summary["mean_placement"]),
        "mean_margin": float(summary["mean_margin"]),
        "win_rate": float(summary["win_rate"]),
        "placement_hist": summary["placement_hist"],
        "weighted_placement": wplace,
        "raw_ae": raw_ae,
        "mission_axis": mission_axis,
        "base_defense_axis": base_defense_axis,
        "opening_axis": opening_axis,
    }


def _run_worker(args: argparse.Namespace) -> int:
    from simulate import run_simulation
    out = run_simulation(
        rounds=args.rounds,
        opponents_spec=args.suite,
        our_name=args.our,
        log_traj=False,
        seed_start=args.seed,
        novice=not args.non_novice,
        us_slot=args.us_slot,
    )
    s = out["summary"]
    payload = _compute_worker_payload(s, args.suite, args.our, args.seed)
    print(RESULT_SENTINEL + json.dumps(payload), flush=True)
    return 0


def _spawn_worker(our: str, suite: str, hash_seed: int, sim_seed: int,
                  rounds: int, extra_env: dict[str, str], non_novice: bool,
                  us_slot: int = 0) -> dict:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(hash_seed)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.update(extra_env)
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--worker", "--our", our, "--suite", suite,
        "--seed", str(sim_seed), "--rounds", str(rounds),
        "--us-slot", str(us_slot),
    ]
    if non_novice:
        cmd.append("--non-novice")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"melee worker failed (our={our}, suite={suite}, hash={hash_seed}, "
            f"sim={sim_seed}, rc={proc.returncode}):\n{(proc.stderr or '')[-2500:]}"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            return json.loads(line[len(RESULT_SENTINEL):])
    raise RuntimeError(
        f"melee worker produced no result line (our={our}, suite={suite}):\n"
        f"{(proc.stdout or '')[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------
def _agg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


# ---------------------------------------------------------------------------
# Reform 3 — paired placement statistics (is the gain real, or noise?)
# ---------------------------------------------------------------------------
def _wilson_lower(successes: float, n: int, z: float = 1.96) -> float:
    """Wilson score-interval lower bound for a binomial proportion.

    Deterministic (no bootstrap RNG) and conservative on the bounded-ordinal
    placement data, where Gaussian CIs would understate width. Ties count as 0.5
    of a success, so ``successes`` may be fractional. Returns 0.0 for n<=0 and
    clamps the lower bound at 0.
    """
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(max(0.0, phat * (1.0 - phat) / n + z * z / (4 * n * n)))
    return max(0.0, (center - margin) / denom)


def _paired_placement_stats(cand: dict, inc: dict) -> dict:
    """Pair candidate vs incumbent per (suite, hash, sim) and quantify the edge.

    Both ran identical (hash_seed, sim_seed) over identical brackets, so the runs
    are paired (common random numbers). ``delta = inc_place - cand_place`` is
    positive when the candidate places BETTER (lower placement is better).
    """
    inc_rows: dict[tuple, float] = {}
    for suite, b in inc.get("per_bracket", {}).items():
        for r in b.get("run_rows", []):
            inc_rows[(suite, r["hash"], r["sim"])] = float(r["mean_placement"])
    deltas: list[float] = []
    wins = ties = losses = 0
    for suite, b in cand.get("per_bracket", {}).items():
        for r in b.get("run_rows", []):
            key = (suite, r["hash"], r["sim"])
            if key not in inc_rows:
                continue
            cp = float(r["mean_placement"])
            ip = inc_rows[key]
            deltas.append(ip - cp)
            if cp < ip - 1e-9:
                wins += 1
            elif cp > ip + 1e-9:
                losses += 1
            else:
                ties += 1
    n = len(deltas)
    succ = wins + 0.5 * ties
    return {
        "n_pairs": n,
        "mean_delta": (sum(deltas) / n) if n else 0.0,
        "poi": (succ / n) if n else 0.0,
        "poi_lower": _wilson_lower(succ, n),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _weighted_placement(cumulative_all: dict, us_agent_id: str,
                        our_mult: float, opp_mult: float) -> int:
    """Finals-scoring placement: rank the 6 agents by ``raw_ae * mult`` and return
    OUR rank (1 = best). Our agent gets ``our_mult``; every opponent ``opp_mult``.
    Ties broken by the agent's original order (stable, deterministic)."""
    order_index = {a: i for i, a in enumerate(cumulative_all)}
    weighted = {a: v * (our_mult if a == us_agent_id else opp_mult)
                for a, v in cumulative_all.items()}
    ranked = sorted(cumulative_all, key=lambda a: (-weighted[a], order_index[a]))
    return ranked.index(us_agent_id) + 1


def _axis_totals(components: dict) -> dict:
    """Collapse the per-step reward-component decomposition into the three gated
    axes. mission/base_defense come from components; opening is handled separately
    (opening_reward). Sign convention: higher is better for ALL axes, so base_defense
    (a sum of negatives) improves when it becomes LESS negative."""
    g = lambda k: float(components.get(k, 0.0))
    return {
        "mission_axis": g("collect_mission") + g("collect_recon"),
        "base_defense_axis": g("own_base_destroyed") + g("self_damage") + g("base_damage"),
    }


def _paired_value_stats(cand: dict, inc: dict, key: str) -> dict:
    """Paired (suite, hash, sim) stats for a per-run numeric ``key`` where HIGHER
    is better (raw_ae, mission_axis, …). delta = cand - inc; PoI = fraction of
    pairs the candidate wins (ties = 0.5). Mirrors _paired_placement_stats."""
    inc_rows: dict[tuple, float] = {}
    for suite, b in inc.get("per_bracket", {}).items():
        for r in b.get("run_rows", []):
            if key in r:
                inc_rows[(suite, r["hash"], r["sim"])] = float(r[key])
    deltas: list[float] = []
    wins = ties = losses = 0
    for suite, b in cand.get("per_bracket", {}).items():
        for r in b.get("run_rows", []):
            k = (suite, r["hash"], r["sim"])
            if k not in inc_rows or key not in r:
                continue
            d = float(r[key]) - inc_rows[k]
            deltas.append(d)
            if d > 1e-9:
                wins += 1
            elif d < -1e-9:
                losses += 1
            else:
                ties += 1
    n = len(deltas)
    succ = wins + 0.5 * ties
    return {
        "n_pairs": n,
        "mean_delta": (sum(deltas) / n) if n else 0.0,
        "poi": (succ / n) if n else 0.0,
        "poi_lower": _wilson_lower(succ, n),
        "wins": wins, "ties": ties, "losses": losses,
    }


# ---------------------------------------------------------------------------
# Reform 4 — held-out composition transfer gap (overfit alarm)
# ---------------------------------------------------------------------------
def _gap_not_widening(cand_gap: float, inc_gap: float, tol: float = GAP_TOL) -> bool:
    """True when the candidate's (heldout - tune) placement gap does not widen
    beyond the incumbent's by more than ``tol``. A widening gap = proxy-overfit."""
    return cand_gap <= inc_gap + tol + 1e-9


def _aggregate_candidate(label: str, spec: dict, runs_by_bracket: dict[str, list[dict]]) -> dict:
    """Pure aggregation of per-(hash,sim) worker rows into a candidate summary.
    ``runs_by_bracket`` maps bracket name -> list of worker payload rows (each row
    already carries hash/sim/raw_ae/axes/weighted_placement)."""
    per_bracket: dict[str, dict] = {}
    robust_candidates: list[float] = []
    for suite, runs in runs_by_bracket.items():
        per_bracket[suite] = {
            "runs": len(runs),
            "mean_placement": _agg([r["mean_placement"] for r in runs]),
            "win_rate": _agg([r["win_rate"] for r in runs]),
            "mean_margin": _agg([r["mean_margin"] for r in runs]),
            "mean_score": _agg([r["mean_score"] for r in runs]),
            "raw_ae": _agg([r["raw_ae"] for r in runs]),
            "mission_axis": _agg([r["mission_axis"] for r in runs]),
            "base_defense_axis": _agg([r["base_defense_axis"] for r in runs]),
            "opening_axis": _agg([r["opening_axis"] for r in runs]),
            "weighted_placement": {
                om: _agg([r["weighted_placement"][om] for r in runs])
                for om in (runs[0]["weighted_placement"] if runs else {})
            },
            "run_rows": [{"hash": r["hash"], "sim": r["sim"],
                          "mean_placement": r["mean_placement"],
                          "raw_ae": r["raw_ae"], "mission_axis": r["mission_axis"],
                          "base_defense_axis": r["base_defense_axis"],
                          "opening_axis": r["opening_axis"]} for r in runs],
        }
        robust_candidates.extend(per_bracket[suite]["weighted_placement"].values())
    return {
        "label": label,
        "our": spec["our"],
        "per_bracket": per_bracket,
        "worst_bracket_placement": max(b["mean_placement"] for b in per_bracket.values()),
        "worst_robust_placement": max(robust_candidates) if robust_candidates else 0.0,
        "tune_mean_placement": _agg([b["mean_placement"] for b in per_bracket.values()]),
        "min_margin": min(b["mean_margin"] for b in per_bracket.values()),
        "semis_mixed_score": per_bracket.get("semis_mixed", {}).get("mean_score", 0.0),
        "raw_ae": _agg([b["raw_ae"] for b in per_bracket.values()]),
        "mission_axis": _agg([b["mission_axis"] for b in per_bracket.values()]),
        "base_defense_axis": _agg([b["base_defense_axis"] for b in per_bracket.values()]),
        "opening_axis": _agg([b["opening_axis"] for b in per_bracket.values()]),
    }


def _evaluate_candidate(label: str, spec: dict, brackets: list[str],
                        hash_seeds: list[int], sim_seeds: list[int],
                        rounds: int, non_novice: bool,
                        us_slots: list[int] | None = None) -> dict:
    us_slots = us_slots or [0]
    runs_by_bracket: dict[str, list[dict]] = {}
    for suite in brackets:
        runs = []
        for slot in us_slots:
            for h in hash_seeds:
                for s in sim_seeds:
                    row = _spawn_worker(spec["our"], suite, h, s, rounds,
                                        spec.get("env", {}), non_novice, us_slot=slot)
                    row["hash"] = h
                    row["sim"] = s
                    row["us_slot"] = slot
                    runs.append(row)
        runs_by_bracket[suite] = runs
        wp = {om: _agg([r["weighted_placement"][om] for r in runs])
              for om in (runs[0]["weighted_placement"] if runs else {})}
        print(f"    {suite:16s} place={_agg([r['mean_placement'] for r in runs]):.2f} "
              f"raw_ae={_agg([r['raw_ae'] for r in runs]):+.0f} "
              f"wplace={{ {', '.join(f'{k}:{v:.2f}' for k, v in wp.items())} }} "
              f"(n={len(runs)})", flush=True)
    return _aggregate_candidate(label, spec, runs_by_bracket)


_AXIS_KEY = {"mission": "mission_axis", "base_defense": "base_defense_axis",
             "opening": "opening_axis"}


def _promotion_verdict(cand: dict, inc: dict, target_axis: str | None = None) -> dict:
    """Finals-aligned gate (2026-06-09). PRIMARY = placement under the opp_mult
    sweep (worst_robust_placement) must not regress. raw_ae is the candidate
    discriminator with a non-crater floor. If ``target_axis`` is set, that axis
    must improve (higher = better for all three, incl. base_defense which improves
    by becoming less negative). The --heldout composition gap must not widen."""
    # PRIMARY — robust placement across (brackets x opp_mults).
    placement_robust_ok = cand["worst_robust_placement"] <= inc["worst_robust_placement"] + 1e-9
    # raw_ae non-crater floor (proportional to our final score; the discriminator).
    rawae_floor = RAWAE_FLOOR_FRAC * inc["raw_ae"]
    rawae_floor_ok = cand["raw_ae"] >= rawae_floor - 1e-9
    # raw_ae real-gain stats (used as the placement-tie discriminator / reporting).
    rawae_stats = _paired_value_stats(cand, inc, "raw_ae")
    # margin/score non-crater floors (carried over from the prior gate).
    margin_floor = inc["min_margin"] - MARGIN_SLACK_FRAC * (abs(inc["min_margin"]) + 1.0)
    margin_ok = cand["min_margin"] >= margin_floor - 1e-9
    score_floor = SCORE_FLOOR_FRAC * inc["semis_mixed_score"]
    score_ok = cand["semis_mixed_score"] >= score_floor - 1e-9
    # Per-axis floor for the targeted change.
    axis_ok = True
    axis_stats = None
    if target_axis is not None:
        key = _AXIS_KEY[target_axis]
        axis_stats = _paired_value_stats(cand, inc, key)
        axis_ok = (axis_stats["n_pairs"] > 0
                   and axis_stats["mean_delta"] >= MIN_EFFECT_AXIS + 1e-9
                   and axis_stats["poi_lower"] > 0.5 + 1e-9)
    verdict = {
        "placement_robust_ok": bool(placement_robust_ok),
        "rawae_floor_ok": bool(rawae_floor_ok),
        "margin_noncrater_ok": bool(margin_ok),
        "score_noncrater_ok": bool(score_ok),
        "rawae_stats": rawae_stats,
        "target_axis": target_axis,
        "axis_ok": bool(axis_ok),
        "axis_stats": axis_stats,
    }
    gap_ok = True
    if "composition_gap" in cand and "composition_gap" in inc:
        gap_ok = _gap_not_widening(cand["composition_gap"], inc["composition_gap"])
        verdict["gap_not_widening_ok"] = bool(gap_ok)
        verdict["composition_gap"] = cand["composition_gap"]
    verdict["promotable"] = bool(
        placement_robust_ok and rawae_floor_ok and margin_ok and score_ok
        and axis_ok and gap_ok)
    return verdict


def _print_report(results: dict[str, dict], incumbent: str, brackets: list[str],
                  target_axis: str | None = None) -> None:
    inc = results[incumbent]
    print("\n" + "=" * 78)
    print(f"  MELEE RE-RANK  (incumbent = {incumbent}; robust placement over "
          f"{len(brackets)} brackets × opp-mults)")
    print("=" * 78)
    for label, cand in results.items():
        print(f"\n  {label}   [worst-bracket placement = {cand['worst_bracket_placement']:.2f}, "
              f"min-margin = {cand['min_margin']:+.1f}, "
              f"semis_mixed score = {cand['semis_mixed_score']:.4f}]")
        print(f"    {'bracket':16s} {'place':>6s} {'win':>6s} {'margin':>9s} {'score':>8s}")
        for suite in brackets:
            b = cand["per_bracket"][suite]
            print(f"    {suite:16s} {b['mean_placement']:6.2f} {b['win_rate']:6.2f} "
                  f"{b['mean_margin']:+9.1f} {b['mean_score']:8.4f}")
        if "composition_gap" in cand:
            print(f"    heldout-gap = {cand['composition_gap']:+.2f} "
                  f"(heldout {cand.get('heldout_mean_placement', float('nan')):.2f} "
                  f"- tune {cand['tune_mean_placement']:.2f})")
        if label != incumbent:
            v = _promotion_verdict(cand, inc, target_axis=target_axis)
            st = v["rawae_stats"]
            flag = "PROMOTABLE ✓" if v["promotable"] else "not promotable"
            cks = [f"placement-robust {'✓' if v['placement_robust_ok'] else '✗'}",
                   f"rawae-floor {'✓' if v['rawae_floor_ok'] else '✗'}",
                   f"margin-floor {'✓' if v['margin_noncrater_ok'] else '✗'}",
                   f"score-floor {'✓' if v['score_noncrater_ok'] else '✗'}"]
            if v.get("target_axis") is not None:
                cks.append(f"axis[{v['target_axis']}] {'✓' if v['axis_ok'] else '✗'}")
            if "gap_not_widening_ok" in v:
                cks.append(f"gap {'✓' if v['gap_not_widening_ok'] else '✗'}")
            print(f"    -> vs {incumbent}: {flag}  ({', '.join(cks)})")
            print(f"       raw_ae: Δ={st['mean_delta']:+.1f} "
                  f"PoI={st['poi']:.2f} (Wilson95 lo={st['poi_lower']:.2f}) "
                  f"W/T/L={st['wins']}/{st['ties']}/{st['losses']} n={st['n_pairs']}")
    print("\n" + "=" * 78)
    print("  PRIMARY = worst robust placement under the opp_mult sweep (lower = less")
    print("  exploitable); raw_ae is the finals-proportional DISCRIMINATOR (floored). Reward")
    print("  floors (margin/score/cloud) are non-crater, not selectors (finals = RELATIVE RANK).")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    global OUR_MULT, OPP_MULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--our", default=None, help="(worker) agent name for simulate._make_our_agent")
    p.add_argument("--us-slot", type=int, default=0,
                   help="(worker) which of the 6 fixed novice spawns to seat us at (0-5)")
    p.add_argument("--us-slots", nargs="+", type=int, default=[0],
                   help="spawns to seat us at, averaged per bracket. Finals seats teams at "
                        "VARYING spawns, so '0 1 2 3 4 5' models uniform seating; the novice "
                        "opening book only fires on bases 9,13/2,6/6,2 (slots 1/3/4).")
    p.add_argument("--suite", default=None, help="(worker) single bracket name")
    p.add_argument("--seed", type=int, default=42, help="(worker) sim seed start")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--non-novice", action="store_true")
    # parent-only
    p.add_argument("--candidates", nargs="+", default=None,
                   help=f"registry labels to rank (default: {DEFAULT_CANDIDATES})")
    p.add_argument("--incumbent", default=DEFAULT_INCUMBENT)
    p.add_argument("--confpol-ckpt", action="append", default=[], metavar="LABEL=PATH",
                   help="register a confpol candidate over a given raw-policy checkpoint "
                        "(repeatable; e.g. native-u200=training/ae/checkpoints/confpol-native-u200.pt)")
    p.add_argument("--policy-ckpt", action="append", default=[], metavar="LABEL=PATH",
                   help="register a raw-policy (full-control) candidate over a checkpoint (repeatable)")
    p.add_argument("--opening-ckpt", action="append", default=[], metavar="LABEL=PATH",
                   help="register an opening_hybrid candidate (novice opening book + confpol planner) "
                        "over a checkpoint; accepts the same @knob=val deploy-profile suffix as "
                        "--confpol-ckpt (repeatable)")
    p.add_argument("--brackets", nargs="+", default=None,
                   help=f"override brackets (default: {MELEE_BRACKETS})")
    p.add_argument("--hash-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--sim-seeds", nargs="+", type=int,
                   default=[42, 137, 7, 99, 256, 512, 1024, 2048])
    p.add_argument("--opp-mults", nargs="+", type=float, default=OPP_MULTS,
                   help="opponent mission-multiplier sweep for placement (default 0.24 0.7)")
    p.add_argument("--our-mult", type=float, default=OUR_MULT,
                   help="our mission-multiplier (perception accuracy; default 0.93)")
    p.add_argument("--target-axis", choices=["mission", "base_defense", "opening"],
                   default=None, help="axis floor to enforce for this change")
    p.add_argument("--heldout", action="store_true",
                   help="also evaluate each candidate on the frozen held-out real-competitor "
                        "compositions (opponents.HELDOUT_COMPOSITIONS) and gate on the "
                        "train-vs-heldout placement GAP (overfit alarm, Reform 4).")
    p.add_argument("--summary-out", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.worker:
        if args.our_mult <= 0 or any(m <= 0 for m in args.opp_mults):
            p.error("--our-mult and --opp-mults must all be > 0")
        os.environ["AE_EVAL_OUR_MULT"] = str(args.our_mult)
        os.environ["AE_EVAL_OPP_MULTS"] = ",".join(f"{m:g}" for m in args.opp_mults)
        OUR_MULT = args.our_mult
        OPP_MULTS = list(args.opp_mults)

    if args.worker:
        if not args.our or not args.suite:
            p.error("--worker requires --our and --suite")
        return _run_worker(args)

    registry = _base_registry()
    for spec_str in args.confpol_ckpt:
        label, env = _parse_confpol_spec(spec_str)
        registry[label] = {"our": "confidence_policy_hybrid", "env": env}
    for spec_str in args.policy_ckpt:
        label, _, path = spec_str.partition("=")
        registry[label] = {"our": "policy",
                           "env": {"AE_POLICY_CHECKPOINT": str(Path(path).resolve())}}
    for spec_str in args.opening_ckpt:
        label, env = _parse_confpol_spec(spec_str)
        registry[label] = {"our": "opening_hybrid", "env": env}

    candidates = args.candidates or DEFAULT_CANDIDATES
    brackets = args.brackets or list(MELEE_BRACKETS)
    incumbent = args.incumbent
    if incumbent not in candidates:
        candidates = [incumbent] + [c for c in candidates if c != incumbent]
    for label in candidates:
        if label not in registry:
            p.error(f"unknown candidate {label!r}; known: {sorted(registry)}")

    print(f"melee_eval: candidates={candidates} brackets={brackets} "
          f"rounds={args.rounds} hash_seeds={args.hash_seeds} sim_seeds={args.sim_seeds}",
          flush=True)
    t0 = time.monotonic()
    results: dict[str, dict] = {}
    for label in candidates:
        print(f"\n[candidate] {label} (our={registry[label]['our']})", flush=True)
        res = _evaluate_candidate(
            label, registry[label], brackets,
            args.hash_seeds, args.sim_seeds, args.rounds, args.non_novice,
            us_slots=args.us_slots,
        )
        if args.heldout:
            print(f"  [heldout compositions] {label}", flush=True)
            ho = _evaluate_candidate(
                label, registry[label], list(HELDOUT_COMPOSITIONS.values()),
                args.hash_seeds, args.sim_seeds, args.rounds, args.non_novice,
                us_slots=args.us_slots,
            )
            res["heldout_per_bracket"] = ho["per_bracket"]
            res["heldout_mean_placement"] = ho["tune_mean_placement"]
            res["heldout_worst_placement"] = ho["worst_bracket_placement"]
            res["composition_gap"] = ho["tune_mean_placement"] - res["tune_mean_placement"]
        results[label] = res
    elapsed = time.monotonic() - t0

    _print_report(results, incumbent, brackets, target_axis=args.target_axis)
    print(f"\nelapsed_s={elapsed:.1f}")

    if args.summary_out is not None:
        report = {
            "candidates": candidates, "incumbent": incumbent, "brackets": brackets,
            "rounds": args.rounds, "hash_seeds": args.hash_seeds, "sim_seeds": args.sim_seeds,
            "elapsed_s": elapsed,
            "results": results,
            "verdicts": {label: _promotion_verdict(results[label], results[incumbent],
                                                    target_axis=args.target_axis)
                         for label in candidates if label != incumbent},
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report, indent=2))
        print(f"summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
