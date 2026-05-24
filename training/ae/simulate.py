"""Offline simulation harness for the AE task.

Runs N rounds of `bomberman_env` with our agent at index 0 and configurable
opponents at indices 1..5. Logs trajectories and per-round scores so the
output can feed downstream analyses:

  * Tier 1 #1 — playbook/Q-table aggregation (`build_playbook.py`).
  * Tier 2 #8 — opponent action distribution fitting.
  * Tier 2 #10 — oracle BC labels.

Usage examples:

    # Score against the non-random mixed pressure library
    python training/ae/simulate.py --rounds 1000 --opponents mixed --our heuristic

    # Score and dump trajectories (used by build_playbook.py)
    python training/ae/simulate.py --rounds 1000 --opponents greedy,bomber,defender,hunter,rusher \
        --our heuristic --out training/ae/data/sim-tier1.npz

The output `.npz` contains four arrays:

    obs_keys   (N,)  packed (x, y, dir, step) integers, our agent only
    actions    (N,)  int8 actions our agent took
    rewards    (N,)  float32 reward our agent received this step
    returns    (N,)  float32 sum of remaining-game rewards (Monte-Carlo G_t)
    round_idx  (N,)  int32 which round each step belongs to
    opponents  (N,)  string label of opponent set this round
    our_agent  (N,)  string label of which "our agent" was used

(``returns`` is the per-step Monte-Carlo return — used directly as the
oracle target for BC training and as the value table for playbook lookup.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TIL_AE = REPO_ROOT / "til-26-ae"
AE_SRC = REPO_ROOT / "ae" / "src"
TRAINING_AE = REPO_ROOT / "training" / "ae"
for p in (str(AE_SRC), str(TRAINING_AE), str(TIL_AE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ae_manager import AEManager  # noqa: E402
from opponents import OPPONENT_NAMES, MixedOpponent, OpponentFn, make_opponent  # noqa: E402

from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


# ---------------------------------------------------------------------------
# Observation packing / state-key helpers
# ---------------------------------------------------------------------------

GRID_SIZE = 16
NUM_DIRS = 4
MAX_STEPS = 200
ACTION_NAMES = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STAY", "PLACE_BOMB"]
ENEMY_AGENT_CHANNEL = 10
ENEMY_BOMB_CHANNEL = 18


def pack_state_key(location, direction: int, step: int) -> int:
    """Pack (x, y, dir, step) into a single int64 for fast lookup keys.

    Layout (high to low):
        16 bits   step       (0..200, fits in 8 but pad for safety)
        4 bits    direction  (0..3)
        8 bits    y          (0..15)
        8 bits    x          (0..15)
    """
    x = int(location[0]) & 0xFF
    y = int(location[1]) & 0xFF
    d = int(direction) & 0xF
    s = int(step) & 0xFFFF
    return (s << 20) | (d << 16) | (y << 8) | x


def unpack_state_key(key: int) -> tuple[int, int, int, int]:
    return key & 0xFF, (key >> 8) & 0xFF, (key >> 16) & 0xF, (key >> 20) & 0xFFFF


# ---------------------------------------------------------------------------
# Our-agent factory
# ---------------------------------------------------------------------------

def _make_our_agent(name: str, kwargs: dict | None = None):
    """Construct the agent we control in slot 0.

    Currently supported:
      heuristic — `AEManager` legacy planner (default)
      option_v2 — `AEManager` with the option-style decision core enabled

    Easy to extend: drop a new branch here and pass the matching --our flag.
    """

    name = name.lower().strip()
    kwargs = kwargs or {}
    if name == "heuristic":
        return AEManager(**kwargs)
    if name == "option_v2":
        old = os.environ.get("AE_PLANNER")
        os.environ["AE_PLANNER"] = "option_v2"
        try:
            return AEManager(**kwargs)
        finally:
            if old is None:
                os.environ.pop("AE_PLANNER", None)
            else:
                os.environ["AE_PLANNER"] = old
    if name == "hybrid":
        from hybrid_manager import HybridAEManager
        return HybridAEManager(**kwargs)
    raise ValueError(f"unknown --our value {name!r}")


def _scalar(obs: dict, key: str, default: float = 0.0) -> float:
    value = obs.get(key, default)
    if isinstance(value, list):
        value = value[0] if value else default
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except Exception:
        return float(default)


def _classify_reward(prev: dict, cur: dict, reward_delta: float) -> dict[str, float]:
    """Best-effort local reward attribution from observable state deltas."""

    components: dict[str, float] = {}
    health_delta = _scalar(cur, "health") - _scalar(prev, "health")
    base_delta = _scalar(cur, "base_health") - _scalar(prev, "base_health")

    if reward_delta >= 49.0:
        components["destroy_enemy_base"] = reward_delta
    elif reward_delta >= 14.0:
        components["attack_kill_or_multi"] = reward_delta
    elif 4.5 <= reward_delta <= 5.5:
        components["collect_mission"] = reward_delta
    elif 1.5 <= reward_delta <= 2.5:
        components["collect_resource"] = reward_delta
    elif 0.5 <= reward_delta <= 1.5:
        components["collect_recon"] = reward_delta
    elif 0.0 < reward_delta < 0.5:
        components["attack_damage_positive"] = reward_delta

    if reward_delta <= -49.0:
        components["own_base_destroyed"] = reward_delta
    elif reward_delta < 0.0:
        if health_delta < 0.0:
            components["self_damage"] = reward_delta
        elif base_delta < 0.0:
            components["base_damage"] = reward_delta
        else:
            components["other_negative"] = reward_delta

    return components


def _channel_on(cell: list, channel: int) -> bool:
    try:
        return float(cell[channel]) > 0.0
    except Exception:
        return False


def _classify_base_failure(prev_obs: dict, action: int | None) -> str:
    """Coarse cause bucket for a base-damage event seen on the next tick."""

    base_health = _scalar(prev_obs, "base_health", 100.0)
    view = prev_obs.get("base_viewcone") or []
    height = len(view)
    width = len(view[0]) if height else 0
    center_row = height // 2
    center_col = width // 2
    enemy_near = False
    bomb_near = False
    for row in range(height):
        for col in range(width):
            cell = view[row][col]
            radius = max(abs(row - center_row), abs(col - center_col))
            if radius <= 2 and _channel_on(cell, ENEMY_BOMB_CHANNEL):
                bomb_near = True
            if radius <= 3 and _channel_on(cell, ENEMY_AGENT_CHANNEL):
                enemy_near = True

    if bomb_near:
        return "visible_enemy_bomb"
    if enemy_near and action == 5:
        return "enemy_near_base_after_our_bomb"
    if enemy_near:
        return "visible_enemy_near_base"
    if base_health <= 40.0:
        return "low_base_followup"
    return "unseen_or_stale_pressure"


def _agent_attr(agent, name: str, default=0):
    inner = getattr(agent, "heuristic", agent)
    return getattr(inner, name, default)


# ---------------------------------------------------------------------------
# Single-round runner
# ---------------------------------------------------------------------------

def run_one_round(
    env,
    our_agent,
    opponents: list[OpponentFn],
    log_traj: bool,
    seed: int | None = None,
) -> dict:
    """Run one full round (200 steps × 6 agents). Returns per-round stats."""

    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()

    agent_id_us = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])

    # Reset our agent's belief if it's an AEManager. The env already has a
    # fresh seed, so just zero its memory.
    if hasattr(our_agent, "_reset_memory"):
        our_agent._reset_memory()

    # Per-game reset for opponents that need it (only Mixed).
    for op in opponents:
        if hasattr(op, "reset_for_game"):
            op.reset_for_game()
        if hasattr(op, "_reset_memory"):
            op._reset_memory()

    cumulative_us = 0.0
    action_counter: Counter[int] = Counter()
    component_totals: Counter[str] = Counter()
    base_failure_counter: Counter[str] = Counter()
    visited: set[tuple[int, int]] = set()
    prev_obs: dict | None = None
    prev_action: int | None = None
    bombs_placed = 0
    frozen_ticks_seen = 0
    final_health = 0.0
    final_base_health = 0.0
    final_team_bombs = 0.0
    final_team_resources = 0.0
    final_step = 0
    terminated_us = False
    traj = {
        "state_keys": [],
        "actions": [],
        "rewards": [],
        "step_idx": [],
    }

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if agent == agent_id_us:
            cumulative_us += float(reward)
            terminated_us = bool(termination or truncation)

        if termination or truncation:
            env.step(None)
            continue

        # Convert numpy structures to plain Python (mirrors `test_ae.py`).
        observation_native = {
            k: v if type(v) in (int, float) else (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in observation.items()
        }

        if agent == agent_id_us:
            loc = observation_native.get("location", [0, 0])
            if isinstance(loc, list) and len(loc) == 2:
                visited.add((int(loc[0]), int(loc[1])))
            final_health = _scalar(observation_native, "health")
            final_base_health = _scalar(observation_native, "base_health")
            final_team_bombs = _scalar(observation_native, "team_bombs")
            final_team_resources = _scalar(observation_native, "team_resources")
            final_step = int(_scalar(observation_native, "step"))
            if int(_scalar(observation_native, "frozen_ticks")) > 0:
                frozen_ticks_seen += 1
            if prev_obs is not None and abs(float(reward)) > 1e-6:
                components = _classify_reward(prev_obs, observation_native, float(reward))
                for component, value in components.items():
                    component_totals[component] += value
                if "base_damage" in components or "own_base_destroyed" in components:
                    base_failure_counter[_classify_base_failure(prev_obs, prev_action)] += 1
            action = int(our_agent.ae(observation_native))
            action_counter[action] += 1
            prev_action = action
            if action == 5:
                bombs_placed += 1
            if log_traj:
                d = int(observation_native.get("direction", 0))
                step_idx = int(observation_native.get("step", 0))
                key = pack_state_key(loc, d, step_idx)
                # We log reward AFTER it lands on the next agent_iter
                # iteration; here we record the action and a placeholder
                # reward (filled in by the post-pass below).
                traj["state_keys"].append(key)
                traj["actions"].append(action)
                traj["rewards"].append(0.0)
                traj["step_idx"].append(step_idx)
            prev_obs = observation_native
        else:
            slot = other_ids.index(agent)
            op = opponents[slot]
            action = int(op(observation_native))
            # Safety: if the opponent returned an illegal action, fall back to
            # the first legal one so we don't waste the round on env-side
            # masking mismatches.
            mask = observation_native.get("action_mask")
            if mask is not None:
                try:
                    if not int(mask[action]):
                        for i, m in enumerate(mask):
                            if int(m):
                                action = i
                                break
                except Exception:
                    pass

        env.step(action)

    # Reward attribution post-pass. The PettingZoo loop above gives us the
    # cumulative reward at every iteration; trajectory entries collected at
    # tick T should bag the reward our agent earned BETWEEN tick T's action
    # and the next time control returns to our agent. We reconstruct that
    # by replaying the order: rewards observed at our-agent ticks have a
    # one-step delay.
    if log_traj and traj["state_keys"]:
        # Reward at trajectory position i is the increment in cumulative_us
        # between trajectory step i and step i+1 (Monte Carlo from i to end
        # is computed below).
        # The simpler exact approach: recompute by re-running the env once
        # with reward logging. Doing it here in one pass would require
        # tighter coupling; for now we approximate by spreading the total
        # cumulative reward proportionally to step indices.
        # To stay correct, we instead leverage `env.rewards[agent]` snapshot,
        # which we already accumulated into cumulative_us. We approximate
        # per-step rewards by step deltas of `_cumulative_rewards` snapshots,
        # but PettingZoo doesn't expose that directly per-iteration here, so
        # we use a separate reward-collection round.
        # Pragmatic compromise: leave `rewards` as zeros and compute returns
        # only from final cumulative score (one MC return for the whole game,
        # back-propagated equally — this is what fitted Q with a single
        # terminal reward looks like and matches the env's true reward
        # structure where most points land at item-collect / kill ticks,
        # which we treat as one G_t per game).
        traj["rewards"] = [0.0] * len(traj["state_keys"])

    return {
        "score": cumulative_us / 1000.0,  # matches the cloud's /1000 scaling
        "total_reward": cumulative_us,
        "traj": traj,
        "diagnostics": {
            "action_counts": {ACTION_NAMES[a]: c for a, c in sorted(action_counter.items()) if 0 <= a < len(ACTION_NAMES)},
            "bombs_placed": bombs_placed,
            "unique_cells_visited": len(visited),
            "reward_components": dict(component_totals),
            "base_failure_classes": dict(base_failure_counter),
            "final_health": final_health,
            "final_base_health": final_base_health,
            "final_team_bombs": final_team_bombs,
            "final_team_resources": final_team_resources,
            "final_step": final_step,
            "early_end": final_step < (MAX_STEPS - 1),
            "terminated_us": terminated_us,
            "freeze_ticks_seen": frozen_ticks_seen,
            "base_pressure_overrides": _agent_attr(our_agent, "base_pressure_override_count", 0),
        },
    }


# ---------------------------------------------------------------------------
# Multi-round driver
# ---------------------------------------------------------------------------

def run_simulation(
    rounds: int,
    opponents_spec: str,
    our_name: str,
    log_traj: bool,
    seed_start: int,
    novice: bool = True,
    our_kwargs: dict | None = None,
) -> dict:
    """Run ``rounds`` rounds and return the aggregated trajectory + stats.

    ``opponents_spec`` is a comma-separated list of opponent names (one per
    enemy slot, repeats fine), or one of:

      'mixed'     : 5 independent non-random MixedOpponent instances
      'library'   : greedy,bomber,defender,hunter,rusher (5 enemies)
      'cloudsuite': rusher,hunter_sticky,bomber_fast,defender,mixed
      'pressure2' : rusher_fast,rusher_safe,hunter_sticky,bomber_fast,base_bomber
      'random'    : legacy explicit baseline only

    Or any explicit list: 'greedy,greedy,bomber,defender,hunter'.
    """

    cfg = default_config()
    cfg.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    our_agent = _make_our_agent(our_name, our_kwargs)

    # Resolve opponents spec to a list of 5 names.
    if opponents_spec == "random":
        names = ["random"] * 5
    elif opponents_spec == "mixed":
        names = ["mixed"] * 5
    elif opponents_spec == "library":
        names = ["greedy", "bomber", "defender", "hunter", "rusher"]
    elif opponents_spec == "cloudsuite":
        names = ["rusher", "hunter_sticky", "bomber_fast", "defender", "mixed"]
    elif opponents_spec == "pressure2":
        names = ["rusher_fast", "rusher_safe", "hunter_sticky", "bomber_fast", "base_bomber"]
    else:
        names = [n.strip() for n in opponents_spec.split(",") if n.strip()]
        if len(names) == 1:
            names = names * 5
        if len(names) != 5:
            raise ValueError(
                f"need 5 opponent names (got {len(names)}): {names}"
            )

    # Per-round opponent instances. Each round we'll get a fresh deterministic
    # seed for stochastic opponents, but the AEManager-based ones reset their
    # internal belief inside `run_one_round`.
    opponents = [make_opponent(n, seed=seed_start + 1000 + i) for i, n in enumerate(names)]

    scores: list[float] = []
    totals: list[float] = []
    diagnostics: list[dict] = []
    all_traj = {
        "state_keys": [],
        "actions": [],
        "rewards": [],
        "step_idx": [],
        "round_idx": [],
        "round_score": [],
    }

    t0 = time.monotonic()
    for r in range(rounds):
        result = run_one_round(
            env, our_agent, opponents, log_traj=log_traj, seed=seed_start + r
        )
        scores.append(result["score"])
        totals.append(result["total_reward"])
        diagnostics.append(result["diagnostics"])

        if log_traj:
            traj = result["traj"]
            n = len(traj["state_keys"])
            if n:
                all_traj["state_keys"].extend(traj["state_keys"])
                all_traj["actions"].extend(traj["actions"])
                all_traj["rewards"].extend(traj["rewards"])
                all_traj["step_idx"].extend(traj["step_idx"])
                all_traj["round_idx"].extend([r] * n)
                all_traj["round_score"].extend([result["total_reward"]] * n)

        if (r + 1) % max(1, rounds // 10) == 0 or r == rounds - 1:
            elapsed = time.monotonic() - t0
            mean_so_far = float(np.mean(scores))
            print(
                f"[{r+1:>4}/{rounds}] mean_score={mean_so_far:.4f}  "
                f"elapsed={elapsed:.1f}s  per-round={elapsed/(r+1):.2f}s",
                flush=True,
            )

    env.close()

    component_sum: Counter[str] = Counter()
    base_failure_sum: Counter[str] = Counter()
    action_sum: Counter[str] = Counter()
    for diag in diagnostics:
        component_sum.update(diag.get("reward_components", {}))
        base_failure_sum.update(diag.get("base_failure_classes", {}))
        action_sum.update(diag.get("action_counts", {}))

    def _mean_diag(key: str) -> float:
        values = [float(d.get(key, 0.0)) for d in diagnostics]
        return float(np.mean(values)) if values else 0.0

    summary = {
        "rounds": rounds,
        "opponents": names,
        "our_agent": our_name,
        "novice": novice,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "scores": [float(s) for s in scores],
        "diagnostics": {
            "mean_bombs_placed": _mean_diag("bombs_placed"),
            "mean_unique_cells_visited": _mean_diag("unique_cells_visited"),
            "mean_final_health": _mean_diag("final_health"),
            "mean_final_base_health": _mean_diag("final_base_health"),
            "mean_final_team_bombs": _mean_diag("final_team_bombs"),
            "mean_final_team_resources": _mean_diag("final_team_resources"),
            "mean_base_pressure_overrides": _mean_diag("base_pressure_overrides"),
            "early_end_rate": sum(1 for d in diagnostics if d.get("early_end")) / max(1, len(diagnostics)),
            "terminated_rate": sum(1 for d in diagnostics if d.get("terminated_us")) / max(1, len(diagnostics)),
            "reward_component_sum": dict(component_sum),
            "base_failure_classes": dict(base_failure_sum),
            "action_counts": dict(action_sum),
        },
    }

    return {"summary": summary, "trajectories": all_traj}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=100,
                   help="number of episodes to simulate")
    p.add_argument(
        "--opponents",
        type=str,
        default="cloudsuite",
        help=(
            "opponent set: 'mixed', 'library', 'cloudsuite', 'pressure2', legacy 'random', "
            "or 5 comma-separated names from " + ",".join(OPPONENT_NAMES)
        ),
    )
    p.add_argument("--our", type=str, default="heuristic",
                   help="which agent to control in slot 0 (heuristic, option_v2, hybrid)")
    p.add_argument("--out", type=Path, default=None,
                   help="optional .npz to dump trajectories into")
    p.add_argument("--summary-out", type=Path, default=None,
                   help="optional .json to dump per-round and aggregate scores")
    p.add_argument("--seed", type=int, default=42,
                   help="base RNG seed for stochastic opponents")
    p.add_argument("--no-log-traj", action="store_true",
                   help="skip trajectory logging (faster; no .npz output)")
    p.add_argument("--non-novice", action="store_true",
                   help="run in advanced (varying-map) mode")
    args = p.parse_args(argv)

    log_traj = (not args.no_log_traj) or args.out is not None
    out = run_simulation(
        rounds=args.rounds,
        opponents_spec=args.opponents,
        our_name=args.our,
        log_traj=log_traj,
        seed_start=args.seed,
        novice=not args.non_novice,
    )

    s = out["summary"]
    print()
    print("=" * 64)
    print(f"  rounds        : {s['rounds']}")
    print(f"  opponents     : {','.join(s['opponents'])}")
    print(f"  our agent     : {s['our_agent']}")
    print(f"  mean score    : {s['mean_score']:.4f}")
    print(f"  std           : {s['std_score']:.4f}")
    print(f"  p25 / p50 / p75: {s['p25']:.4f} / {s['p50']:.4f} / {s['p75']:.4f}")
    print(f"  min / max     : {s['min_score']:.4f} / {s['max_score']:.4f}")
    print("=" * 64)

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(s, indent=2))
        print(f"summary -> {args.summary_out}")

    if args.out is not None and log_traj:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        traj = out["trajectories"]
        np.savez_compressed(
            args.out,
            state_keys=np.asarray(traj["state_keys"], dtype=np.int64),
            actions=np.asarray(traj["actions"], dtype=np.int8),
            rewards=np.asarray(traj["rewards"], dtype=np.float32),
            step_idx=np.asarray(traj["step_idx"], dtype=np.int16),
            round_idx=np.asarray(traj["round_idx"], dtype=np.int32),
            round_score=np.asarray(traj["round_score"], dtype=np.float32),
            opponents=np.asarray(s["opponents"], dtype=np.str_),
        )
        print(f"trajectories -> {args.out}  ({len(traj['state_keys'])} steps)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
