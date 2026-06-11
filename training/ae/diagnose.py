"""Per-round reward-component diagnostic for AE.

CODEX recommendation (see ae/NOTES.md §"CODEX recommendation"):

> "Make a replay/diagnostic harness. Add logging to local AE eval that
> records per-episode reward components, final position, collected
> mission / recon / resource counts, bombs placed, bomb hits,
> self-damage, base damage, freezes, and visited-cell coverage."

This script runs N games against the local bomberman env with a chosen
manager (heuristic / policy / hybrid) controlling agent 0, and emits a
JSON report with per-round, per-component metrics. The intent is to
characterise *what* the local agent is scoring on so we can compare it
against any per-game data we manage to extract from official Eval URLs.

Usage:

    # Heuristic only (no torch needed at runtime).
    python training/ae/diagnose.py --manager heuristic --games 12 --out diagnose-heuristic.json

    # Policy only (uses ae/models/bc.pt by default).
    python training/ae/diagnose.py --manager policy --games 12 \\
        --checkpoint training/ae/checkpoints/ppo.pt --out diagnose-policy.json

    # Hybrid (the new structural attempt).
    python training/ae/diagnose.py --manager hybrid --games 12 \\
        --checkpoint training/ae/checkpoints/ppo.pt --out diagnose-hybrid.json

Notes:
- Per-step reward delta is attributed to action_taken / agent_health
  delta / base_health delta / item-collection events (inferred from
  health/base/resource/bomb pool deltas plus reward sign).
- The output JSON is small and human-skimmable; aggregate medians and
  per-round means are pre-computed at the bottom.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "ae" / "src"))

from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


ACTION_NAMES = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STAY", "PLACE_BOMB"]
MAX_SCORE = 1000.0


def _obs_to_python(obs: Any) -> dict:
    out = {}
    for key, value in obs.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _load_manager(name: str, checkpoint: str | None):
    name = name.lower().strip()
    if name == "heuristic":
        from ae_manager import AEManager  # noqa: WPS433
        return AEManager(), "heuristic"
    if name == "policy":
        import os
        if checkpoint:
            os.environ["AE_POLICY_CHECKPOINT"] = checkpoint
        from policy_manager import PolicyAEManager  # noqa: WPS433
        return PolicyAEManager(), "policy"
    raise ValueError(f"unknown manager '{name}'; pick heuristic|policy")


def _classify_step(
    prev: dict,
    cur: dict,
    reward_delta: float,
    action: int | None,
) -> dict:
    """Best-effort attribution of one step's reward to a component.

    The env doesn't tell us *which* reward fired for our agent, so we
    infer from observable state deltas:
    - +5 around mission tile-step
    - +2 around resource tile-step (also +0.5 to team_resources)
    - +1 around recon tile-step
    - large negative ≈ -attack_damage / 1.0 (bomb hit, base destroyed)
    - +30 attack_kill (rare; appears as a +N spike from a single bomb tick)
    - +50 destroy_enemy_base / -50 own_base_destroyed (huge magnitude)
    """

    classification: dict[str, float] = {}

    def _val(d, key, default=0.0):
        x = d.get(key, default)
        if isinstance(x, list):
            x = x[0] if x else default
        if hasattr(x, "item"):
            x = x.item()
        try:
            return float(x)
        except Exception:
            return float(default)

    health_delta = _val(cur, "health") - _val(prev, "health")
    base_delta = _val(cur, "base_health") - _val(prev, "base_health")
    resources_delta = _val(cur, "team_resources") - _val(prev, "team_resources")
    bombs_delta = _val(cur, "team_bombs") - _val(prev, "team_bombs")

    # Big positive: collected something or destroyed enemy base / killed enemy.
    if reward_delta >= 49.0:
        classification["destroy_enemy_base"] = reward_delta
    elif reward_delta >= 14.0:
        classification["attack_kill_or_multi"] = reward_delta
    elif 4.5 <= reward_delta <= 5.5:
        classification["collect_mission"] = reward_delta
    elif 1.5 <= reward_delta <= 2.5:
        classification["collect_resource"] = reward_delta
    elif 0.5 <= reward_delta <= 1.5:
        classification["collect_recon"] = reward_delta
    elif 0 < reward_delta < 0.5:
        classification["attack_damage_positive"] = reward_delta

    # Negative: damage taken, base hit, etc.
    if reward_delta <= -49.0:
        classification["own_base_destroyed"] = reward_delta
    elif reward_delta < 0:
        if health_delta < 0:
            classification["self_damage"] = reward_delta
        elif base_delta < 0:
            classification["base_damage"] = reward_delta
        else:
            classification["other_negative"] = reward_delta

    return {
        "action": ACTION_NAMES[action] if action is not None and 0 <= action < 6 else None,
        "reward": reward_delta,
        "health_delta": health_delta,
        "base_delta": base_delta,
        "resources_delta": resources_delta,
        "bombs_delta": bombs_delta,
        "components": classification,
    }


def evaluate(args: argparse.Namespace) -> None:
    manager, mode = _load_manager(args.manager, args.checkpoint)
    print(f"manager: {mode}; games: {args.games}; novice: {args.novice}")

    cfg = default_config()
    cfg.env.novice = bool(args.novice)
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    our_agent = env.possible_agents[0]

    report: dict[str, Any] = {
        "mode": mode,
        "games": args.games,
        "novice": bool(args.novice),
        "rounds": [],
    }

    for round_idx in range(args.games):
        env.reset()
        # Reset our manager — the env is single-process here so we just
        # rebuild it the same way the server does at /reset.
        if hasattr(manager, "heuristic"):
            manager.heuristic._reset_memory()
            if hasattr(manager, "policy") and hasattr(manager.policy, "stacker"):
                manager.policy.stacker.reset()
                manager.policy._last_step = None
        elif hasattr(manager, "_reset_memory"):
            manager._reset_memory()
        elif hasattr(manager, "stacker"):
            manager.stacker.reset()
            manager._last_step = None

        prev_obs: dict | None = None
        action_counter: Counter[int] = Counter()
        component_totals: Counter[str] = Counter()
        steps: list[dict] = []
        visited: set[tuple[int, int]] = set()
        bombs_placed = 0
        freeze_ticks_seen = 0
        # PettingZoo AEC: env.last()[1] is the reward our agent earned
        # since its last turn. Accumulate that as the round total.
        cumulative_reward = 0.0

        start = time.time()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            elif agent == our_agent:
                cumulative_reward += float(reward)
                obs_py = _obs_to_python(observation)
                action = int(manager.ae(obs_py))
                action_counter[action] += 1
                if action == 5:
                    bombs_placed += 1
                if obs_py.get("frozen_ticks", 0) and int(obs_py["frozen_ticks"]) > 0:
                    freeze_ticks_seen += 1
                loc = obs_py.get("location")
                if isinstance(loc, list) and len(loc) == 2:
                    visited.add((int(loc[0]), int(loc[1])))

                # Per-turn reward attribution: `reward` is what we earned
                # between our previous action and the observation we're
                # about to act on.
                if prev_obs is not None and abs(float(reward)) > 1e-6:
                    step_info = _classify_step(prev_obs, obs_py, float(reward), action)
                    for component, val in step_info["components"].items():
                        component_totals[component] += val
                    if args.dump_steps:
                        steps.append(step_info)
                prev_obs = obs_py
            else:
                action = env.action_space(agent).sample()
            env.step(action)

        elapsed = time.time() - start
        round_reward = cumulative_reward
        round_score = round_reward / MAX_SCORE
        round_report = {
            "round": round_idx + 1,
            "score": round_score,
            "reward_total": round_reward,
            "elapsed_s": round(elapsed, 2),
            "action_counts": {ACTION_NAMES[a]: c for a, c in sorted(action_counter.items())},
            "bombs_placed": bombs_placed,
            "freeze_ticks_seen": freeze_ticks_seen,
            "unique_cells_visited": len(visited),
            "reward_components": dict(component_totals),
        }
        if args.dump_steps:
            round_report["steps"] = steps
        report["rounds"].append(round_report)
        print(
            f"  round {round_idx+1}: score={round_score:.3f}  reward={round_reward:.1f}  "
            f"time={elapsed:.1f}s  bombs={bombs_placed}  cells={len(visited)}  "
            f"components={dict(component_totals)}"
        )

    env.close()

    scores = [r["score"] for r in report["rounds"]]
    component_sum: Counter[str] = Counter()
    for r in report["rounds"]:
        for k, v in r["reward_components"].items():
            component_sum[k] += v
    summary = {
        "mean_score": statistics.fmean(scores) if scores else 0.0,
        "median_score": statistics.median(scores) if scores else 0.0,
        "stdev_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "reward_component_sum": dict(component_sum),
    }
    report["summary"] = summary

    print()
    print(f"== summary ==")
    print(f"  mean={summary['mean_score']:.3f}  median={summary['median_score']:.3f}  "
          f"stdev={summary['stdev_score']:.3f}  range=[{summary['min_score']:.3f}, {summary['max_score']:.3f}]")
    print(f"  reward components (sum across all rounds):")
    for k, v in sorted(component_sum.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:24}  {v:+8.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  wrote: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manager", choices=["heuristic", "policy"], default="heuristic")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a torch checkpoint for policy/hybrid mode.")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--novice", type=int, default=1, help="1=novice fixed map, 0=varied")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--dump-steps", action="store_true",
                        help="Dump per-step records into the report (large output).")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
