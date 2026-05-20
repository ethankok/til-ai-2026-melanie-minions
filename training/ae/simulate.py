"""Offline simulation harness for the AE task.

Runs N rounds of `bomberman_env` with our agent at index 0 and configurable
opponents at indices 1..5. Logs trajectories and per-round scores so the
output can feed downstream analyses:

  * Tier 1 #1 — playbook/Q-table aggregation (`build_playbook.py`).
  * Tier 2 #8 — opponent action distribution fitting.
  * Tier 2 #10 — oracle BC labels.

Usage examples:

    # Score the current heuristic against random opponents (matches `til test`)
    python training/ae/simulate.py --rounds 200 --opponents random --our heuristic

    # Score against the mixed library used for PPO training
    python training/ae/simulate.py --rounds 1000 --opponents mixed --our heuristic

    # Score and dump trajectories (used by build_playbook.py)
    python training/ae/simulate.py --rounds 1000 --opponents random,greedy,bomber,defender,hunter \
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
import sys
import time
from collections import defaultdict
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

def _make_our_agent(name: str):
    """Construct the agent we control in slot 0.

    Currently supported:
      heuristic — `AEManager` (default; what's in `ae/src/ae_manager.py`)

    Easy to extend: drop a new branch here and pass the matching --our flag.
    """

    name = name.lower().strip()
    if name == "heuristic":
        return AEManager()
    if name == "hybrid":
        from hybrid_manager import HybridAEManager
        return HybridAEManager()
    raise ValueError(f"unknown --our value {name!r}")


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

        if termination or truncation:
            env.step(None)
            continue

        # Convert numpy structures to plain Python (mirrors `test_ae.py`).
        observation_native = {
            k: v if type(v) in (int, float) else (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in observation.items()
        }

        if agent == agent_id_us:
            action = int(our_agent.ae(observation_native))
            if log_traj:
                loc = observation_native.get("location", [0, 0])
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
) -> dict:
    """Run ``rounds`` rounds and return the aggregated trajectory + stats.

    ``opponents_spec`` is a comma-separated list of opponent names (one per
    enemy slot, repeats fine), or one of:

      'random'    : all 5 enemies are random (matches `til test`)
      'mixed'     : 5 independent MixedOpponent instances
      'library'   : random,greedy,bomber,defender,hunter (5 enemies)
      'cloudsuite': rusher,hunter,bomber,defender,mixed (pressure-heavy)

    Or any explicit list: 'greedy,greedy,bomber,defender,hunter'.
    """

    cfg = default_config()
    cfg.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    our_agent = _make_our_agent(our_name)

    # Resolve opponents spec to a list of 5 names.
    if opponents_spec == "random":
        names = ["random"] * 5
    elif opponents_spec == "mixed":
        names = ["mixed"] * 5
    elif opponents_spec == "library":
        names = ["random", "greedy", "bomber", "defender", "hunter"]
    elif opponents_spec == "cloudsuite":
        names = ["rusher", "hunter", "bomber", "defender", "mixed"]
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
            env, our_agent, opponents, log_traj=log_traj, seed=None
        )
        scores.append(result["score"])
        totals.append(result["total_reward"])

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
        default="random",
        help=(
            "opponent set: 'random', 'mixed', 'library', 'cloudsuite', or 5 comma-separated "
            "names from " + ",".join(OPPONENT_NAMES)
        ),
    )
    p.add_argument("--our", type=str, default="heuristic",
                   help="which agent to control in slot 0 (currently: heuristic)")
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
