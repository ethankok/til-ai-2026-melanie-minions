"""Collect (observation, action) pairs from planner-v3b for behavior cloning.

Runs the bomberman env with our planner controlling agent 0 and random
opponents for the rest. Every time the planner picks an action we record
the encoded observation and the action label, then save everything as a
single compressed npz at the end.

Usage:
    python training/ae/collect_bc.py --games 200 --out training/ae/data/bc.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import trange

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))                       # encoder.py
sys.path.insert(0, str(REPO_ROOT / "ae" / "src"))       # ae_manager.py

from ae_manager import AEManager
from til_environment import bomberman_env
from til_environment.config import default_config

from encoder import encode_observation


def _obs_to_python(obs) -> dict:
    """The planner expects Python-native types in the observation dict."""

    out = {}
    for key, value in obs.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def collect_dataset(
    games: int,
    out_path: Path,
    novice: bool = True,
    seed: int | None = None,
) -> None:
    config = default_config()
    config.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=config)
    our_agent = env.possible_agents[0]
    print(f"Controlling {our_agent} of {env.possible_agents}")

    agent_views: list[np.ndarray] = []
    base_views: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    actions: list[int] = []

    start = time.time()
    for game in trange(games, desc="games"):
        if seed is not None:
            env.reset(seed=seed + game)
        else:
            env.reset()
        planner = AEManager()

        for agent in env.agent_iter():
            obs, _reward, termination, truncation, _info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            if agent == our_agent:
                obs_py = _obs_to_python(obs)
                if obs_py.get("step") == 0:
                    planner = AEManager()
                action = planner.ae(obs_py)
                encoded = encode_observation(obs_py)
                agent_views.append(encoded["agent_view"])
                base_views.append(encoded["base_view"])
                scalars.append(encoded["scalars"])
                action_masks.append(encoded["action_mask"])
                actions.append(int(action))
            else:
                action = env.action_space(agent).sample()
            env.step(action)

    env.close()
    elapsed = time.time() - start
    print(
        f"Collected {len(actions):,} samples across {games} games in {elapsed:.1f}s "
        f"({len(actions) / max(elapsed, 1):.1f} samples/sec)"
    )

    if not actions:
        raise SystemExit("No samples collected — env probably terminated immediately?")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        agent_views=np.stack(agent_views).astype(np.float32),
        base_views=np.stack(base_views).astype(np.float32),
        scalars=np.stack(scalars).astype(np.float32),
        action_masks=np.stack(action_masks).astype(np.float32),
        actions=np.asarray(actions, dtype=np.int64),
    )
    counts = np.bincount(actions, minlength=6)
    labels = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STAY", "PLACE_BOMB"]
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Action distribution:")
    for label, count in zip(labels, counts):
        print(f"  {label:11s} {count:>7d}  ({count / len(actions) * 100:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200,
                        help="Number of full games to roll out (default 200)")
    parser.add_argument("--out", type=str, default="training/ae/data/bc.npz")
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--no-novice", dest="novice", action="store_false",
                        help="Disable novice mode (variable seed instead of fixed seed 88)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for env.reset (offset by game index)")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    collect_dataset(args.games, out_path, novice=args.novice, seed=args.seed)


if __name__ == "__main__":
    main()
