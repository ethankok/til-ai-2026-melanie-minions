"""Collect frame-stacked (observation, action) pairs from planner-v3b for BC.

Runs the bomberman env with our planner controlling agent 0 and random
opponents for the rest. Each saved sample is the **4-frame stacked**
encoding at the planner's step, so the BC dataset matches the v2 policy's
expected input shape.

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
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "ae" / "src"))

from ae_manager import AEManager  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402

from encoder import FrameStacker  # noqa: E402


def _obs_to_python(obs) -> dict:
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
    n_frames: int = 4,
) -> None:
    config = default_config()
    config.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=config)
    our_agent = env.possible_agents[0]
    print(f"Controlling {our_agent} of {env.possible_agents}; n_frames={n_frames}")

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
        stacker = FrameStacker(n_frames)

        for agent in env.agent_iter():
            obs, _reward, termination, truncation, _info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            if agent == our_agent:
                obs_py = _obs_to_python(obs)
                if obs_py.get("step") == 0:
                    planner = AEManager()
                    stacker.reset()
                action = planner.ae(obs_py)
                stacked = stacker.observe(obs_py)
                agent_views.append(stacked["agent_view"])
                base_views.append(stacked["base_view"])
                scalars.append(stacked["scalars"])
                action_masks.append(stacked["action_mask"])
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
        n_frames=np.asarray(n_frames, dtype=np.int32),
    )
    counts = np.bincount(actions, minlength=6)
    labels = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STAY", "PLACE_BOMB"]
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Shapes: agent_views={agent_views[0].shape}, base_views={base_views[0].shape}, scalars={scalars[0].shape}")
    print("Action distribution:")
    for label, count in zip(labels, counts):
        print(f"  {label:11s} {count:>7d}  ({count / len(actions) * 100:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--out", type=str, default="training/ae/data/bc.npz")
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--no-novice", dest="novice", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=4)
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    collect_dataset(args.games, out_path, novice=args.novice, seed=args.seed, n_frames=args.n_frames)


if __name__ == "__main__":
    main()
