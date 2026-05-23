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

from encoder import FrameStacker, rasterize_belief  # noqa: E402
from opponents import MixedOpponent, OpponentFn, make_opponent  # noqa: E402


def _resolve_opponent_names(spec: str) -> list[str]:
    """Mirror of simulate.run_simulation()'s opponent resolution.

    Keeps `collect_bc.py` independent of simulate.py while matching its
    naming conventions ('random', 'mixed', 'library', 'cloudsuite', or an
    explicit comma-separated list).
    """

    if spec == "random":
        return ["random"] * 5
    if spec == "mixed":
        return ["mixed"] * 5
    if spec == "library":
        return ["random", "greedy", "bomber", "defender", "hunter"]
    if spec == "cloudsuite":
        return ["rusher", "hunter", "bomber", "defender", "mixed"]
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if len(names) == 1:
        names = names * 5
    if len(names) != 5:
        raise ValueError(f"need 5 opponent names (got {len(names)}): {names}")
    return names


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
    with_belief: bool = True,
    opponents_spec: str = "random",
) -> None:
    """Collect a BC dataset from planner-v3b rollouts.

    When ``with_belief=True`` (default), each sample also includes the
    rasterized belief tensor at the planner's step. Setting it False
    keeps the file size down for legacy single-frame BC training.

    ``opponents_spec`` controls what the other 5 agents do during data
    collection. Use 'random' (legacy) to reproduce the old single-game
    distribution, or one of 'mixed' / 'library' / 'cloudsuite' / a
    5-comma-separated list to expose the planner to the same opponent
    mix the qualifier is likely to use. Mixing here makes the BC dataset
    cover a wider state distribution, which is how we avoid the
    bc-belief-hybrid failure mode (overfit to planner-vs-random).
    """
    config = default_config()
    config.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=config)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])

    names = _resolve_opponent_names(opponents_spec)
    opponent_seed = seed if seed is not None else 0
    opponents: list[OpponentFn] = [
        make_opponent(n, seed=opponent_seed + 1000 + i) for i, n in enumerate(names)
    ]
    print(
        f"Controlling {our_agent} of {env.possible_agents}; "
        f"n_frames={n_frames}; with_belief={with_belief}; "
        f"opponents_spec={opponents_spec} -> {names}"
    )

    agent_views: list[np.ndarray] = []
    base_views: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    beliefs: list[np.ndarray] = []
    actions: list[int] = []

    start = time.time()
    for game in trange(games, desc="games"):
        if seed is not None:
            env.reset(seed=seed + game)
        else:
            env.reset()
        # Per-game reset for opponents that need it (Mixed picks a fresh
        # archetype each game; AEManager-based opponents zero their belief).
        for op in opponents:
            if hasattr(op, "reset_for_game"):
                op.reset_for_game()
            if hasattr(op, "_reset_memory"):
                op._reset_memory()
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
                # Run the planner — this updates planner.* belief state
                # AND picks the action we'll BC against.
                action = planner.ae(obs_py)
                belief = rasterize_belief(planner, obs_py) if with_belief else None
                stacked = stacker.observe(obs_py, belief_map=belief)
                agent_views.append(stacked["agent_view"])
                base_views.append(stacked["base_view"])
                scalars.append(stacked["scalars"])
                action_masks.append(stacked["action_mask"])
                if with_belief:
                    beliefs.append(belief)
                actions.append(int(action))
            else:
                obs_py = _obs_to_python(obs)
                slot = other_ids.index(agent)
                op = opponents[slot]
                try:
                    action = int(op(obs_py))
                except Exception:
                    action = env.action_space(agent).sample()
                # Safety: if the opponent returned an illegal action, fall
                # back to the first legal one (mirrors simulate.py).
                mask = obs_py.get("action_mask")
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

    env.close()
    elapsed = time.time() - start
    print(
        f"Collected {len(actions):,} samples across {games} games in {elapsed:.1f}s "
        f"({len(actions) / max(elapsed, 1):.1f} samples/sec)"
    )

    if not actions:
        raise SystemExit("No samples collected — env probably terminated immediately?")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        agent_views=np.stack(agent_views).astype(np.float32),
        base_views=np.stack(base_views).astype(np.float32),
        scalars=np.stack(scalars).astype(np.float32),
        action_masks=np.stack(action_masks).astype(np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        n_frames=np.asarray(n_frames, dtype=np.int32),
        with_belief=np.asarray(int(with_belief), dtype=np.int32),
    )
    if with_belief:
        save_kwargs["beliefs"] = np.stack(beliefs).astype(np.float32)
    np.savez_compressed(out_path, **save_kwargs)
    counts = np.bincount(actions, minlength=6)
    labels = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STAY", "PLACE_BOMB"]
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Shapes: agent_views={agent_views[0].shape}, base_views={base_views[0].shape}, "
          f"scalars={scalars[0].shape}"
          + (f", beliefs={beliefs[0].shape}" if with_belief else ""))
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
    parser.add_argument("--no-belief", dest="with_belief", action="store_false",
                        help="Skip belief-map rasterization (legacy single-frame BC)")
    parser.add_argument(
        "--opponents", type=str, default="random",
        help=(
            "Opponent set used for the other 5 agents during data collection. "
            "Same vocabulary as simulate.py: 'random' (legacy), 'mixed', "
            "'library', 'cloudsuite', or 5 comma-separated names."
        ),
    )
    parser.set_defaults(with_belief=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    collect_dataset(
        args.games,
        out_path,
        novice=args.novice,
        seed=args.seed,
        n_frames=args.n_frames,
        with_belief=args.with_belief,
        opponents_spec=args.opponents,
    )


if __name__ == "__main__":
    main()
