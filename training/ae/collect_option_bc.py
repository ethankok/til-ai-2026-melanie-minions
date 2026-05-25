"""Collect option-policy BC data from heuristic rollouts.

The saved label is an 8-way strategic option derived from the planner's
decision reason, not the raw Bomberman action. This gives PPO a stable
warm-start: the actor first learns when to rush, defend, collect, hunt, or
escape, while the runtime planner still turns options into safe actions.
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
AE_SRC = REPO_ROOT / "ae" / "src"
for path in (THIS_DIR, AE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ae_manager import AEManager  # noqa: E402
from encoder import FrameStacker, rasterize_belief  # noqa: E402
from opponents import make_opponent, resolve_opponent_spec  # noqa: E402
from option_policy import NUM_OPTIONS, OPTION_NAMES, option_from_manager  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


DEFAULT_SUITES = [
    "strong_realistic",
    "base_rush_exploit",
    "bracket_proxy",
    "top_seed_proxy",
    "cloudsuite",
    "pressure2",
]


def _obs_to_python(obs) -> dict:
    out = {}
    for key, value in obs.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _make_opponents(spec: str, seed: int) -> list:
    names = resolve_opponent_spec(spec)
    return [make_opponent(name, seed=seed + 1000 + i) for i, name in enumerate(names)]


def collect_dataset(
    games: int,
    out_path: Path,
    suite_cycle: list[str],
    novice: bool = True,
    seed: int | None = None,
    n_frames: int = 4,
    with_belief: bool = True,
) -> None:
    cfg = default_config()
    cfg.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])

    print(
        f"Collecting option BC for {our_agent}; games={games}; "
        f"n_frames={n_frames}; with_belief={with_belief}; "
        f"suite_cycle={suite_cycle}"
    )

    agent_views: list[np.ndarray] = []
    base_views: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    beliefs: list[np.ndarray] = []
    actions: list[int] = []
    options: list[int] = []
    decisions: list[str] = []
    suite_labels: list[str] = []

    start = time.time()
    base_seed = seed if seed is not None else 0
    for game in trange(games, desc="games"):
        game_seed = base_seed + game
        if seed is not None:
            env.reset(seed=game_seed)
        else:
            env.reset()
        suite = suite_cycle[game % len(suite_cycle)]
        opponents = _make_opponents(suite, game_seed)
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

            obs_py = _obs_to_python(obs)
            if agent == our_agent:
                if obs_py.get("step") == 0:
                    planner = AEManager()
                    stacker.reset()
                action = int(planner.ae(obs_py))
                option = int(option_from_manager(planner, action=action))
                belief = rasterize_belief(planner, obs_py) if with_belief else None
                stacked = stacker.observe(obs_py, belief_map=belief)

                agent_views.append(stacked["agent_view"])
                base_views.append(stacked["base_view"])
                scalars.append(stacked["scalars"])
                if with_belief:
                    beliefs.append(belief)
                actions.append(action)
                options.append(option)
                decisions.append(str(getattr(planner, "last_decision", "")))
                suite_labels.append(suite)
            else:
                slot = other_ids.index(agent)
                op = opponents[slot]
                try:
                    action = int(op(obs_py))
                except Exception:
                    action = env.action_space(agent).sample()
                mask = obs_py.get("action_mask")
                if mask is not None:
                    try:
                        if not int(mask[action]):
                            for i, allowed in enumerate(mask):
                                if int(allowed):
                                    action = i
                                    break
                    except Exception:
                        pass
            env.step(action)

    env.close()
    elapsed = time.time() - start
    if not options:
        raise SystemExit("No samples collected.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "agent_views": np.stack(agent_views).astype(np.float32),
        "base_views": np.stack(base_views).astype(np.float32),
        "scalars": np.stack(scalars).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "options": np.asarray(options, dtype=np.int64),
        "decisions": np.asarray(decisions),
        "suites": np.asarray(suite_labels),
        "option_names": np.asarray(OPTION_NAMES),
        "n_frames": np.asarray(n_frames, dtype=np.int32),
        "with_belief": np.asarray(int(with_belief), dtype=np.int32),
    }
    if with_belief:
        save_kwargs["beliefs"] = np.stack(beliefs).astype(np.float32)
    np.savez_compressed(out_path, **save_kwargs)

    counts = np.bincount(np.asarray(options, dtype=np.int64), minlength=NUM_OPTIONS)
    print(
        f"Collected {len(options):,} samples across {games} games in {elapsed:.1f}s "
        f"({len(options) / max(elapsed, 1.0):.1f} samples/sec)"
    )
    print(f"Saved -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Option distribution:")
    for name, count in zip(OPTION_NAMES, counts):
        print(f"  {name:16s} {int(count):>7d}  ({count / len(options) * 100:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--out", default="training/ae/data/option_bc.npz")
    parser.add_argument("--opponents", default="strong_realistic",
                        help="single opponent suite/name when --suite-cycle is omitted")
    parser.add_argument("--suite-cycle", nargs="+", default=None,
                        help="cycle through these opponent suites, one suite per game")
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--no-novice", dest="novice", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--no-belief", dest="with_belief", action="store_false")
    parser.set_defaults(with_belief=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    suite_cycle = args.suite_cycle if args.suite_cycle else [args.opponents]
    collect_dataset(
        games=args.games,
        out_path=out_path,
        suite_cycle=suite_cycle,
        novice=args.novice,
        seed=args.seed,
        n_frames=args.n_frames,
        with_belief=args.with_belief,
    )


if __name__ == "__main__":
    main()
