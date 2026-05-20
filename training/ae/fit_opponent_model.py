"""Fit a simple action distribution model for cloud-like opponents.

Tier 2 #8.

The MCTS in `ae_manager.py` doesn't simulate opponent moves directly — it
searches over our own action sequences and uses opponent positions as
fixed input to the score function. Predictive bombing (Tier 1 #7) does
estimate the probability that an enemy walks into a blast over the bomb
timer, but currently uses a hand-tuned `0.30 - 0.06 * d` formula.

This script replaces that hand-tuned formula with one fit to actual
opponent behavior, by simulating large numbers of rounds and recording
each enemy's per-step action distribution. The output is a small JSON:

    {
      "by_opponent_name": {
        "random":   {"FORWARD": 0.18, "BACKWARD": 0.17, ...},
        "greedy":   {...},
        ...
      },
      "expected_walk_distance_per_step": {
        "random":   0.42,
        "greedy":   0.65,
        ...
      },
      "weighted_mix": {            # weighted mean across cloud-like mix
        "FORWARD": 0.21, ...
      }
    }

The AE manager reads ``expected_walk_distance_per_step["weighted_mix"]``
to scale the predictive-walk hit probability:

    P(hit blast cell within N steps) ≈ 1 - (1 - p_step) ** (N * walk_dist)

where ``p_step`` is the probability that a random walk lands on a specific
adjacent cell each step. With expected_walk_distance ~= 0.5 (mostly
movement) the random-walk approximation underestimates real opponents.

We DON'T plug this into MCTS rollouts because that would slow planning;
the manager just uses the resulting expected-walk distance as a scalar
multiplier on its predictive-walk credit.

Usage:

    python training/ae/fit_opponent_model.py \
        --inputs training/ae/data/sim-random.npz \
                 training/ae/data/sim-library.npz \
        --out ae/models/opponent_model.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_AE = REPO_ROOT / "training" / "ae"
if str(TRAINING_AE) not in sys.path:
    sys.path.insert(0, str(TRAINING_AE))

from opponents import OPPONENT_NAMES  # noqa: E402


ACTION_NAMES = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STAY", "PLACE_BOMB"]


def fit_simple(rounds: int, opponents_spec: str, seed: int) -> dict:
    """Run rounds and observe what each opponent slot does, per-step."""

    # Local import to keep this module importable without the heavy env.
    from simulate import _make_our_agent  # noqa: WPS433

    from til_environment import bomberman_env
    from til_environment.config import default_config

    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)

    # Resolve opponents spec — exactly the same logic as simulate.py but
    # this script doesn't need trajectory logging for our own agent.
    if opponents_spec == "random":
        names = ["random"] * 5
    elif opponents_spec == "library":
        names = ["random", "greedy", "bomber", "defender", "hunter"]
    elif opponents_spec == "cloudsuite":
        names = ["rusher", "hunter", "bomber", "defender", "mixed"]
    elif opponents_spec == "mixed":
        names = ["mixed"] * 5
    else:
        names = [n.strip() for n in opponents_spec.split(",") if n.strip()]
        if len(names) == 1:
            names *= 5

    from opponents import make_opponent  # noqa: WPS433
    opponents = [make_opponent(n, seed=seed + 1000 + i) for i, n in enumerate(names)]
    our_agent = _make_our_agent("heuristic")

    by_name_counts: dict[str, Counter] = defaultdict(Counter)
    by_name_walks: dict[str, list[int]] = defaultdict(list)

    for r in range(rounds):
        env.reset()
        if hasattr(our_agent, "_reset_memory"):
            our_agent._reset_memory()
        for op in opponents:
            if hasattr(op, "reset_for_game"):
                op.reset_for_game()
            if hasattr(op, "_reset_memory"):
                op._reset_memory()

        agent_id_us = env.possible_agents[0]
        other_ids = list(env.possible_agents[1:])

        # Track per-opponent last position so we can count actual movements.
        last_positions: dict[str, tuple[int, int] | None] = {a: None for a in other_ids}

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                continue

            obs_native = {
                k: v if type(v) in (int, float) else (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in obs.items()
            }

            if agent == agent_id_us:
                action = int(our_agent.ae(obs_native))
            else:
                slot = other_ids.index(agent)
                op_name = names[slot]
                op = opponents[slot]
                action = int(op(obs_native))
                # Mask check.
                mask = obs_native.get("action_mask")
                if mask is not None:
                    try:
                        if not int(mask[action]):
                            for i, m in enumerate(mask):
                                if int(m):
                                    action = i
                                    break
                    except Exception:
                        pass
                by_name_counts[op_name][action] += 1
                # Movement detection: compare current vs last logged position.
                pos = obs_native.get("location")
                if pos is not None:
                    cur = (int(pos[0]), int(pos[1]))
                    prev = last_positions[agent]
                    if prev is not None:
                        dist = abs(cur[0] - prev[0]) + abs(cur[1] - prev[1])
                        by_name_walks[op_name].append(dist)
                    last_positions[agent] = cur

            env.step(action)

        if (r + 1) % max(1, rounds // 5) == 0:
            print(f"[{r+1}/{rounds}] fitted opponent traces", flush=True)

    env.close()

    by_name = {}
    walk_dists = {}
    for op_name, counts in by_name_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        dist = {ACTION_NAMES[a]: counts[a] / total for a in range(6)}
        by_name[op_name] = dist
        walks = by_name_walks.get(op_name, [])
        walk_dists[op_name] = float(np.mean(walks)) if walks else 0.0

    # Weighted mix — even weights across the named opponents observed.
    if by_name:
        weighted = {a: 0.0 for a in ACTION_NAMES}
        for d in by_name.values():
            for a in ACTION_NAMES:
                weighted[a] += d.get(a, 0.0)
        for a in ACTION_NAMES:
            weighted[a] /= len(by_name)
        weighted_walk = float(np.mean(list(walk_dists.values())))
    else:
        weighted = {a: 1 / 6 for a in ACTION_NAMES}
        weighted_walk = 0.5

    return {
        "rounds": rounds,
        "opponents": names,
        "by_opponent_name": by_name,
        "expected_walk_distance_per_step": walk_dists,
        "weighted_mix": weighted,
        "weighted_walk_distance": weighted_walk,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--opponents", type=str, default="library",
                   help="opponent spec — see simulate.py")
    p.add_argument("--out", type=Path, default=Path("ae/models/opponent_model.json"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    model = fit_simple(args.rounds, args.opponents, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2))
    print(f"opponent model -> {args.out}")
    print(f"weighted_walk_distance = {model['weighted_walk_distance']:.3f}")
    print("by-opponent walk distance:")
    for name, dist in model["expected_walk_distance_per_step"].items():
        print(f"  {name:>10}  {dist:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
