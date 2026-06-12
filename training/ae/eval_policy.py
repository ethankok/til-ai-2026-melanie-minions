"""Evaluate a trained AE policy locally against the bomberman env.

Runs N games with the policy controlling agent 0. Opponent pool is
selectable via ``--opponents`` (same options as train_ppo.py) so we can
A/B different checkpoints against the same opponent distribution.

Reads ``n_frames`` from the checkpoint and configures the FrameStacker
accordingly, so the same script works for both single-frame and
frame-stacked policies.

Usage:
    # pressure-heavy scripted eval (default)
    python training/ae/eval_policy.py --checkpoint training/ae/checkpoints/bc.pt --games 6

    # apples-to-apples against the ppo-v1-era 'mixed' baseline
    python training/ae/eval_policy.py --checkpoint training/ae/checkpoints/ppo-selfplay-v1.pt --opponents mixed --games 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(1, str(REPO_ROOT / "ae" / "src"))

from encoder import FrameStacker, rasterize_belief  # noqa: E402
from model import PolicyNetwork, build_policy_network  # noqa: E402
from ae_manager import AEManager  # noqa: E402

from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402

# Reuse train_ppo's opponent factories so eval-time and train-time
# opponent distributions match exactly.
from train_ppo import OPPONENT_MODES, _make_opponents, _random_opponent  # noqa: E402


NUM_ROUNDS_DEFAULT = 6
MAX_SCORE = 1000.0


def _obs_to_python(obs) -> dict:
    out = {}
    for key, value in obs.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _belief_for_eval(planner: AEManager, obs_py: dict, use_belief: bool):
    if not use_belief:
        return None
    step = planner._as_int(obs_py.get("step"), default=(planner.last_step or 0) + 1)
    if planner.last_step is None or step == 0 or step < planner.last_step:
        planner._reset_memory()
    planner._age_bombs(step)
    planner._blast_cache = {}
    planner.last_step = step
    location = planner._location(obs_py.get("location"))
    direction = planner._as_int(obs_py.get("direction"), default=0) % 4
    planner._update_memory(obs_py, step, location, direction)
    return rasterize_belief(planner, obs_py)


def _action_from_policy(model: PolicyNetwork, stacker: FrameStacker, planner: AEManager,
                        obs: dict, device: torch.device, use_belief: bool, greedy: bool = True) -> int:
    belief = _belief_for_eval(planner, obs, use_belief)
    stacked = stacker.observe(obs, belief_map=belief)
    agent_v = torch.from_numpy(stacked["agent_view"]).to(device)
    base_v = torch.from_numpy(stacked["base_view"]).to(device)
    scalars = torch.from_numpy(stacked["scalars"]).to(device)
    mask = torch.from_numpy(stacked["action_mask"]).to(device)
    belief_t = torch.from_numpy(belief).to(device) if belief is not None else None
    return model.select_action(
        agent_v, base_v, scalars, action_mask=mask, greedy=greedy, belief_map=belief_t,
    )


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_frames = int(checkpoint.get("n_frames", 1))
    use_belief = bool(checkpoint.get("use_belief", False))
    print(
        f"checkpoint: epoch={checkpoint.get('epoch')}, "
        f"val_acc={checkpoint.get('val_acc')}, n_frames={n_frames}, use_belief={use_belief}"
    )

    state_dict = checkpoint["model_state_dict"]
    model = build_policy_network(
        n_frames=n_frames,
        use_belief=use_belief,
        state_dict=state_dict,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"policy_arch={getattr(model, 'model_arch', 'default')}")

    config = default_config()
    config.env.novice = args.novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=config)
    our_agent = env.possible_agents[0]
    rewards = {a: 0.0 for a in env.possible_agents}
    print(f"opponents: {args.opponents}")

    start = time.time()
    for game in trange(args.games, desc="games"):
        env.reset()
        stacker = FrameStacker(n_frames)
        planner = AEManager()
        # snapshot_pool=None: frozen opponents deepcopy the model under eval,
        # giving a stable "play against your shadow" baseline.
        opponents = _make_opponents(
            model, device, args.opponents,
            [a for a in env.possible_agents if a != our_agent],
            n_frames,
            snapshot_pool=None,
        )
        for op in opponents.values():
            if hasattr(op, "reset"):
                op.reset()
        for agent in env.agent_iter():
            obs, _reward, termination, truncation, _info = env.last()
            for a in env.agents:
                rewards[a] += float(env.rewards[a])
            if termination or truncation:
                env.step(None)
                continue
            obs_py = _obs_to_python(obs)
            if agent == our_agent:
                action = _action_from_policy(
                    model, stacker, planner, obs_py,
                    device, use_belief, greedy=args.greedy,
                )
            else:
                action = int(opponents.get(agent, _random_opponent)(env, agent, obs_py))
                # Fall back to a legal action if the opponent returns an illegal one.
                mask = np.asarray(obs_py.get("action_mask", [1, 1, 1, 1, 1, 1]), dtype=np.float32).reshape(-1)
                if action < 0 or action >= mask.size or not bool(mask[action]):
                    legal = np.flatnonzero(mask > 0)
                    action = int(legal[0]) if legal.size else int(env.action_space(agent).sample())
            env.step(action)
    env.close()
    elapsed = time.time() - start

    score = rewards[our_agent] / args.games / MAX_SCORE
    print(f"\ntotal rewards: {rewards[our_agent]:.1f}")
    print(f"score: {score:.4f}")
    print(f"elapsed: {elapsed:.1f}s ({elapsed / args.games:.1f}s/game)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/ae/checkpoints/bc.pt")
    parser.add_argument("--games", type=int, default=NUM_ROUNDS_DEFAULT)
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--no-novice", dest="novice", action="store_false")
    parser.add_argument("--stochastic", dest="greedy", action="store_false")
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument(
        "--opponents",
        choices=OPPONENT_MODES,
        default="cloudsuite",
        help="Opponent distribution. 'cloudsuite' is the default pressure-heavy "
             "scripted proxy suite; 'pressure2' is the harder stress suite; "
             "'mixed' and 'league' include planner/frozen-policy opponents; "
             "'random' is retained only for legacy sanity checks.",
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
