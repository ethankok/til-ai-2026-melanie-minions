"""Evaluate a trained AE policy locally against the bomberman env.

Runs N games with the policy controlling agent 0 and random opponents.
Reads `n_frames` from the checkpoint and configures the FrameStacker
accordingly, so the same script works for both single-frame and
frame-stacked policies.

Usage:
    python training/ae/eval_policy.py --checkpoint training/ae/checkpoints/bc.pt --games 6
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

from encoder import FrameStacker  # noqa: E402
from model import PolicyNetwork  # noqa: E402

from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


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


def _action_from_policy(model: PolicyNetwork, stacker: FrameStacker, obs: dict,
                        device: torch.device, greedy: bool = True) -> int:
    stacked = stacker.observe(obs)
    agent_v = torch.from_numpy(stacked["agent_view"]).to(device)
    base_v = torch.from_numpy(stacked["base_view"]).to(device)
    scalars = torch.from_numpy(stacked["scalars"]).to(device)
    mask = torch.from_numpy(stacked["action_mask"]).to(device)
    return model.select_action(agent_v, base_v, scalars, action_mask=mask, greedy=greedy)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_frames = int(checkpoint.get("n_frames", 1))
    print(
        f"checkpoint: epoch={checkpoint.get('epoch')}, "
        f"val_acc={checkpoint.get('val_acc')}, n_frames={n_frames}"
    )

    model = PolicyNetwork(n_frames=n_frames).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    config = default_config()
    config.env.novice = args.novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=config)
    our_agent = env.possible_agents[0]
    rewards = {a: 0.0 for a in env.possible_agents}

    start = time.time()
    for game in trange(args.games, desc="games"):
        env.reset()
        stacker = FrameStacker(n_frames)
        for agent in env.agent_iter():
            obs, _reward, termination, truncation, _info = env.last()
            for a in env.agents:
                rewards[a] += float(env.rewards[a])
            if termination or truncation:
                env.step(None)
                continue
            if agent == our_agent:
                action = _action_from_policy(model, stacker, _obs_to_python(obs), device, greedy=args.greedy)
            else:
                action = env.action_space(agent).sample()
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
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
