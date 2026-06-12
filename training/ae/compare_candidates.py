"""Compare PPO candidate checkpoints with the 24x3-seed gate protocol.

Why this exists: validate_cloud_suite.py only knows --our heuristic or
--our hybrid. The hybrid path goes through ae/src/policy_manager.py
which (post the 21 May fixed-map-v5 restore) does NOT support the
legacy-small architecture our PPO checkpoints use. So we can't gate
PPO checkpoints with validate_cloud_suite directly without temporarily
re-adding legacy support to deployed ae/src.

This script bypasses that entirely by using train_ppo.py's existing
load_actor + evaluate_selection. Both honor the checkpoint's saved
model_arch tag via build_policy_network, so legacy-small loads cleanly.

Output: per-(checkpoint, seed) aggregate score + per-suite breakdown,
then a final tabular summary across all checkpoints.

Note: this evaluates the *pure policy* (not the hybrid wrapper). The
question we're answering here is "which of these checkpoints is better
as a policy", which is the right comparison for picking which to deploy
into the hybrid wrapper. The hybrid wrapper's contribution is roughly
independent of which PPO weights it wraps.
"""

from __future__ import annotations

# Pin PYTHONHASHSEED=0 before any other import so checkpoint comparisons are reproducible.
import os
import sys

if os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ae" / "src"))
sys.path.insert(0, str(REPO_ROOT / "training" / "ae"))

from train_ppo import evaluate_selection, load_actor  # noqa: E402


def make_args(checkpoint_path: Path, seed: int, games_per_suite: int) -> SimpleNamespace:
    """Synthesise the minimal argparse-shaped args object evaluate_selection
    and load_actor need. Values mirror the training defaults from
    run_full_rl_elo_v1.py so the eval matches what gating did during training."""
    return SimpleNamespace(
        bc_checkpoint=str(checkpoint_path),
        load_critic=False,
        load_optimizer=False,
        use_belief=False,
        n_frames=1,
        selection_suites="scripted,cloudsuite,pressure2",
        selection_weights="1,1,1",
        selection_games=games_per_suite,
        selection_device="cpu",
        eval_games=games_per_suite,
        eval_opponents="cloudsuite",
        novice=True,
        vary_maps=False,
        reward_scale=50.0,
        return_clip=10.0,
        seed=seed,
        eval_seed=seed * 100,
        games_per_update=1,  # not used by evaluate
        selection_manager="policy",  # pure-policy, not hybrid
        selection_fixed_map_shortcut=False,
        # Reward shaping flags evaluate doesn't use, but load_actor/construction sometimes touch.
        explore_bonus=0.0,
        explore_bonus_final=0.0,
        explore_horizon=200,
        health_delta_bonus=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to PPO checkpoints to compare",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 42, 1337],
        help="Seeds to evaluate at (one row per seed per checkpoint)",
    )
    parser.add_argument(
        "--games-per-suite",
        type=int,
        default=24,
        help="Games per suite per seed (matches validate_cloud_suite --rounds default)",
    )
    args_cli = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\nComparing {len(args_cli.checkpoints)} checkpoints x {len(args_cli.seeds)} seeds, "
          f"{args_cli.games_per_suite} games/suite, device={device}")
    print(f"{'=' * 110}\n")

    # checkpoint_label -> seed -> (aggregate, per-suite dict)
    results: dict[str, dict[int, tuple[float, dict[str, float]]]] = {}

    for ckpt_path in args_cli.checkpoints:
        label = Path(ckpt_path).name
        results[label] = {}
        print(f"--- {label} ---")
        for seed in args_cli.seeds:
            args = make_args(Path(ckpt_path), seed, args_cli.games_per_suite)
            actor, _use_belief, _warm, _ckpt = load_actor(args, device)
            actor.eval()
            agg, parts = evaluate_selection(actor, args, device)
            results[label][seed] = (agg, parts)
            print(f"  seed={seed}: agg={agg:.4f}  "
                  f"scripted={parts.get('scripted', 0):.4f} "
                  f"cloudsuite={parts.get('cloudsuite', 0):.4f} "
                  f"pressure2={parts.get('pressure2', 0):.4f}")
        aggs = [results[label][s][0] for s in args_cli.seeds]
        mean = sum(aggs) / len(aggs)
        spread = max(aggs) - min(aggs)
        print(f"  -> mean_agg={mean:.4f}  spread={spread:.4f}\n")

    print(f"\n{'=' * 110}")
    print("Comparison summary (per-seed aggregate, mean across seeds):")
    print(f"{'=' * 110}")
    header = f"{'checkpoint':<45} " + " ".join(f"seed={s:<6}" for s in args_cli.seeds) + " mean      cs_mean"
    print(header)
    for label in results:
        per_seed = " ".join(f"{results[label][s][0]:.4f}    " for s in args_cli.seeds)
        agg_mean = sum(results[label][s][0] for s in args_cli.seeds) / len(args_cli.seeds)
        cs_mean = sum(results[label][s][1].get("cloudsuite", 0) for s in args_cli.seeds) / len(args_cli.seeds)
        print(f"{label:<45} {per_seed} {agg_mean:.4f}    {cs_mean:.4f}")


if __name__ == "__main__":
    main()
