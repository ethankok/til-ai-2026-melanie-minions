#!/usr/bin/env python3
"""Pandemonium-scale CNN-PPO launcher (AE, from scratch).

Context (29 May 2026, ae/NOTES.md): every prior AE PPO line (v1, elo-v1,
belief-v1, friend-v1) capped at cloud ~0.41-0.47 vs the heuristic's ~0.57-0.60,
because the learned policy overfits the local opponent distribution (the
local<->cloud transfer gap). The ONE recipe we have never run is the one the
0.731 team (Pandemonium) described: CNN-over-viewcone + MLP, trained FROM
SCRATCH at ~10M Novice + ~5M self-play steps, with a rule fallback arbiter.

This launcher runs exactly that against the EXISTING infrastructure (the
`PolicyNetwork` CNN+MLP arch and `train_ppo.py` PPO loop already match the
Pandemonium spec). The only genuinely new variable is SCALE + from-scratch +
their hyperparameters (gamma 0.99, gae_lambda 0.95, ent_coef 0.01, lr decay).

Two phases:
  Phase 1 (Novice, mixed opponents, from scratch): orthogonal init, no warm
    start, `--preset full-rl` opponent mix on fixed Novice geometry. Target
    ~10M agent env-steps.
  Phase 2 (self-play fine-tune): warm-start from phase-1 latest (actor+critic),
    `--opponents selfplay` (pool-only historical self snapshots). Target ~5M.

Save discipline: train_ppo's in-training save-gate is a SINGLE-SEED proxy and
has historically overfit (the elo line predicted +0.028, cloud gave -0.21). So
this launcher does NOT trust the gate for promotion -- it writes `-latest.pt`
every update and the REAL promotion decision is a post-hoc n>=3 multi_seed_eval
(and ultimately a cloud variance-farm via variance_farm.py). See the printed
"NEXT STEPS" block at the end.

Deployment fallback: we already have a stronger arbiter than Pandemonium's BFS
-- the full AEManager heuristic, reachable via AE_MODE=hybrid / confidence_hybrid
(heuristic-veto). No separate BFS manager is built; the trained policy plugs into
the existing hybrid wrapper.

Compute: the rollout is CPU-bound on the scripted/heuristic opponents (the NN is
cheap), so MPS gives little speedup and a faithful 15M-step run is a multi-day
wall-clock job. Size phases with --phase1-updates / --phase2-updates after
reading STEPS_PER_GAME from a smoke run. Run detached:

    PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=1 \\
      .venv/bin/python -u training/ae/run_pandemonium_v1.py 2>&1 | tee \\
      training/ae/checkpoints/pandemonium-v1.log

Resume a crashed/paused run with --skip-phase1 (uses the existing phase-1
latest as the phase-2 warm-start).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "training" / "ae" / "checkpoints"
TRAIN = REPO_ROOT / "training" / "ae" / "train_ppo.py"

# STEPS_PER_GAME = OUR agent's acted timesteps per Novice episode (~episode_len).
# Updates needed for a step target = target_steps / (games_per_update * STEPS_PER_GAME).
STEPS_PER_GAME = 533

PHASE1_TARGET_STEPS = 10_000_000
PHASE2_TARGET_STEPS = 5_000_000


def updates_for(target_steps: int, games_per_update: int) -> int:
    return max(1, round(target_steps / (games_per_update * STEPS_PER_GAME)))


def run(cmd: list[str], log: Path) -> None:
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print("\n$ " + " ".join(cmd) + f"\n  (log -> {log})\n", flush=True)
    with open(log, "a") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"command failed (rc={proc.returncode}); see {log}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games-per-update", type=int, default=12)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--phase1-updates", type=int, default=None,
                    help="override; default sized to PHASE1_TARGET_STEPS")
    ap.add_argument("--phase2-updates", type=int, default=None,
                    help="override; default sized to PHASE2_TARGET_STEPS")
    ap.add_argument("--snapshot-interval", type=int, default=10)
    ap.add_argument("--snapshot-pool-size", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="Save an unconditional -u<update> ladder every N updates. "
                         "Critical: the cloud-best checkpoint is always EARLY "
                         "(inverted-U), so we farm the ladder, never the latest.")
    ap.add_argument("--skip-phase1", action="store_true")
    ap.add_argument("--tag", default="pandemonium-v1")
    args = ap.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    p1_out = CKPT_DIR / f"{args.tag}-phase1.pt"
    p1_latest = CKPT_DIR / f"{args.tag}-phase1-latest.pt"
    p2_out = CKPT_DIR / f"{args.tag}.pt"
    p2_latest = CKPT_DIR / f"{args.tag}-latest.pt"
    log = CKPT_DIR / f"{args.tag}.log"

    n1 = args.phase1_updates or updates_for(PHASE1_TARGET_STEPS, args.games_per_update)
    n2 = args.phase2_updates or updates_for(PHASE2_TARGET_STEPS, args.games_per_update)

    common = [
        sys.executable, "-u", str(TRAIN),
        "--games-per-update", str(args.games_per_update),
        "--ppo-epochs", str(args.ppo_epochs),
        "--gamma", "0.99", "--gae-lambda", "0.95",
        "--entropy-coef", "0.01", "--entropy-final-coef", "0.001",
        "--lr", "2.5e-4",
        "--n-frames", "4",
        "--snapshot-interval", str(args.snapshot_interval),
        "--snapshot-pool-size", str(args.snapshot_pool_size),
        "--eval-every", str(args.eval_every),
        "--checkpoint-every", str(args.checkpoint_every),  # dense -u<N> ladder
        "--baseline-eval",  # gate vs the warm-start/scratch eval baseline
    ]

    print(f"Pandemonium-scale CNN-PPO  tag={args.tag}")
    print(f"  STEPS_PER_GAME={STEPS_PER_GAME} (calibrate from smoke!)")
    print(f"  phase1: {n1} updates  (~{n1*args.games_per_update*STEPS_PER_GAME:,} steps, target {PHASE1_TARGET_STEPS:,})")
    print(f"  phase2: {n2} updates  (~{n2*args.games_per_update*STEPS_PER_GAME:,} steps, target {PHASE2_TARGET_STEPS:,})")

    if not args.skip_phase1:
        run(common + [
            "--preset", "full-rl",
            # non-existent path -> load_actor() falls back to from-scratch init
            "--bc-checkpoint", str(CKPT_DIR / "__force_from_scratch__.pt"),
            "--orthogonal-init",
            "--updates", str(n1),
            "--out", str(p1_out), "--latest-out", str(p1_latest),
            "--seed", "0",
        ], log)
    if not p1_latest.exists():
        raise SystemExit(f"phase-1 latest missing at {p1_latest}; cannot start phase 2")

    # Phase 2: self-play fine-tune, warm-start actor+critic from phase 1.
    run(common + [
        "--opponents", "selfplay", "--eval-opponents", "cloudsuite",
        "--bc-checkpoint", str(p1_latest), "--load-critic",
        "--updates", str(n2),
        "--out", str(p2_out), "--latest-out", str(p2_latest),
        "--seed", "1",
    ], log)

    print(f"""
================= DONE — DO NOT PROMOTE ON THE IN-TRAINING GATE =================
Candidates:
  phase1 latest : {p1_latest}
  phase2 latest : {p2_latest}   <- the Pandemonium-scale candidate
  phase2 gated  : {p2_out}      (only exists if it cleared the single-seed gate)

NEXT STEPS (the real gate — single-seed eval is known to overfit here):
  1. n>=3 multi-seed local eval of the candidate under the deployed hybrid
     wrapper (heuristic-veto), vs the C+bomb7 heuristic baseline:
       PYTHONHASHSEED=0 .venv/bin/python training/ae/multi_seed_eval.py ...
     If it does not at least TIE the heuristic locally, stop — do not submit.
  2. Only if it ties/beats locally: cloud variance-farm n>=5 (Workbench
     til build/submit), then log each result and decide with:
       python training/ae/variance_farm.py add --config pandemonium-v1 --acc <x>
       python training/ae/variance_farm.py compare --target pandemonium-v1 --baseline heuristic-A
  Reminder (project_ae_cloud_power): sub-0.05 cloud effects are unresolvable;
  this only matters if the policy clears the heuristic by a wide, visible margin.
================================================================================
""")


if __name__ == "__main__":
    main()
