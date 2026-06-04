#!/usr/bin/env python3
"""Dir-2 v1 — BC warm-start + self-play league + dense ladder (AE).

Context (2026-06-04, ae/NOTES.md "Dir-2 RL pipeline").  The Stage-B confpol
line (confpol-semis2b-u75, cloud 0.414 on the 4-Jun eval) exhausted its
ceiling.  The ONE recipe we have never run is curry's actual Semifinals bet:
BC-clone warm-start -> short self-play LEAGUE -> deploy as confpol consultant.

Shaping recipe: `--preset full-rl` (the proven Stage-B shaping knobs:
  entropy floor 0.008 → 0.003 anneal, clip 0.14/0.08, target_kl 0.015,
  policy selection, load_critic=True, baseline_eval=True,
  min_save_improvement 0.015) PLUS `--opponent-mix-preset dir2-league`
  (the self-play-augmented fast mix below).  This mirrors exactly the pattern
  that produced the champion confpol-semis2b-u75 (`--preset full-rl
  --opponent-mix-preset semis-foreign`) but adds the self-play league layer.
  lr is kept at 1e-4 (gentle warm-start); full-rl does NOT override lr.

The key differences from Stage-B:
  * Warm-start from the BC clone (`dir2-bc.pt`) trained on curry's own play
    data.  BC gives a strong behavioral prior BEFORE any RL pressure — the
    hypothesis is that RL from a BC init escapes the ~0.60–0.63 consultant band
    that three RL-from-scratch runs all hit.
  * Opponent mix — `--opponent-mix-preset dir2-league`:
      foreign_train:0.45, selfplay:0.25, scripted:0.15, cloudsuite:0.15
    - `selfplay:0.25` is the critical NEW self-play league layer.  The `selfplay`
      mode in train_ppo's _make_opponents() is the ONLY mode (besides frozen/
      mixed/league) that constructs a FrozenPolicyOpponent from the SnapshotPool.
      Without this slice the pool is populated but NEVER drawn from — the run
      would be a plain foreign curriculum with no league layer at all.  We use
      `selfplay` (NOT `league`) deliberately: `league` also injects the slow
      PlannerOpponent + AggressivePlannerOpponent (the 3-4-day bottleneck); this
      run favours fast opponents only.
    - `foreign_train:0.45` keeps the proven Stage-B foreign curriculum (the
      champion confpol-semis2b-u75 came from it) AND provides the heuristic
      anchor: _foreign_train_blend includes self_heuristic (C+bomb7), so we do
      NOT need a separate slow-planner slice.  Note: the blend is 4 foreign
      TRAIN-OK opponents (curry_aggro, self_policy, evbot, self_heuristic) plus
      the local scripted `rusher` proxy — so NOT 5 foreign opponents; `rusher` is
      a fast local heuristic included for behavioral diversity, not a foreign team.
      FOREIGN_EVAL_ONLY names are NEVER sampled — train_ppo's
      _foreign_train_blend() hard-aborts if any leak in.
    - `scripted:0.15, cloudsuite:0.15` keep fast behavioral diversity.
    - Net: ~0.70 weight on (fast NN foreign + self-play snapshots), NO slow
      planner/aggressive slices.
  * Snapshot self-play (`--snapshot-interval 10`, `--snapshot-pool-size 8`):
    the SnapshotPool adds our OWN policy at various historical snapshots to
    the league mix via the `selfplay` slice above.  These are fast (NN inference
    only); the pool falls back to the live actor while it is still empty.
  * Gentle lr 1e-4, NOT the 2.5e-4 from-scratch value.  The blowup history
    (hitting a warm policy with 10x lr re-stochasticizes it in one update) means
    we refine, we don't shake.
  * Dense checkpoint ladder every 25 updates: cloud-best is always an EARLY rung
    (inverted-U), so we never want to rely on the final checkpoint only.

SAVE DISCIPLINE — same lesson as every prior run. The in-training save gate is
a single-seed proxy and is ANTI-correlated with cloud-consultant value past
early rungs. This launcher does NOT trust it for promotion. It writes a DENSE
LADDER (`dir2-v1-u<N>.pt`) and the real decision is a post-hoc cloud
variance-farm of several early/mid rungs. See "NEXT STEPS" at the end.

Run detached (CPU-bound; Metal/MPS crashed Stage-B; AE_FORCE_CPU=1 enforced):

    nohup caffeinate -is .venv/bin/python -u training/ae/run_dir2_v1.py \\
      > training/ae/checkpoints/dir2-v1.run.log 2>&1 &
    tail -f training/ae/checkpoints/dir2-v1.log   # watch training

    kill:  pkill -f run_dir2_v1.py
    (caffeinate keeps the Mac awake; a CLOSED LID still clamshell-sleeps.)

Pre-condition: `training/ae/checkpoints/dir2-bc.pt` must exist (the BC clone
checkpoint produced by collect_bc.py + train_bc.py).  If it is missing the
launcher exits with a clear error — do not start the RL league until BC is done.
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

# BC warm-start checkpoint (produced by train_bc.py; does not exist at authoring
# time — the launcher checks for it at runtime and fails fast if absent).
DEFAULT_WARMSTART = CKPT_DIR / "dir2-bc.pt"

# Calibrated from Stage-B: ~533 agent-acted steps per full-episode game.
# The semis-foreign mix has NO slow planner opponents, so the per-game wall-clock
# should be FASTER than Stage-B (which mixed in curry_aggro A* planning).
# Re-calibrate from the first smoke update if materially different.
STEPS_PER_GAME = 533

# Run "as many steps as possible in the time window" — same open-ended philosophy
# as confpol-native. The sweet spot has always been in the early rungs; kill when
# the cloud/melee-best plateau is confirmed. ~2 days of CPU wall-clock at 16g/u.
TARGET_STEPS = 5_000_000


def updates_for(target_steps: int, games_per_update: int) -> int:
    return max(1, round(target_steps / (games_per_update * STEPS_PER_GAME)))


def run(cmd: list[str], log: Path) -> None:
    env = dict(os.environ)
    env["AE_FORCE_CPU"] = "1"           # Metal/MPS crashed Stage-B; CPU is correct
    env.setdefault("PYTHONHASHSEED", "0")
    print("\n$ " + " ".join(cmd) + f"\n  (log -> {log})\n", flush=True)
    with open(log, "a") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"command failed (rc={proc.returncode}); see {log}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--warmstart", default=str(DEFAULT_WARMSTART),
                    help="BC clone checkpoint to warm-start actor+critic from")
    ap.add_argument("--tag", default="dir2-v1",
                    help="output tag; all rungs written as <tag>-u<N>.pt")
    ap.add_argument("--games-per-update", type=int, default=16,
                    help="rollout games collected per PPO update")
    ap.add_argument("--ppo-epochs", type=int, default=3,
                    help="PPO minibatch passes per update")
    ap.add_argument("--lr", default="1e-4",
                    help="gentle refinement lr; do NOT raise toward 2.5e-4")
    ap.add_argument("--updates", type=int, default=None,
                    help="total PPO updates; default sized to TARGET_STEPS")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="unconditional -u<N> ladder every N updates (CRITICAL for farming)")
    ap.add_argument("--snapshot-interval", type=int, default=10,
                    help="add own-policy snapshot to self-play pool every N updates")
    ap.add_argument("--snapshot-pool-size", type=int, default=8,
                    help="max historical self-play snapshots kept (FIFO)")
    args = ap.parse_args()

    warm = Path(args.warmstart)
    if not warm.exists():
        raise SystemExit(
            f"BC warm-start checkpoint missing: {warm}\n"
            f"Wait for collect_bc.py + train_bc.py to finish before launching the RL league."
        )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out = CKPT_DIR / f"{args.tag}.pt"           # best-by-eval (gate only; not for direct promotion)
    latest = CKPT_DIR / f"{args.tag}-latest.pt"  # most recent
    log = CKPT_DIR / f"{args.tag}.log"

    n = args.updates or updates_for(TARGET_STEPS, args.games_per_update)

    cmd = [
        sys.executable, "-u", str(TRAIN),
        # Stage-B shaping: entropy floor/anneal, clip 0.14/0.08, target_kl 0.015,
        # policy selection, load_critic=True, baseline_eval=True,
        # min_save_improvement=0.015.  apply_preset("full-rl") leaves
        # opponent_mix_preset alone when it is already non-"none", so the
        # dir2-league mix below is honoured without being overridden.
        "--preset", "full-rl",
        # Self-play-augmented fast mix (registered preset in train_ppo.py):
        #   foreign_train:0.45 + selfplay:0.25 + scripted:0.15 + cloudsuite:0.15.
        # The `selfplay` slice is what makes this a real league — it draws from
        # the SnapshotPool that --snapshot-interval populates.  FAST opponents
        # only: no planner/aggressive/league slices.
        "--opponent-mix-preset", "dir2-league",
        # Self-play snapshots: the selfplay slice above draws from this pool.
        "--snapshot-interval", str(args.snapshot_interval),
        "--snapshot-pool-size", str(args.snapshot_pool_size),
        # BC warm-start (actor always; critic avoids cold value bootstrap).
        # --load-critic and --baseline-eval are already set by --preset full-rl.
        "--bc-checkpoint", str(warm),
        # Gentle refinement lr; full-rl does NOT override lr.
        "--lr", str(args.lr),
        "--games-per-update", str(args.games_per_update),
        "--ppo-epochs", str(args.ppo_epochs),
        "--gamma", "0.99", "--gae-lambda", "0.95",
        "--n-frames", "4",
        "--eval-every", str(args.eval_every),
        # Dense ladder — the cloud-best rung is always early; we must have it.
        "--checkpoint-every", str(args.checkpoint_every),
        "--updates", str(n),
        "--out", str(out), "--latest-out", str(latest),
        "--seed", "0",
    ]

    print(f"Dir-2 v1 BC warm-start league  tag={args.tag}")
    print(f"  warm-start : {warm.name}")
    print(f"  preset     : full-rl  opponent-mix-preset: dir2-league")
    print(f"  snapshots  : every {args.snapshot_interval} updates, pool size {args.snapshot_pool_size}")
    print(f"  lr={args.lr}  gamma=0.99  gae_lambda=0.95")
    print(f"  updates={n}  (~{n * args.games_per_update * STEPS_PER_GAME:,} steps,"
          f" target {TARGET_STEPS:,})")
    print(f"  checkpoint ladder every {args.checkpoint_every} updates -> {args.tag}-u<N>.pt")
    print(f"  AE_FORCE_CPU=1 (enforced in env; Metal/MPS crashed Stage-B)\n")

    run(cmd, log)

    print(f"""
================ DONE — DO NOT PROMOTE ON THE IN-TRAINING GATE ================
Local/in-training eval is a single-seed proxy and is ANTI-correlated with
cloud-consultant value past early rungs (confpol-native confirmed this pattern).
The real decision is a cloud variance-farm of several ladder rungs.

Candidates (the dense ladder + endpoints):
  ladder : {args.tag}-u*.pt   <- FARM THESE (esp. the EARLY rungs: u25, u50, u75, u100)
  latest : {latest}
  best   : {out}   (best-by-single-seed eval; informative only — do not promote directly)

NEXT STEPS (in order):
  1. Copy 3-4 early/mid ladder rungs to ae/models/bc.pt and do a canary:
       cp training/ae/checkpoints/{args.tag}-u<N>.pt ae/models/bc.pt
       AE_MODE=confidence_policy_hybrid til test ae
     Confirm the log says "AE policy loaded ... epoch=<N>", NOT "falling back".
  2. Cloud variance-farm n>=5 per rung via variance_farm.py.
  3. Re-farm the shipped incumbent (confpol-semis2b-u75, cloud 0.414) to n>=5
     in the SAME window — apples-to-apples comparison.
  4. Promote ONLY if a rung clears the incumbent by a visible margin (>0.05).
     Sub-0.05 cloud effects are unresolvable (project_ae_cloud_power).

  Incumbent deploy: confpol-semis2b-u75 @ cloud 0.414 (new 4-Jun eval baseline).
================================================================================
""")


if __name__ == "__main__":
    main()
