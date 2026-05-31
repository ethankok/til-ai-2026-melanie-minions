#!/usr/bin/env python3
"""confpol-native launcher — confidence-gated raw-policy PPO (AE).

Context (31 May 2026, ae/NOTES.md "NEXT BIG BET"). The Pandemonium CNN-PPO
policy is a poor *global controller* (pure/hybrid cloud ~0.51 < heuristic ~0.59)
but a useful *consultant*: gated behind the heuristic on its low-confidence
ticks (AE_MODE=confidence_policy_hybrid), the u860 checkpoint farmed mean 0.634
(n=13) — the first AE config with a farmed mean above the ~0.59 bar.

The smoking gun for THIS run: the SAME extra training (u860 -> u1400) moved the
PURE policy UP (0.507 -> 0.550) but the CONFPOL wrapper DOWN (0.634 -> 0.600).
Opposite responses => the policy is trained as a global controller but deployed
as a consultant on the heuristic's hard ticks — a train/deploy mismatch. This
launcher fixes the mismatch: train the raw CNN policy ON the deployed
distribution. During rollouts the confidence gate acts (heuristic owns confident
ticks; the policy acts AND collects transitions ONLY on low-confidence ticks),
and the save-gate scores the wrapped confpol policy — see train_ppo.py
`collect_rollouts_gated` / `--confidence-gated`.

Design choices baked in here:
  * Warm-start from u860 (the best known consultant), actor + critic.
  * GENTLE refinement, not a reset: lr 1e-4 (vs phase-1's 2.5e-4). The phase-2
    self-play blowup proved that hitting a warm policy with 10x lr + entropy
    bonus re-stochasticizes it (greedy eval 0.72 -> 0.34 in one update) and
    never recovers. We refine, we don't shake.
  * full-rl opponent MIX (scripted/cloudsuite/pressure2/aggressive/league) as
    the best available proxy for the cloud distribution. full-rl floors entropy
    at 0.008 but does NOT touch --lr, so the gentle lr survives.
  * Deploy-default gate (eps=5.0, floor=10.0, override_target_none=1) so the
    ticks the policy trains on == the ticks confpol hands it at deploy.

SAVE DISCIPLINE — the hard-won lesson. Local eval is ANTI-correlated with
cloud-consultant value past ~u860 (u1400 had higher local eval but lower confpol
cloud). So this launcher does NOT trust the in-training gate for promotion. It
writes a DENSE LADDER of unconditional periodic checkpoints (--checkpoint-every)
and the real decision is a post-hoc cloud variance-farm of several rungs. The
previous run's fatal gap was keeping no intermediates; this one keeps them.

Run detached (owns the machine; multi-day, CPU-bound on opponent planners):

    PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=1 \\
      nohup caffeinate -is .venv/bin/python -u \\
      training/ae/run_confpol_native.py > \\
      training/ae/checkpoints/confpol-native.run.log 2>&1 &
    tail -f training/ae/checkpoints/confpol-native.log   # watch training

  kill:  pkill -f run_confpol_native.py
  (caffeinate keeps the Mac awake, but a CLOSED LID still clamshell-sleeps.)
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

# Warm-start: the best known consultant (farmed confpol mean 0.634, n=13).
DEFAULT_WARMSTART = CKPT_DIR / "pandemonium-v1-best-u860.pt"

# Gated rollouts log a transition only on low-conf ticks (~40% of our agent's
# acted steps in the smoke), but the ENVIRONMENT still runs full episodes, so
# wall-clock per update tracks the ungated run. STEPS_PER_GAME here is the count
# of LOGGED (low-conf) transitions per game, used only for the step-budget print.
STEPS_PER_GAME = 210  # ~0.40 * 533 (calibrated from the 31 May gated smoke)
TARGET_STEPS = 10_000_000  # "as many steps as possible"; kill when the farm plateaus


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
    ap.add_argument("--games-per-update", type=int, default=16)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--lr", default="1e-4", help="gentle refinement; do NOT raise toward 2.5e-4")
    ap.add_argument("--updates", type=int, default=None,
                    help="override; default sized to TARGET_STEPS")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="dense ladder of unconditional checkpoints for cloud farming")
    ap.add_argument("--conf-margin-epsilon", type=float, default=5.0)
    ap.add_argument("--conf-top-floor", type=float, default=10.0)
    ap.add_argument("--conf-override-target-none", type=int, default=1)
    ap.add_argument("--warmstart", default=str(DEFAULT_WARMSTART))
    ap.add_argument("--tag", default="confpol-native")
    args = ap.parse_args()

    warm = Path(args.warmstart)
    if not warm.exists():
        raise SystemExit(f"warm-start checkpoint missing: {warm}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out = CKPT_DIR / f"{args.tag}.pt"           # best-by-gated-eval (NOT for promotion)
    latest = CKPT_DIR / f"{args.tag}-latest.pt"  # most recent
    log = CKPT_DIR / f"{args.tag}.log"

    n = args.updates or updates_for(TARGET_STEPS, args.games_per_update)

    cmd = [
        sys.executable, "-u", str(TRAIN),
        "--preset", "full-rl",            # opponent mix + shaping; does not touch --lr
        "--confidence-gated",             # train == deploy: policy learns only low-conf ticks
        "--conf-margin-epsilon", str(args.conf_margin_epsilon),
        "--conf-top-floor", str(args.conf_top_floor),
        "--conf-override-target-none", str(args.conf_override_target_none),
        "--games-per-update", str(args.games_per_update),
        "--ppo-epochs", str(args.ppo_epochs),
        "--gamma", "0.99", "--gae-lambda", "0.95",
        "--lr", str(args.lr),
        "--n-frames", "4",
        "--eval-every", str(args.eval_every),
        "--checkpoint-every", str(args.checkpoint_every),
        "--bc-checkpoint", str(warm), "--load-critic",
        "--baseline-eval",
        "--updates", str(n),
        "--out", str(out), "--latest-out", str(latest),
        "--seed", "0",
    ]

    print(f"confpol-native (confidence-gated raw-policy PPO)  tag={args.tag}")
    print(f"  warm-start : {warm.name}  (best consultant, confpol farmed 0.634)")
    print(f"  lr={args.lr}  gate eps={args.conf_margin_epsilon} floor={args.conf_top_floor}")
    print(f"  updates={n}  (~{n*args.games_per_update*STEPS_PER_GAME:,} logged steps, "
          f"target {TARGET_STEPS:,}; env runs full episodes regardless)")
    print(f"  checkpoint ladder every {args.checkpoint_every} updates -> {args.tag}-u<N>.pt")

    run(cmd, log)

    print(f"""
================ DONE — DO NOT PROMOTE ON THE IN-TRAINING GATE ================
Local eval is anti-correlated with cloud-consultant value past ~u860. The real
decision is a cloud variance-farm of several ladder rungs, NOT the best-by-eval.

Candidates (the dense ladder + endpoints):
  ladder : {args.tag}-u*.pt   <- FARM THESE (esp. the EARLY/MID rungs)
  latest : {latest}
  best   : {out}   (best-by-gated-eval; informative only)

NEXT STEPS (in order):
  1. Pick 3-4 ladder rungs (early, mid, late). For each, deploy under
     AE_MODE=confidence_policy_hybrid (copy to ae/models/bc.pt; canary the
     `til test` log for "AE policy loaded ... epoch=<N>", NOT "falling back").
  2. Cloud variance-farm n>=5 each via variance_farm.py add/summary.
  3. Re-farm the shipped heuristic (tether/C+bomb7) to n>=5 in the SAME window
     so the comparison is apples-to-apples (the standing 0.634 is only "+0.035,
     p=0.30" vs the incumbent because the incumbent is under-sampled at n=3).
  4. Promote a rung ONLY if it clears confpol-u860's 0.634 by a visible margin.
     Reminder (project_ae_cloud_power): sub-0.05 cloud effects are unresolvable.

  Meanwhile confpol-u860 (0.634) remains the deploy floor — independent of this.
==============================================================================
""")


if __name__ == "__main__":
    main()
