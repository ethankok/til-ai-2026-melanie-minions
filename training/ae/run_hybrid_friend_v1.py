"""PPO training run targeted at the "hybrid PPO ~0.7" recipe a teammate's
friend reported on 24 May 2026, with the lessons from our prior attempts baked in.

Differences vs run_full_rl_v1 / run_full_rl_elo_v1:

  * Warm-starts from `deployed-bc-v1.pt` (the proven BC) rather than from
    the prior PPO checkpoint. The friend's hint was "further training
    regressed to 0.5", which suggests the right artefact is an early
    PPO checkpoint near the BC manifold, not the deeply-trained PPO.

  * PYTHONHASHSEED=0 is pinned in the subprocess env so the within-process
    evals during training are reproducible (avoids the 0.10+ swing the
    24 May evening calibration uncovered).

  * Eval cadence tightened to every 3 updates (vs 10 in elo-v1) so we
    catch the peak fast. With 60 updates that's 20 eval points.

  * Save floor = baseline eval + 0.005 (vs 0.015 elsewhere). Lower bar
    so peak-then-regress trajectories actually deposit a checkpoint;
    later attempts that fail to beat the captured peak leave --out alone,
    which is exactly the behavior we want.

  * 60 updates total, ~2 min/update on Workbench ≈ 2h wall.

Outputs:
  base in  : training/ae/checkpoints/deployed-bc-v1.pt
  out      : training/ae/checkpoints/hybrid-friend-v1.pt       (peak-gate)
  latest   : training/ae/checkpoints/hybrid-friend-v1-latest.pt (every step)
  log      : training/ae/checkpoints/hybrid-friend-v1.log

After training, compare snapshot quality at PYTHONHASHSEED=0:

  PYTHONHASHSEED=0 .venv/bin/python training/ae/compare_candidates.py \\
      --checkpoints training/ae/checkpoints/hybrid-friend-v1.pt \\
                    training/ae/checkpoints/hybrid-friend-v1-latest.pt \\
                    training/ae/checkpoints/ppo-full-rl-v1.pt

Deploy the winner as hybrid (cloud's actual proven mode):
  - copy chosen .pt to ae/models/bc.pt
  - ensure ae/Dockerfile has  ENV AE_MODE=hybrid
  - til build ae hybrid-friend-vfN
  - til submit ae hybrid-friend-vfN
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "training" / "ae" / "checkpoints"
BASE_CHECKPOINT = CHECKPOINT_DIR / "deployed-bc-v1.pt"
REFERENCE_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-v1.pt"
OUT_CHECKPOINT = CHECKPOINT_DIR / "hybrid-friend-v1.pt"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "hybrid-friend-v1-latest.pt"
LOG_PATH = CHECKPOINT_DIR / "hybrid-friend-v1.log"


def main() -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not BASE_CHECKPOINT.exists():
        print(f"Missing base checkpoint: {BASE_CHECKPOINT}", file=sys.stderr)
        print(
            "Expected deployed-bc-v1.pt in training/ae/checkpoints/. "
            "Restore from ~/ae-checkpoints-backup/ on Workbench if needed.",
            file=sys.stderr,
        )
        return 2

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "training" / "ae" / "train_ppo.py"),
        "--preset", "full-rl",
        "--bc-checkpoint", str(BASE_CHECKPOINT),
        "--out", str(OUT_CHECKPOINT),
        "--latest-out", str(LATEST_CHECKPOINT),
        "--updates", "60",
        "--n-frames", "1",
        "--eval-every", "3",
        "--selection-manager", "policy",
        "--selection-device", "cpu",
        "--baseline-eval",
        "--min-save-improvement", "0.005",
        "--selection-games", "24",
        "--seed", "488",
        "--eval-seed", "48800",
        "--snapshot-interval", "5",
        "--snapshot-pool-size", "8",
    ]
    if REFERENCE_CHECKPOINT.exists():
        cmd.extend(["--reference-checkpoint", str(REFERENCE_CHECKPOINT)])

    env = os.environ.copy()
    # Pin Python hash seed so the within-process eval scores are reproducible
    # and not drawn from the "unlucky hash" tail of the AEManager-bearing
    # opponent code paths.
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("Running hybrid-friend-v1 PPO training:")
    print(" ".join(cmd))
    print(f"\nCheckpoint: {OUT_CHECKPOINT}")
    print(f"Latest:     {LATEST_CHECKPOINT}")
    print(f"Log:        {LOG_PATH}\n")

    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
