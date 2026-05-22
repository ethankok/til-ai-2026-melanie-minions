"""Run the full fixed-Novice PPO training attempt.

This is a thin, reproducible launcher around ``training/ae/train_ppo.py``.
Run it from a terminal on the Mac so logs stream live and are also saved to:

    training/ae/checkpoints/ppo-full-rl-v1.log

The resulting checkpoint is written to:

    training/ae/checkpoints/ppo-full-rl-v1.pt

This run keeps Novice geometry fixed, varies rollout seeds, and samples a
stratified opponent mix every PPO update:

    10% random, 35% scripted, 35% cloudsuite,
    5% planner, 5% aggressive, 10% league/self-play snapshots
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "training" / "ae" / "checkpoints"
BASE_CHECKPOINT = CHECKPOINT_DIR / "ppo-qualifier-best-v1.pt"
OUT_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-v1.pt"
LOG_PATH = CHECKPOINT_DIR / "ppo-full-rl-v1.log"


def main() -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not BASE_CHECKPOINT.exists():
        print(f"Missing base checkpoint: {BASE_CHECKPOINT}", file=sys.stderr)
        print("Copy ppo-qualifier-best-v1.pt into training/ae/checkpoints/ first.", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "training" / "ae" / "train_ppo.py"),
        "--preset",
        "full-rl",
        "--bc-checkpoint",
        str(BASE_CHECKPOINT),
        "--out",
        str(OUT_CHECKPOINT),
        "--updates",
        "240",
        "--n-frames",
        "1",
        "--eval-every",
        "5",
        "--seed",
        "488",
        "--eval-seed",
        "48800",
    ]

    print("Running full RL PPO training:")
    print(" ".join(cmd))
    print(f"\nCheckpoint: {OUT_CHECKPOINT}")
    print(f"Log:        {LOG_PATH}\n")

    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
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
