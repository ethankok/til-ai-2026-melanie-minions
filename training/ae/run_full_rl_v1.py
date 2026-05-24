"""Run the full fixed-Novice PPO training attempt.

This is a thin, reproducible launcher around ``training/ae/train_ppo.py``.
Run it from a terminal on the Mac so logs stream live and are also saved to:

    training/ae/checkpoints/ppo-full-rl-v1.log

The resulting checkpoint is written to:

    training/ae/checkpoints/ppo-full-rl-v1.pt

This run keeps Novice geometry fixed, varies rollout seeds, and samples a
stratified opponent mix every PPO update:

    35% scripted, 35% cloudsuite, 15% pressure2,
    5% planner, 5% aggressive, 5% league/self-play snapshots

Selection scores use the pure learned policy. This is intentionally stricter
than the deployed hybrid wrapper: the run should only save a candidate if the
RL policy itself improves under the fixed-Novice pressure suites.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "training" / "ae" / "checkpoints"
BASE_CHECKPOINT = CHECKPOINT_DIR / "ppo-qualifier-best-v1.pt"
REFERENCE_CHECKPOINT = CHECKPOINT_DIR / "ppo-qualifier-best-v4-balanced.pt"
OUT_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-v1.pt"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-v1-latest.pt"
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
        "--latest-out",
        str(LATEST_CHECKPOINT),
        "--updates",
        "240",
        "--n-frames",
        "1",
        "--eval-every",
        "10",
        "--selection-manager",
        "policy",
        "--selection-device",
        "cpu",
        "--baseline-eval",
        "--min-save-improvement",
        "0.015",
        "--selection-games",
        "24",
        "--seed",
        "488",
        "--eval-seed",
        "48800",
    ]
    if REFERENCE_CHECKPOINT.exists():
        cmd.extend(["--reference-checkpoint", str(REFERENCE_CHECKPOINT)])
    else:
        print(
            f"Reference checkpoint not found: {REFERENCE_CHECKPOINT}\n"
            "The run will gate only against the starting checkpoint.",
            file=sys.stderr,
        )

    print("Running full RL PPO training:")
    print(" ".join(cmd))
    print(f"\nCheckpoint: {OUT_CHECKPOINT}")
    print(f"Latest:     {LATEST_CHECKPOINT}")
    print(f"Log:        {LOG_PATH}\n")

    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        env = os.environ.copy()
        env.setdefault("PYTHONHASHSEED", "0")
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
