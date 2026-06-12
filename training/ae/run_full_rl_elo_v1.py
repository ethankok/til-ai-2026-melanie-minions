"""Run the full fixed-Novice PPO training attempt WITH Elo population.

24 May 2026 experiment. Mirrors run_full_rl_v1.py but:
  * warm-starts from `ppo-full-rl-v1.pt` (the current 0.638 cloud high)
    instead of the older `ppo-qualifier-best-v1.pt` — we want Elo to
    improve on our best, not re-derive it.
  * adds `--elo-population` so frozen-snapshot opponents are sampled
    via Gaussian matchmaking on live Elo (arxiv 2407.00662 style).
  * 150 updates instead of 240 to fit a 6-8h overnight window.
  * Different output path so the existing ppo-full-rl-v1.pt is preserved
    untouched while training runs.

Reads/writes:
  base in : training/ae/checkpoints/ppo-full-rl-v1.pt  (the proven high)
  out     : training/ae/checkpoints/ppo-full-rl-elo-v1.pt
  latest  : training/ae/checkpoints/ppo-full-rl-elo-v1-latest.pt
  log     : training/ae/checkpoints/ppo-full-rl-elo-v1.log

Save floor is automatically set to the warm-start's eval score + 0.015 by
`--baseline-eval --min-save-improvement 0.015`, so a checkpoint is only
emitted to `--out` if it beats the warm-start by at least 0.015 weighted
eval. The `-latest.pt` file gets every checkpoint regardless of gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "training" / "ae" / "checkpoints"
BASE_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-v1.pt"
REFERENCE_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-v1.pt"
OUT_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-elo-v1.pt"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-elo-v1-latest.pt"
LOG_PATH = CHECKPOINT_DIR / "ppo-full-rl-elo-v1.log"


def main() -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not BASE_CHECKPOINT.exists():
        print(f"Missing base checkpoint: {BASE_CHECKPOINT}", file=sys.stderr)
        print("Need ppo-full-rl-v1.pt in training/ae/checkpoints/ first.", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "training" / "ae" / "train_ppo.py"),
        "--preset", "full-rl",
        "--bc-checkpoint", str(BASE_CHECKPOINT),
        "--out", str(OUT_CHECKPOINT),
        "--latest-out", str(LATEST_CHECKPOINT),
        "--updates", "150",
        "--n-frames", "1",
        "--eval-every", "10",
        "--selection-manager", "policy",
        "--selection-device", "cpu",
        "--baseline-eval",
        "--min-save-improvement", "0.015",
        "--selection-games", "24",
        "--seed", "488",
        "--eval-seed", "48800",
        # sigma=200 = moderate matchmaking; K=32 = chess default;
        # baseline=0.55 ~= heuristic aggregate, so above-baseline plays push Elo up.
        "--elo-population",
        "--elo-sigma", "200",
        "--elo-k", "32",
        "--elo-initial", "1200",
        "--elo-baseline", "0.55",
        # Promote to pool more often so Elo has diverse opponents early.
        "--snapshot-interval", "5",
        "--snapshot-pool-size", "8",
    ]
    if REFERENCE_CHECKPOINT.exists() and REFERENCE_CHECKPOINT != BASE_CHECKPOINT:
        cmd.extend(["--reference-checkpoint", str(REFERENCE_CHECKPOINT)])

    print("Running full-RL Elo population PPO training:")
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
