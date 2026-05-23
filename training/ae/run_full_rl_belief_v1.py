"""Full belief-aware PPO training pipeline.

End-to-end:
  1. Collect a mixed-opponent BC dataset with belief tensors enabled.
     The legacy `bc-belief-hybrid` failed because its BC dataset was
     planner-vs-random — the policy overfit a weak local distribution.
     This launcher collects against a wider opponent mix (default
     library: random+greedy+bomber+defender+hunter) so the BC reader
     sees state-distribution closer to the cloud's mixed pool.
  2. Train belief-aware BC on that dataset.
  3. Launch PPO with --preset full-rl --use-belief from the new BC,
     gated against `ppo-full-rl-v1.pt` if present, with the same save
     floor (`min_save_improvement=0.015`).

All artifacts live under training/ae/{data,checkpoints}/, all gitignored.
Logs stream both to stdout and to
training/ae/checkpoints/ppo-full-rl-belief-v1.log.

Run on the Mac (the rollout loop is faster on MPS than Workbench CUDA
per the run_full_rl_v1.py benchmark notes):

    .venv/bin/python -u training/ae/run_full_rl_belief_v1.py

Skip stages that already produced their artifact via flags below.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "training" / "ae"
DATA_DIR = TRAIN_DIR / "data"
CHECKPOINT_DIR = TRAIN_DIR / "checkpoints"

BC_DATASET = DATA_DIR / "bc-belief-mixed-v1.npz"
BC_CHECKPOINT = CHECKPOINT_DIR / "bc-belief-mixed-v1.pt"
REFERENCE_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-v1.pt"
OUT_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-belief-v1.pt"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "ppo-full-rl-belief-v1-latest.pt"
LOG_PATH = CHECKPOINT_DIR / "ppo-full-rl-belief-v1.log"


def _stream_subprocess(cmd: list[str], log_handle, env: dict | None = None) -> int:
    """Run cmd, mirroring stdout to terminal + log file. Returns exit code."""

    print(">>>", " ".join(cmd), flush=True)
    log_handle.write(">>> " + " ".join(cmd) + "\n")
    log_handle.flush()
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
        sys.stdout.write(line)
        sys.stdout.flush()
        log_handle.write(line)
        log_handle.flush()
    return proc.wait()


def collect_bc(args: argparse.Namespace, log_handle) -> int:
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_DIR / "collect_bc.py"),
        "--games", str(args.bc_games),
        "--out", str(BC_DATASET),
        "--n-frames", "1",
        "--seed", str(args.bc_seed),
        "--opponents", args.bc_opponents,
    ]
    return _stream_subprocess(cmd, log_handle)


def train_bc(args: argparse.Namespace, log_handle) -> int:
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_DIR / "train_bc.py"),
        "--data", str(BC_DATASET),
        "--out", str(BC_CHECKPOINT),
        "--epochs", str(args.bc_epochs),
        "--batch-size", "256",
    ]
    return _stream_subprocess(cmd, log_handle)


def train_ppo(args: argparse.Namespace, log_handle) -> int:
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_DIR / "train_ppo.py"),
        "--preset", "full-rl",
        "--use-belief",
        "--bc-checkpoint", str(BC_CHECKPOINT),
        "--out", str(OUT_CHECKPOINT),
        "--latest-out", str(LATEST_CHECKPOINT),
        "--updates", str(args.updates),
        "--n-frames", "1",
        "--eval-every", "10",
        "--selection-manager", "policy",
        "--selection-device", "cpu",
        "--baseline-eval",
        "--min-save-improvement", "0.015",
        "--selection-games", "24",
        "--seed", str(args.ppo_seed),
        "--eval-seed", str(args.ppo_seed * 100),
    ]
    if REFERENCE_CHECKPOINT.exists():
        cmd.extend(["--reference-checkpoint", str(REFERENCE_CHECKPOINT)])
        log_handle.write(
            f"using reference checkpoint {REFERENCE_CHECKPOINT}\n"
        )
    else:
        log_handle.write(
            f"reference checkpoint missing at {REFERENCE_CHECKPOINT}; "
            "PPO will gate only against the BC starting checkpoint.\n"
        )

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    return _stream_subprocess(cmd, log_handle, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Skip BC dataset collection (use existing %(default)s)" % {"default": BC_DATASET},
    )
    parser.add_argument(
        "--skip-bc", action="store_true",
        help=f"Skip BC training (use existing {BC_CHECKPOINT})",
    )
    parser.add_argument(
        "--skip-ppo", action="store_true",
        help="Run only BC collection and training, skip PPO. Useful for "
             "previewing BC convergence before committing to the long run.",
    )
    parser.add_argument(
        "--bc-games", type=int, default=600,
        help="Games to roll out for the BC dataset",
    )
    parser.add_argument(
        "--bc-opponents", type=str, default="library",
        help=(
            "Opponent spec for BC collection. 'library' covers "
            "random/greedy/bomber/defender/hunter. Use 'cloudsuite' for "
            "rusher/hunter/bomber/defender/mixed if you want a "
            "harder distribution. Use 'mixed' for fully randomized."
        ),
    )
    parser.add_argument("--bc-seed", type=int, default=4242)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--updates", type=int, default=240)
    parser.add_argument("--ppo-seed", type=int, default=4881)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("Belief-aware AE PPO pipeline")
    print(f"  bc dataset    : {BC_DATASET}")
    print(f"  bc checkpoint : {BC_CHECKPOINT}")
    print(f"  ppo out       : {OUT_CHECKPOINT}")
    print(f"  ppo latest    : {LATEST_CHECKPOINT}")
    print(f"  reference     : {REFERENCE_CHECKPOINT} (exists={REFERENCE_CHECKPOINT.exists()})")
    print(f"  log           : {LOG_PATH}\n")

    with LOG_PATH.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"args: {vars(args)}\n")

        if not args.skip_collect:
            if BC_DATASET.exists():
                print(f"NOTE: {BC_DATASET} already exists; pass --skip-collect to skip rerunning.\n")
            rc = collect_bc(args, log_handle)
            if rc != 0:
                print(f"BC collection failed with code {rc}", file=sys.stderr)
                return rc
        elif not BC_DATASET.exists():
            print(f"--skip-collect set but {BC_DATASET} missing", file=sys.stderr)
            return 2

        if not args.skip_bc:
            rc = train_bc(args, log_handle)
            if rc != 0:
                print(f"BC training failed with code {rc}", file=sys.stderr)
                return rc
        elif not BC_CHECKPOINT.exists():
            print(f"--skip-bc set but {BC_CHECKPOINT} missing", file=sys.stderr)
            return 2

        if args.skip_ppo:
            print("\n--skip-ppo set; stopping after BC.")
            return 0

        rc = train_ppo(args, log_handle)
        if rc != 0:
            print(f"PPO training failed with code {rc}", file=sys.stderr)
            return rc

    print("\nAll stages complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
