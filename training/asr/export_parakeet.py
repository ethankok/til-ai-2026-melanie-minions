"""Stage a fine-tuned Parakeet `.nemo` into `asr/models/` for the docker build.

Unlike the distil-whisper path (LoRA → merge → CT2 convert) Parakeet keeps
its native `.nemo` format end-to-end — there's nothing to convert. This
script just:

  1. Validates the input `.nemo` exists.
  2. Buffers the slang prompt in memory if it lives inside `--output-dir`
     (gotcha #8 from the whisper pipeline: the wipe loop deleted the slang
     file before copy-back if it lived inside the wipe target).
  3. Copies the trained `.nemo` to `asr/models/<filename>` atomically.
  4. Rewrites the slang prompt back if it was inside the wipe target.
  5. Verifies the output is loadable.

Usage::

    python training/asr/export_parakeet.py \\
        --trained-nemo training/asr/runs/parakeet-ft-v1/best.nemo \\
        --output-dir asr/models \\
        --filename parakeet-tdt-0.6b-v2.nemo \\
        --slang-file asr/models/slang_prompt.txt

After this runs successfully:

    til build asr parakeet-ft-v1
    til test  asr parakeet-ft-v1
    til submit asr parakeet-ft-v1     # only if local Eng-WER beats nemo-zs
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trained-nemo",
        type=Path,
        required=True,
        help="Fine-tuned .nemo from train_parakeet.py.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("asr/models"),
        help="Where the docker build expects to find the .nemo.",
    )
    ap.add_argument(
        "--filename",
        type=str,
        default="parakeet-tdt-0.6b-v2.nemo",
        help="Target filename inside --output-dir. For the TDT-v2 fine-tune "
             "path this stays parakeet-tdt-0.6b-v2.nemo; for the unified "
             "zero-shot A/B use download_models_nemo.py instead.",
    )
    ap.add_argument(
        "--slang-file",
        type=Path,
        default=Path("asr/models/slang_prompt.txt"),
        help="If this file exists, it's preserved across the export. If "
             "missing the export still succeeds (the manager handles a "
             "missing slang prompt gracefully).",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-copy load test. Saves ~30s of NeMo import time "
             "and ~2 GB of CPU mem; only use if you've already verified.",
    )
    args = ap.parse_args()

    if not args.trained_nemo.exists():
        raise SystemExit(f"Trained .nemo not found at {args.trained_nemo}")

    # Refuse to copy the zero-shot base over itself: that would silently
    # produce a "ft-v1" image bit-identical to nemo-zs and waste a
    # submission slot. The training script may have crashed before
    # writing best.nemo, in which case --trained-nemo points to the
    # base file (or doesn't exist at all).
    target = args.output_dir / args.filename
    if args.trained_nemo.resolve() == target.resolve():
        raise SystemExit(
            f"Refusing to overwrite {target} with itself. The training run "
            "probably crashed before producing best.nemo. Re-run "
            "train_parakeet.py and verify it printed 'Saving best model to "
            "...' before re-running this export."
        )
    if args.trained_nemo.stat().st_size < 100 * 1024 * 1024:
        # Real Parakeet .nemo is ~2.4 GB. Anything under 100 MB is almost
        # certainly an empty / partial / wrong file.
        raise SystemExit(
            f"Trained .nemo at {args.trained_nemo} is only "
            f"{args.trained_nemo.stat().st_size / (1024*1024):.1f} MB. "
            "That's too small to be a real Parakeet checkpoint; refusing "
            "to stage it."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Buffer slang prompt if it's about to be inside the same dir we're
    # writing into. Mirror of the whisper export gotcha.
    slang_bytes: bytes | None = None
    slang_target: Path | None = None
    if args.slang_file.exists():
        try:
            args.slang_file.relative_to(args.output_dir)
            slang_bytes = args.slang_file.read_bytes()
            slang_target = args.output_dir / args.slang_file.name
            print(
                f"Buffered slang prompt ({len(slang_bytes)} bytes) "
                f"because it lives in --output-dir.",
                flush=True,
            )
        except ValueError:
            # Slang file is outside output-dir; safe to leave alone.
            pass

    print(f"Copying {args.trained_nemo} -> {target} ...", flush=True)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    if tmp_target.exists():
        tmp_target.unlink()
    shutil.copyfile(args.trained_nemo, tmp_target)
    os.replace(tmp_target, target)

    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"OK: {target} ({size_mb:.1f} MB)", flush=True)

    if slang_bytes is not None and slang_target is not None:
        if not slang_target.exists() or slang_target.read_bytes() != slang_bytes:
            slang_target.write_bytes(slang_bytes)
            print(f"Restored slang prompt to {slang_target}", flush=True)

    if not args.no_verify:
        print("Verifying .nemo loads ...", flush=True)
        try:
            from nemo.collections.asr.models import ASRModel
            ASRModel.restore_from(str(target))
            print("Verified.", flush=True)
        except Exception as exc:
            print(f"WARN: verification failed: {exc}", file=sys.stderr)
            return 1

    print(
        f"\nNext:\n"
        f"  til build asr <tag>\n"
        f"  til test asr <tag>\n"
        f"  til submit asr <tag>   # only if local Eng-WER beats nemo-zs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
