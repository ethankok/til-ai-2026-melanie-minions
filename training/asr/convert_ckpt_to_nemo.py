"""Convert a PyTorch Lightning checkpoint (.ckpt) from Parakeet fine-tuning to a standalone .nemo model.

Useful for exporting the best checkpoint if we need to stop training early.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
from nemo.collections.asr.models import ASRModel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the PyTorch Lightning checkpoint (.ckpt) file.",
    )
    ap.add_argument(
        "--base-model",
        type=Path,
        default=Path("asr/models/parakeet-tdt-0.6b-v2.nemo"),
        help="Path to the base pre-trained .nemo model.",
    )
    ap.add_argument(
        "--out-nemo",
        type=Path,
        required=True,
        help="Path where the converted .nemo model will be saved.",
    )
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"ERROR: Checkpoint not found at {args.ckpt}", file=sys.stderr)
        return 1
    if not args.base_model.exists():
        print(f"ERROR: Base model not found at {args.base_model}", file=sys.stderr)
        return 1

    print(f"Loading base model from {args.base_model} ...", flush=True)
    model = ASRModel.restore_from(restore_path=str(args.base_model))

    print(f"Loading checkpoint state dict from {args.ckpt} ...", flush=True)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    print("Loading weights into model ...", flush=True)
    model.load_state_dict(state_dict, strict=False)

    args.out_nemo.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving standalone model to {args.out_nemo} ...", flush=True)
    model.save_to(str(args.out_nemo))

    print("Success.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
