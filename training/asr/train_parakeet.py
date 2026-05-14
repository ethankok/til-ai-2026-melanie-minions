"""Fine-tune NVIDIA Parakeet-TDT-0.6B-v2 on the novice ASR manifest.

Loads the pre-trained `.nemo` checkpoint, points it at our NeMo-format
manifests (built by `prepare_data_nemo.py`), and runs a short fine-tune on
the GCP Workbench T4. The output is a fine-tuned `.nemo` ready for
`export_parakeet.py` to drop into `asr/models/`.

Usage::

    python training/asr/train_parakeet.py \\
        --data-dir training/asr/data_nemo \\
        --base-model asr/models/parakeet-tdt-0.6b-v2.nemo \\
        --output-dir training/asr/runs/parakeet-ft-v1 \\
        --epochs 5 \\
        --lr 5e-5 \\
        --batch-size 8

Knobs worth knowing about:

  --freeze-encoder       (default ON) — Parakeet's conformer encoder is
                          large and the encoder generalizes well; full
                          unfreeze burns a lot of T4 budget for marginal
                          gain on 4110 in-domain clips. Same call we made
                          on distil-whisper LoRA. Pass `--no-freeze-encoder`
                          to disable.
  --batch-size           per-GPU train batch. T4 fits 8 at fp16 for clips
                          up to 40s. Drop to 4 if you OOM on long-clip
                          buckets; bump grad-accum to compensate.
  --grad-accum           gradient accumulation steps. Effective batch =
                          batch-size * grad-accum.
  --epochs               5 is a reasonable default; the manifest has 4110
                          clips and we don't want to overfit. Watch
                          val_wer.
  --lr                   learning rate; 5e-5 to 1e-4 is the working range
                          for fine-tuning Parakeet.
  --val-every            run validation every N global steps (smaller =
                          finer-grained early stopping but slower).

The trainer ALWAYS uses a fresh PyTorch-Lightning Trainer (we don't try to
resume a previous run — Workbench T4 is fast enough that a clean run is
the simpler default). On graceful exit it saves both the best-val
checkpoint and the last checkpoint.
"""

from __future__ import annotations

# --- NumPy 2.0 / NeMo 2.0.0 compatibility shim --------------------------- #
# NeMo 2.0.0's `AudioSegment._convert_samples_to_float32` uses
# `np.sctypes['int']`, which was removed in NumPy 2.0. The Workbench env has
# numpy 2.4.x. Without this shim the dataloader worker dies on the first
# batch with `AttributeError: 'np.sctypes' was removed`. Restoring the
# minimum subset NeMo accesses keeps everything self-contained — no env
# downgrade, no NeMo bump.
#
# This MUST run before the heavy NeMo / torch imports below, since NeMo
# resolves these dtype lists at module load.
import numpy as _np_compat
if not hasattr(_np_compat, "sctypes"):
    _np_compat.sctypes = {
        "int": [_np_compat.int8, _np_compat.int16, _np_compat.int32, _np_compat.int64],
        "uint": [_np_compat.uint8, _np_compat.uint16, _np_compat.uint32, _np_compat.uint64],
        "float": [_np_compat.float16, _np_compat.float32, _np_compat.float64],
        "complex": [_np_compat.complex64, _np_compat.complex128],
        "others": [bool, object, bytes, str, _np_compat.void],
    }

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# NeMo imports are heavy and require CUDA-bearing wheels installed; defer
# them so `--help` works on a developer laptop.


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("training/asr/data_nemo"),
        help="Directory with train_manifest.jsonl + val_manifest.jsonl "
             "(produced by prepare_data_nemo.py).",
    )
    ap.add_argument(
        "--base-model",
        type=Path,
        default=Path("asr/models/parakeet-tdt-0.6b-v2.nemo"),
        help="Path to the pre-trained .nemo checkpoint to start from. "
             "Should already be on disk via download_models_nemo.py.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/asr/runs/parakeet-ft-v1"),
        help="Where checkpoints and logs land.",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Linear warmup steps. Short to keep Workbench wall clock down.",
    )
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--val-every",
        type=int,
        default=200,
        help="Run validation every N global steps.",
    )
    ap.add_argument(
        "--freeze-encoder",
        dest="freeze_encoder",
        action="store_true",
        default=True,
        help="Freeze the conformer encoder (default ON).",
    )
    ap.add_argument(
        "--no-freeze-encoder",
        dest="freeze_encoder",
        action="store_false",
        help="Train the encoder along with the decoder. Slower, more VRAM.",
    )
    ap.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="PyTorch-Lightning precision string. T4 has no bf16 so '16-mixed' "
             "is the correct choice; on Ampere+ use 'bf16-mixed'.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Hard cap on optimizer steps. -1 = let --epochs decide.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return ap


def _check_paths(args: argparse.Namespace) -> None:
    if not args.base_model.exists():
        raise SystemExit(
            f"Base .nemo not found at {args.base_model}. Run "
            f"`python training/asr/download_models_nemo.py` first."
        )
    train_m = args.data_dir / "train_manifest.jsonl"
    val_m = args.data_dir / "val_manifest.jsonl"
    if not train_m.exists() or not val_m.exists():
        raise SystemExit(
            f"Missing manifests in {args.data_dir}. Run "
            f"`python training/asr/prepare_data_nemo.py` first."
        )


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep NeMo's verbose dataloader / RNNT decoding logs out of the way; the
    # ones we care about (val_wer per validation pass) still print.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")


def _build_train_ds_cfg(
    manifest: Path, batch_size: int, num_workers: int
) -> dict:
    return {
        "manifest_filepath": str(manifest),
        "sample_rate": 16000,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "max_duration": 40.0,
        "min_duration": 0.5,
        "trim_silence": False,
        "use_lhotse": False,
    }


def _build_val_ds_cfg(manifest: Path, batch_size: int, num_workers: int) -> dict:
    return {
        "manifest_filepath": str(manifest),
        "sample_rate": 16000,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "max_duration": 40.0,
        "use_lhotse": False,
    }


def _build_optim_cfg(lr: float, warmup_steps: int) -> dict:
    return {
        "name": "adamw",
        "lr": lr,
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": warmup_steps,
            "min_lr": 1e-7,
        },
    }


def main() -> int:
    args = _build_arg_parser().parse_args()
    _check_paths(args)
    _setup_logging(args.output_dir)

    # --- imports (heavy) ---------------------------------------------------
    try:
        import pytorch_lightning as pl
        from omegaconf import OmegaConf, open_dict
        from nemo.collections.asr.models import ASRModel
        from pytorch_lightning.callbacks import (
            ModelCheckpoint,
            EarlyStopping,
            LearningRateMonitor,
        )
        from pytorch_lightning.loggers import CSVLogger
    except ImportError as exc:  # pragma: no cover - env issue
        raise SystemExit(
            "Training deps missing. Install with `pip install -r "
            "asr/requirements-nemo.txt` on Workbench."
        ) from exc

    pl.seed_everything(args.seed, workers=True)

    print(f"Loading base model from {args.base_model} ...", flush=True)
    model = ASRModel.restore_from(restore_path=str(args.base_model))

    # Wire the manifests in. NeMo expects the cfg-driven dataloader, not a
    # raw torch DataLoader, so we hand it dict configs.
    model.setup_training_data(
        OmegaConf.create(
            _build_train_ds_cfg(
                args.data_dir / "train_manifest.jsonl",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )
    )
    model.setup_validation_data(
        OmegaConf.create(
            _build_val_ds_cfg(
                args.data_dir / "val_manifest.jsonl",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )
    )

    # Override the optimizer config with our LR + warmup. NeMo reads
    # `model.cfg.optim` at `configure_optimizers` time.
    with open_dict(model.cfg):
        model.cfg.optim = OmegaConf.create(
            _build_optim_cfg(lr=args.lr, warmup_steps=args.warmup_steps)
        )

    if args.freeze_encoder:
        # Same call we made on distil-whisper. Parakeet's conformer encoder
        # is well-trained on enough English audio that further FT on 4110
        # clips mostly hurts. Decoder + joint network capture the in-world
        # vocabulary adaptation.
        if hasattr(model, "encoder"):
            for p in model.encoder.parameters():
                p.requires_grad = False
            print("Froze encoder parameters.", flush=True)
        else:
            print(
                "WARN: model has no `.encoder` attribute; --freeze-encoder ignored",
                flush=True,
            )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {n_trainable / 1e6:.2f}M of {n_total / 1e6:.2f}M "
        f"({100 * n_trainable / n_total:.1f}%)",
        flush=True,
    )

    # --- Lightning trainer -------------------------------------------------
    ckpt_dir = args.output_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="parakeet-ft-{step}-{val_wer:.4f}",
            monitor="val_wer",
            mode="min",
            save_top_k=2,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val_wer",
            mode="min",
            patience=3,
            min_delta=1e-4,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer_kwargs: dict = dict(
        accelerator="gpu" if "CUDA_VISIBLE_DEVICES" not in os.environ
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "gpu",
        devices=1,
        precision=args.precision,
        max_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        accumulate_grad_batches=args.grad_accum,
        val_check_interval=args.val_every,
        check_val_every_n_epoch=None,
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        callbacks=callbacks,
        logger=CSVLogger(str(args.output_dir), name="parakeet_ft"),
        default_root_dir=str(args.output_dir),
    )
    trainer = pl.Trainer(**trainer_kwargs)

    # Hand our trainer back to the model so logged WERs route correctly.
    model.set_trainer(trainer)

    print("Starting fit ...", flush=True)
    trainer.fit(model)

    # Save the best-checkpoint as a single .nemo for export.
    best_ckpt = callbacks[0].best_model_path  # ModelCheckpoint
    best_score = callbacks[0].best_model_score
    print(f"Best val_wer = {best_score} at {best_ckpt}", flush=True)

    if best_ckpt and Path(best_ckpt).exists():
        print("Restoring best weights from checkpoint ...", flush=True)
        model.load_state_dict(
            __import__("torch").load(best_ckpt, map_location="cpu")["state_dict"],
            strict=False,
        )

    final_nemo = args.output_dir / "best.nemo"
    print(f"Saving best model to {final_nemo} ...", flush=True)
    model.save_to(str(final_nemo))

    summary = {
        "base_model": str(args.base_model),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "freeze_encoder": args.freeze_encoder,
        "best_ckpt": best_ckpt,
        "best_val_wer": float(best_score) if best_score is not None else None,
        "final_nemo": str(final_nemo),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
