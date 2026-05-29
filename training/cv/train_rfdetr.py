#!/usr/bin/env python3
"""Train RF-DETR (Roboflow) on the TIL CV dataset.

RF-DETR ships no `yolo`-style CLI in our flow, so this is a Python entrypoint
(unlike the train_*.sh wrappers). Env-overridable in the same spirit as
training/cv/train_deep_backbone.sh.

Why this exists:
  Every in-family YOLO lever in cv/NOTES.md is exhausted; the CV ceiling is
  distribution shift (~80% different scene content), not backbone capacity.
  RF-DETR's DINOv2 backbone + domain-transfer design is the architecture bet
  whose mechanism matches that failure. This trains it for a clean local→cloud
  gap read vs the YOLO11l lineage.

Build the dataset first:
  python training/cv/build_rfdetr_dataset.py

Quick smoke (validate data loading + class indexing BEFORE the full run):
  EPOCHS=3 NAME=rfdetr-base-728-smoke python training/cv/train_rfdetr.py

Full run (T4: batch=4 grad_accum=4 -> effective batch 16):
  python training/cv/train_rfdetr.py

Knobs (env):
  VARIANT=base|large       RF-DETR model size (default base)
  RESOLUTION=728           input resolution, divisible by 56 (default 728)
  EPOCHS=60  BATCH=4  GRAD_ACCUM=4  LR=1e-4
  RFDETR_DATASET=/home/jupyter/cv_rfdetr_dataset
  OUTPUT_DIR=/home/jupyter/cv_runs/<NAME>
  NAME=rfdetr-base-728-v1
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def main() -> None:
    variant = os.environ.get("VARIANT", "base").strip().lower()
    resolution = _env_int("RESOLUTION", 728)
    if resolution % 56 != 0:
        snapped = max(56, round(resolution / 56) * 56)
        print(f"RESOLUTION {resolution} not divisible by 56; snapping to {snapped}")
        resolution = snapped

    name = os.environ.get("NAME", f"rfdetr-{variant}-{resolution}-v1")
    dataset_dir = os.environ.get("RFDETR_DATASET", "/home/jupyter/cv_rfdetr_dataset")
    output_dir = os.environ.get("OUTPUT_DIR", f"/home/jupyter/cv_runs/{name}")
    epochs = _env_int("EPOCHS", 60)
    batch = _env_int("BATCH", 4)
    grad_accum = _env_int("GRAD_ACCUM", 4)
    lr = _env_float("LR", 1e-4)

    from rfdetr import RFDETRBase, RFDETRLarge

    model_cls = RFDETRLarge if variant == "large" else RFDETRBase
    print(
        f"[train_rfdetr] variant={variant} resolution={resolution} epochs={epochs} "
        f"batch={batch} grad_accum={grad_accum} lr={lr} "
        f"dataset={dataset_dir} output={output_dir}",
        flush=True,
    )

    model = model_cls(resolution=resolution)
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        lr=lr,
        output_dir=output_dir,
        early_stopping=True,
    )

    best = Path(output_dir) / "checkpoint_best_total.pth"
    print()
    print(f"Best checkpoint (for inference): {best}")
    print()
    print("Next steps:")
    print("  mkdir -p cv/models")
    print(f"  cp {best} cv/models/best.pth")
    print(f"  # rfdetr serving env (offset=1 mapping from build_rfdetr_dataset.py):")
    print(
        "  til build cv {name} \\\n"
        "    # at build/run, override: CV_MODEL_FAMILY=rfdetr "
        "CV_MODEL_PATH=/workspace/models/cv/best.pth \\\n"
        f"    # CV_RFDETR_RESOLUTION={resolution} CV_RFDETR_VARIANT={variant} "
        "CV_CONF=0.30 CV_CROSS_CLASS_NMS_IOU=0 \\\n"
        "    # CV_CATEGORY_MAP from build_rfdetr_dataset.py output".format(name=name)
    )
    print()
    print("  python training/cv/sweep_cv_http.py \\")
    print(f"    --image melanie-minions-cv:{name} \\")
    print("    --model-family rfdetr \\")
    print("    --data-dir /home/jupyter/novice/cv \\")
    print("    --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \\")
    print(f"    --out-dir /home/jupyter/cv_eval_sweeps/{name} \\")
    print("    --conf 0.20,0.30,0.40,0.50 \\")
    print(f"    --rfdetr-resolution {resolution} \\")
    print("    --cross-class-nms-iou 0")


if __name__ == "__main__":
    main()
