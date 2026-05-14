#!/usr/bin/env bash
# Tier 2 retrain: YOLOv11m at imgsz=1024 with stronger augmentation.
#
# Run on Workbench after `prepare_yolo_dataset.py` has produced the hard split
# at /home/jupyter/cv_yolo_dataset/data.yaml.
#
# Goal: lift official CV from 0.549 -> ~0.70 by adding capacity (m vs s) and
# resolution (1024 vs 768), plus copy-paste/mosaic for small objects.
#
# Approx training time on a Tesla T4: 4-5 hours for 120 epochs, batch=6.
# If T4 OOMs at batch=6, drop to batch=4 (slower, still trains).
# If a stronger GPU is available (A100/L4), bump batch=12 and/or imgsz=1280.

set -euo pipefail

DATA_YAML="${DATA_YAML:-/home/jupyter/cv_yolo_dataset/data.yaml}"
PROJECT="${PROJECT:-/home/jupyter/cv_runs}"
NAME="${NAME:-til-yolo11m-1024-hard-v3}"
MODEL="${MODEL:-yolo11m.pt}"
EPOCHS="${EPOCHS:-120}"
IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-6}"
DEVICE="${DEVICE:-0}"

yolo detect train \
  model="${MODEL}" \
  data="${DATA_YAML}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  project="${PROJECT}" \
  name="${NAME}" \
  patience=30 \
  optimizer=AdamW \
  lr0=0.001 \
  cos_lr=True \
  warmup_epochs=3 \
  mosaic=1.0 \
  close_mosaic=20 \
  mixup=0.15 \
  copy_paste=0.30 \
  degrees=10 \
  translate=0.15 \
  scale=0.70 \
  shear=2.0 \
  perspective=0.0005 \
  fliplr=0.50 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  cls=0.5 \
  box=7.5 \
  dfl=1.5

echo
echo "Best weights: ${PROJECT}/${NAME}/weights/best.pt"
echo
echo "Next steps:"
echo "  cp ${PROJECT}/${NAME}/weights/best.pt cv/models/best.pt"
echo "  til build cv cv-yolo11m-v3"
echo
echo "Then run training/cv/sweep_cv_http.py to find the best CV_CONF/IOU/IMGSZ."
