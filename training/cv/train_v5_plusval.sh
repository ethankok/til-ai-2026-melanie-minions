#!/usr/bin/env bash
# Final-data YOLOv8s fine-tune.
#
# Why this exists:
#   The hard split intentionally held back many rare/dense images. That made a
#   useful stress test, but final leaderboard training should use more labels.
#   This script trains on old train+val and validates on old hard test.
#
# First build the plus-val dataset:
#   python training/cv/build_final_yolo_dataset.py
#
# Default starts from cv/models/best.pt, which should be the current ry-v2
# checkpoint. Override MODEL=yolov8s.pt if you want a from-pretrain rerun.

set -euo pipefail

DATA_YAML="${DATA_YAML:-/home/jupyter/cv_yolo_dataset_plusval/data.yaml}"
PROJECT="${PROJECT:-/home/jupyter/cv_runs}"
NAME="${NAME:-til-yolov8s-896-plusval-v5}"
MODEL="${MODEL:-/home/jupyter/til/cv/models/best.pt}"
EPOCHS="${EPOCHS:-60}"
IMGSZ="${IMGSZ:-896}"
BATCH="${BATCH:-10}"
DEVICE="${DEVICE:-0}"

if [[ ! -f "${MODEL}" ]]; then
  echo "MODEL=${MODEL} not found; falling back to yolov8s.pt"
  MODEL="yolov8s.pt"
fi

yolo detect train \
  model="${MODEL}" \
  data="${DATA_YAML}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  project="${PROJECT}" \
  name="${NAME}" \
  patience=20 \
  optimizer=AdamW \
  lr0=0.0004 \
  lrf=0.05 \
  cos_lr=True \
  warmup_epochs=2 \
  mosaic=0.7 \
  close_mosaic=10 \
  mixup=0.05 \
  copy_paste=0.0 \
  degrees=3 \
  translate=0.08 \
  scale=0.55 \
  shear=0.0 \
  perspective=0.0 \
  fliplr=0.50 \
  hsv_h=0.010 \
  hsv_s=0.50 \
  hsv_v=0.30 \
  cls=0.5 \
  box=7.5 \
  dfl=1.5

echo
echo "Best weights: ${PROJECT}/${NAME}/weights/best.pt"
echo
echo "Next steps:"
echo "  cp ${PROJECT}/${NAME}/weights/best.pt cv/models/best.pt"
echo "  til build cv ry-v4-plusval"
echo "  til test cv ry-v4-plusval"
echo
echo "Recommended quick HTTP eval:"
echo "  docker rm -f cv-eval 2>/dev/null || true"
echo "  docker run -d --rm --gpus all --name cv-eval -p 5002:5002 melanie-minions-cv:ry-v4-plusval"
echo "  python training/cv/eval_cv_http.py \\"
echo "    --data-dir /home/jupyter/novice/cv \\"
echo "    --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \\"
echo "    --batch-size 4 --timeout 180 --retries 8 --retry-delay 10 \\"
echo "    --predictions-json /home/jupyter/cv_eval/ry-v4-plusval/predictions.json \\"
echo "    --summary-json /home/jupyter/cv_eval/ry-v4-plusval/summary.json"
echo
echo "Keep cv/Dockerfile on the restored ry-v2 serving row unless a sweep beats it."
