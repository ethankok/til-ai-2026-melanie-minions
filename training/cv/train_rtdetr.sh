#!/usr/bin/env bash
# Phase D candidate: RT-DETR-L on the canonical hard split.
#
# Why this exists:
#   The shipped v8s tier1 model is near the ceiling for the YOLOv8s/YOLOv11m
#   recipe space in cv/NOTES.md. RT-DETR is the next useful architecture test:
#   anchor-free, NMS-free, stronger global context, and still practical on a T4.
#
# Run on Workbench after `prepare_yolo_dataset.py` has produced:
#   /home/jupyter/cv_yolo_dataset/data.yaml
#
# Start with the clean canonical dataset, not the augc1 tile-crop dataset. The
# augc1 run improved hard held-out but failed to transfer to cloud, so this
# script tests architecture generalization before adding more distribution bias.

set -euo pipefail

DATA_YAML="${DATA_YAML:-/home/jupyter/cv_yolo_dataset/data.yaml}"
PROJECT="${PROJECT:-/home/jupyter/cv_runs}"
NAME="${NAME:-til-rtdetr-l-896-hard-v1}"
MODEL="${MODEL:-rtdetr-l.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-896}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-0}"

yolo train \
  model="${MODEL}" \
  data="${DATA_YAML}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  project="${PROJECT}" \
  name="${NAME}" \
  patience=25 \
  optimizer=AdamW \
  lr0=0.0001 \
  lrf=0.01 \
  cos_lr=True \
  warmup_epochs=3 \
  mosaic=0.5 \
  close_mosaic=15 \
  mixup=0.05 \
  copy_paste=0.0 \
  degrees=5 \
  translate=0.10 \
  scale=0.50 \
  shear=0.0 \
  perspective=0.0 \
  fliplr=0.50 \
  hsv_h=0.015 \
  hsv_s=0.5 \
  hsv_v=0.3

echo
echo "Best weights: ${PROJECT}/${NAME}/weights/best.pt"
echo
echo "Next steps:"
echo "  cp ${PROJECT}/${NAME}/weights/best.pt cv/models/best.pt"
echo "  til build cv cv-rtdetr-l-v1"
echo
echo "Sweep the HTTP service with RT-DETR loading and decoder/query options:"
echo "  python training/cv/sweep_cv_http.py \\"
echo "    --image melanie-minions-cv:cv-rtdetr-l-v1 \\"
echo "    --model-family rtdetr \\"
echo "    --data-dir /home/jupyter/novice/cv \\"
echo "    --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \\"
echo "    --out-dir /home/jupyter/cv_eval_sweeps/rtdetr-l-v1 \\"
echo "    --conf 0.05,0.10,0.20,0.30 \\"
echo "    --iou 0.50,0.60,0.70 \\"
echo "    --imgsz 896,1024,1280 \\"
echo "    --augment 0 \\"
echo "    --cross-class-nms-iou 0,0.90,0.95 \\"
echo "    --rtdetr-eval-idx 3,5 \\"
echo "    --rtdetr-num-queries 100,300"
