#!/usr/bin/env bash
# C.1 retrain: v8s @ imgsz=1024 on the small-AP-targeted augmented dataset.
#
# Pre-requisite: run training/cv/build_aug_dataset.py first to produce
#   /home/jupyter/cv_yolo_dataset_augc1/
# with originals + native-resolution 1024x1024 crops + JPEG-recompressed
# train samples (val/test passed through untouched).
#
# Variables changed vs tier1's recipe:
#   - imgsz 768 -> 1024 (model can resolve smaller objects at high IoU)
#   - scale 0.60 -> 0.80 (training-time scale jitter pressures small-AP)
#   - train data: ~3x via offline crops + JPEG recompression
#
# Variables matched to tier1 (the only known-good v8s recipe):
#   - yolov8s.pt backbone
#   - copy_paste=0.10 (16/05 retrain confirmed 0.40 was toxic)
#   - mosaic=1.0, mixup=0.10
#   - epochs=80, optimizer=AdamW, lr0=0.001, cos_lr=True
#
# Approx training time on a Tesla T4: ~6 hours for 80 epochs at batch=8.
# If OOM at batch=8, drop to batch=6 (slower; still trains).
#
# Decision rule:
#   cloud > 0.556  -> new tier1 candidate, keep iterating
#   cloud = 0.556 ± 0.01 -> hypothesis holds but ceiling is here, consider
#                          stacking more aug or pivoting to a different
#                          backbone (RT-DETR-L, YOLOv9c)
#   cloud < 0.546  -> Phase C scope is wrong, fall back to tier1 and pivot

set -euo pipefail

DATA_YAML="${DATA_YAML:-/home/jupyter/cv_yolo_dataset_augc1/data.yaml}"
PROJECT="${PROJECT:-/home/jupyter/cv_runs}"
NAME="${NAME:-til-yolov8s-1024-augc1-v4}"
MODEL="${MODEL:-yolov8s.pt}"
EPOCHS="${EPOCHS:-80}"
IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-8}"
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
  close_mosaic=10 \
  mixup=0.10 \
  copy_paste=0.10 \
  degrees=5 \
  translate=0.10 \
  scale=0.80 \
  shear=0.0 \
  perspective=0.0 \
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
echo "  til build cv cv-augc1-v4"
echo
echo "Sweep at imgsz=1024 (matched) and imgsz=1280 (upscaled) with aug=0/1:"
echo "  python training/cv/sweep_cv_http.py \\"
echo "    --image melanie-minions-cv:cv-augc1-v4 \\"
echo "    --data-dir /home/jupyter/novice/cv \\"
echo "    --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \\"
echo "    --out-dir /home/jupyter/cv_eval_sweeps/augc1-v4 \\"
echo "    --conf 0.001,0.05,0.20 \\"
echo "    --iou 0.50,0.60,0.70 \\"
echo "    --imgsz 1024,1280 \\"
echo "    --augment 0,1"
echo
echo "Then submit the best blended row regardless of how it compares to tier1."
echo "Submission slots are uncapped; the leaderboard keeps the higher score."
