#!/usr/bin/env bash
# Deep CV backbone training sweep entrypoint.
#
# Purpose:
#   Train larger/deeper supervised detectors. Default candidate is YOLO11-L.
#   Override MODEL for YOLO11-X, YOLOv8-X, or RT-DETR-X.
#
# Expected input:
#   /home/jupyter/cv_yolo_dataset_plusval/data.yaml
#
# Build it first with:
#   python training/cv/prepare_yolo_dataset.py
#   python training/cv/build_final_yolo_dataset.py
#
# Examples:
#   bash training/cv/train_deep_backbone.sh
#   MODEL=yolo11x.pt NAME=til-yolo11x-1024-plusval-v1 BATCH=3 bash training/cv/train_deep_backbone.sh
#   MODEL=rtdetr-x.pt NAME=til-rtdetr-x-1024-plusval-v1 BATCH=2 bash training/cv/train_deep_backbone.sh

set -euo pipefail

DATA_YAML="${DATA_YAML:-/home/jupyter/cv_yolo_dataset_plusval/data.yaml}"
PROJECT="${PROJECT:-/home/jupyter/cv_runs}"
MODEL="${MODEL:-yolo11l.pt}"
NAME="${NAME:-til-yolo11l-1024-plusval-v1}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-0}"
PATIENCE="${PATIENCE:-25}"

MODEL_LC="$(printf '%s' "${MODEL}" | tr '[:upper:]' '[:lower:]')"

COMMON_ARGS=(
  "model=${MODEL}"
  "data=${DATA_YAML}"
  "epochs=${EPOCHS}"
  "imgsz=${IMGSZ}"
  "batch=${BATCH}"
  "device=${DEVICE}"
  "project=${PROJECT}"
  "name=${NAME}"
  "patience=${PATIENCE}"
  "optimizer=AdamW"
  "cos_lr=True"
  "warmup_epochs=3"
  "mosaic=0.7"
  "close_mosaic=15"
  "mixup=0.05"
  "copy_paste=0.0"
  "degrees=3"
  "translate=0.08"
  "scale=0.55"
  "shear=0.0"
  "perspective=0.0"
  "fliplr=0.50"
  "hsv_h=0.010"
  "hsv_s=0.45"
  "hsv_v=0.30"
)

if [[ "${MODEL_LC}" == rtdetr* ]]; then
  # RT-DETR tends to prefer a smaller LR and no YOLO-specific loss weights.
  yolo train \
    "${COMMON_ARGS[@]}" \
    "lr0=${LR0:-0.0001}" \
    "lrf=${LRF:-0.01}"
else
  yolo detect train \
    "${COMMON_ARGS[@]}" \
    "lr0=${LR0:-0.00025}" \
    "lrf=${LRF:-0.03}" \
    "box=${BOX_LOSS:-8.0}" \
    "cls=${CLS_LOSS:-0.5}" \
    "dfl=${DFL_LOSS:-1.5}"
fi

BEST="${PROJECT}/${NAME}/weights/best.pt"

echo
echo "Best weights: ${BEST}"
echo
echo "Next steps:"
echo "  mkdir -p cv/models"
echo "  cp ${BEST} cv/models/best.pt"
if [[ "${MODEL_LC}" == rtdetr* ]]; then
  echo "  # For RT-DETR candidates, set CV_MODEL_FAMILY=rtdetr in cv/Dockerfile before final build."
  echo "  til build cv ${NAME}"
  echo "  python training/cv/sweep_cv_http.py \\"
  echo "    --image melanie-minions-cv:${NAME} \\"
  echo "    --model-family rtdetr \\"
  echo "    --data-dir /home/jupyter/novice/cv \\"
  echo "    --annotations /home/jupyter/cv_yolo_dataset_plusval/coco/annotations_val.json \\"
  echo "    --out-dir /home/jupyter/cv_eval_sweeps/${NAME} \\"
  echo "    --conf 0.03,0.05,0.10,0.15,0.20 \\"
  echo "    --iou 0.50,0.55,0.60,0.70 \\"
  echo "    --imgsz 896,1024,1280 \\"
  echo "    --augment 0 \\"
  echo "    --cross-class-nms-iou 0,0.97 \\"
  echo "    --rtdetr-eval-idx 3,5 \\"
  echo "    --rtdetr-num-queries 100,300"
else
  echo "  til build cv ${NAME}"
  echo "  python training/cv/sweep_cv_http.py \\"
  echo "    --image melanie-minions-cv:${NAME} \\"
  echo "    --data-dir /home/jupyter/novice/cv \\"
  echo "    --annotations /home/jupyter/cv_yolo_dataset_plusval/coco/annotations_val.json \\"
  echo "    --out-dir /home/jupyter/cv_eval_sweeps/${NAME} \\"
  echo "    --conf 0.03,0.05,0.10,0.15,0.20 \\"
  echo "    --iou 0.50,0.55,0.60,0.70 \\"
  echo "    --imgsz 896,1024,1280 \\"
  echo "    --augment 0,1 \\"
  echo "    --cross-class-nms-iou 0,0.97"
fi
