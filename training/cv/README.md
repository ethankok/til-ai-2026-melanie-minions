# CV fine-tuning workflow

Goal: replace the weak COCO `yolov8n.pt` baseline (`0.044 / 0.961`) with a detector trained on the 18 TIL CV classes.

Run on GCP Workbench, not the Mac, because the data lives under `/home/jupyter/novice/cv`.

## 1. Convert annotations to YOLO format

```bash
cd /home/jupyter/til
python training/cv/prepare_yolo_dataset.py \
  --data-dir /home/jupyter/novice/cv \
  --out-dir /home/jupyter/cv_yolo_dataset \
  --val-frac 0.10
```

Verify it prints nonzero train/val image and box counts.

## 2. Train a first model

Start with YOLOv8s: it should be much stronger than `yolov8n` while still likely fast enough.

```bash
pip install ultralytics

yolo detect train \
  model=yolov8s.pt \
  data=/home/jupyter/cv_yolo_dataset/data.yaml \
  epochs=60 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=/home/jupyter/cv_runs \
  name=til-yolov8s-v1 \
  patience=15
```

If batch 16 OOMs, retry with `batch=8`.

## 3. Export/copy best checkpoint into the CV image

```bash
mkdir -p cv/models
cp /home/jupyter/cv_runs/til-yolov8s-v1/weights/best.pt cv/models/best.pt
```

Then patch `cv/Dockerfile` for the trained run:

```Dockerfile
ENV CV_MODEL_PATH=/workspace/models/cv/best.pt
COPY models /workspace/models/cv
```

Keep the existing YOLOv8n fallback path in `cv_manager.py`; if the custom model fails to load, the service should still return schema-valid outputs instead of crashing.

## 4. Build/test/submit

```bash
til build cv cv-yolo-ft-v1
til test cv cv-yolo-ft-v1
til submit cv cv-yolo-ft-v1
```

Submit only if `til test` has 0 errors. A lower local speed is acceptable if mAP improves materially: CV's blended task value is `0.75 * score + 0.25 * speed`.

## Next A/Bs after v1

- If score improves but speed remains high: try `model=yolov8m.pt`.
- If score improves but speed drops too much: keep `yolov8s`, reduce `imgsz=512`, or export TensorRT/ONNX only if the Docker path stays simple.
- If validation mAP is high but official is low: hidden distribution differs; add stronger augmentation, not more class-map guessing.
