# CV fine-tuning workflow

Goal: reproduce or improve the shipped `cv-yolo-ft-v1` detector. It replaced the weak COCO `yolov8n.pt` baseline (`0.044 / 0.961`) with a YOLOv8s detector trained on the 18 TIL CV classes from the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#cv). `cv-yolo-ft-v1` officially scored `0.402 / 0.963` with `0 / 500` errors; local Docker mAP50-95 was `0.885`. CV scoring is mAP@.5:.05:.95 plus speed; outputs must be LTWH `[l, t, w, h]` boxes with `category_id` 0-17.

Run on GCP Workbench, not the Mac, because the data lives under `/home/jupyter/novice/cv`.

## 1. Convert annotations to YOLO format

```bash
cd /home/jupyter/til
python training/cv/prepare_yolo_dataset.py \
  --data-dir /home/jupyter/novice/cv \
  --out-dir /home/jupyter/cv_yolo_dataset \
  --val-frac 0.10
```

Verify it prints nonzero train/val image and box counts. The converter preserves the official category order: `0 cargo aircraft`, `1 commercial aircraft`, `2 drone`, `3 fighter jet`, `4 fighter plane`, `5 helicopter`, `6 light aircraft`, `7 missile`, `8 truck`, `9 car`, `10 tank`, `11 bus`, `12 van`, `13 cargo ship`, `14 yacht`, `15 cruise ship`, `16 warship`, `17 sailboat`.

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

Watch validation `metrics/mAP50-95(B)` first because it matches the official CV accuracy metric. `metrics/mAP50(B)` is useful for debugging but overestimates leaderboard quality.

If batch 16 OOMs, retry with `batch=8`.

## 3. Export/copy best checkpoint into the CV image

```bash
mkdir -p cv/models
cp /home/jupyter/cv_runs/til-yolov8s-v1/weights/best.pt cv/models/best.pt
```

Then ensure `cv/Dockerfile` includes the trained model path (already true on `main` after `fix(cv): load trained YOLO checkpoint`):

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
