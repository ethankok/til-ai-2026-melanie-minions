# CV fine-tuning and eval workflow

Goal: improve the shipped `cv-yolo-ft-v1` detector without trusting the leaky
full-local `til test` score. `cv-yolo-ft-v1` officially scored `0.402 / 0.963`
with `0 / 500` errors; local Docker mAP50-95 was `0.885`, but that local number
was measured on the same `/home/jupyter/novice/cv` images used for training.

Run this on GCP Workbench. The Mac checkout does not contain the CV images.

## 0. Setup

```bash
cd /home/jupyter/til
git pull
pip install -r requirements-dev.txt
pip install ultralytics
```

Confirm the data exists:

```bash
ls /home/jupyter/novice/cv/annotations.json /home/jupyter/novice/cv/images | head
```

## 1. Create non-leaky splits

The converter now writes YOLO labels plus split-specific COCO files:

```bash
python training/cv/prepare_yolo_dataset.py \
  --data-dir /home/jupyter/novice/cv \
  --out-dir /home/jupyter/cv_yolo_dataset \
  --val-frac 0.10 \
  --test-frac 0.10 \
  --split-mode hard
```

Outputs that matter:

```text
/home/jupyter/cv_yolo_dataset/data.yaml
/home/jupyter/cv_yolo_dataset/split_manifest.json
/home/jupyter/cv_yolo_dataset/coco/annotations_train.json
/home/jupyter/cv_yolo_dataset/coco/annotations_val.json
/home/jupyter/cv_yolo_dataset/coco/annotations_test.json
```

Use `annotations_test.json` for model selection. It is held out from training.
`--split-mode hard` reserves smaller, rarer, and more crowded target images for
held-out eval, which is closer to the hidden-score problem than a friendly
random split.

## 2. Train a candidate

Fast baseline candidate:

```bash
yolo detect train \
  model=yolov8s.pt \
  data=/home/jupyter/cv_yolo_dataset/data.yaml \
  epochs=80 \
  imgsz=768 \
  batch=12 \
  device=0 \
  project=/home/jupyter/cv_runs \
  name=til-yolov8s-768-hard-v2 \
  patience=20 \
  close_mosaic=10 \
  mosaic=1.0 \
  mixup=0.10 \
  degrees=5 \
  translate=0.10 \
  scale=0.60 \
  fliplr=0.50
```

If this OOMs on the T4, retry with `batch=8`. If speed still has headroom after
held-out eval, try the same command with `model=yolov8m.pt`, `batch=8`, and a
new `name=til-yolov8m-768-hard-v1`.

The Ultralytics validation metric is useful, but the selection metric is the
container HTTP eval in step 4.

## 3. Build the CV container

```bash
mkdir -p cv/models
cp /home/jupyter/cv_runs/til-yolov8s-768-hard-v2/weights/best.pt cv/models/best.pt

til build cv cv-yolo-v2
```

The Dockerfile already sets:

```Dockerfile
ENV CV_MODEL_PATH=/workspace/models/cv/best.pt
ENV CV_CATEGORY_MAP='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]'
```

Keep the identity category map for fine-tuned 18-class models.

## 4. Evaluate the actual HTTP service on held-out data

Start the image built by `til build`. If your team/image naming differs, replace
`melanie-minions-cv:cv-yolo-v2` with the image name printed by `til build`.

```bash
docker rm -f cv-eval 2>/dev/null || true
docker run -d --rm --gpus all \
  --name cv-eval \
  -p 5002:5002 \
  melanie-minions-cv:cv-yolo-v2

python training/cv/eval_cv_http.py \
  --data-dir /home/jupyter/novice/cv \
  --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \
  --predictions-json /home/jupyter/cv_eval/cv-yolo-v2-test-preds.json \
  --summary-json /home/jupyter/cv_eval/cv-yolo-v2-test-summary.json

docker rm -f cv-eval
```

Read the printed `mAP50-95`, area AP, and per-class AP. This is the number to
trust before submitting. The full `til test cv <tag>` score is still useful as a
final smoke test for schema, speed, and zero errors, but it is not a clean model
selection metric.

## 5. Sweep inference knobs

The model has high official speed (`0.963` for v1), so try lower confidence and
larger image sizes before retraining again:

```bash
python training/cv/sweep_cv_http.py \
  --image melanie-minions-cv:cv-yolo-v2 \
  --data-dir /home/jupyter/novice/cv \
  --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \
  --out-dir /home/jupyter/cv_eval_sweeps/yolo-v2 \
  --conf 0.05,0.10,0.15,0.20,0.25 \
  --iou 0.50,0.60,0.70 \
  --imgsz 640,768,896
```

The sweep writes:

```text
/home/jupyter/cv_eval_sweeps/yolo-v2/sweep_results.csv
/home/jupyter/cv_eval_sweeps/yolo-v2/sweep_results.json
```

Take the best held-out setting and bake it into the submitted image by editing
`cv/Dockerfile`, or pass the same env values when doing extra local checks. For
example, if the best row is `conf=0.10`, `iou=0.60`, `imgsz=768`, set:

```Dockerfile
ENV CV_CONF=0.10
ENV CV_IOU=0.60
ENV CV_IMGSZ=768
```

Then rebuild with a new tag.

## 6. Final smoke test and submit

```bash
til build cv cv-yolo-v2-best
til test cv cv-yolo-v2-best
til submit cv cv-yolo-v2-best
```

Submit only if `til test` has `0` errors. A lower speed is acceptable if held-out
mAP improves materially: CV's task score is approximately `0.75 * mAP + 0.25 *
speed`.

## What to look for

- If held-out AP is strong but official stays low: hidden distribution is still
  different. Prefer stronger augmentation, `yolov8m`, and harder splits.
- If small-object AP is weak: try `imgsz=896`, lower `CV_CONF`, and keep
  `max_det=100`.
- If a few classes are near zero: use class-balanced sampling or targeted
  augmentation for those categories before increasing model size.
- If `til test` errors: fix schema/container startup first; never submit a
  crashing image.
