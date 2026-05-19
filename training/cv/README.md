# CV fine-tuning and eval workflow

Goal: reproduce or improve the shipped CV detector without trusting the leaky
full-local `til test` score. The current best shipped tag is `ry-v2`, which
officially scored `0.608 / 0.961` with `0 / 500` errors. The useful local
selection metric is the hard held-out HTTP eval, not the full local `til test`
mAP.

As of the latest CV notes, YOLOv8s/YOLOv11m inference and retraining levers are
mostly exhausted. The next useful experiment is `training/cv/train_rtdetr.sh`,
which tests RT-DETR-L on the clean canonical hard split before adding more
distribution-shift augmentation.

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
ENV CV_CONF=0.15
ENV CV_IOU=0.55
ENV CV_IMGSZ=896
ENV CV_AUGMENT=0
ENV CV_CROSS_CLASS_NMS_IOU=0.97
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

By default the sweep uses an automatically chosen host port for each temporary
container, so a still-running `cv-eval` container on port `5002` will not break
the sweep. Pass `--port 5002` only when you specifically want that fixed port.

The sweep writes:

```text
/home/jupyter/cv_eval_sweeps/yolo-v2/sweep_results.csv
/home/jupyter/cv_eval_sweeps/yolo-v2/sweep_results.json
```

Take the best held-out setting and bake it into the submitted image by editing
`cv/Dockerfile`, or pass the same env values when doing extra local checks.
For `ry-v2`, the submitted serving row was:

```Dockerfile
ENV CV_CONF=0.15
ENV CV_IOU=0.55
ENV CV_IMGSZ=896
ENV CV_AUGMENT=0
ENV CV_CROSS_CLASS_NMS_IOU=0.97
```

This row scored local hard held-out `mAP50-95=0.9234`, up from the previous
`ruiyang-v1` eval at `0.9125`, and official improved from `0.588 / 0.955` to
`0.608 / 0.961`. Nearby `CV_IOU=0.50..0.70` rows tied on mAP; `0.55` is the
selected conservative row. `CV_CROSS_CLASS_NMS_IOU=0.97` is effectively neutral
locally and may trim near-identical hidden subclass duplicates.

For the next accuracy-first candidate, `ensemble_lab.tgz` showed that the best
cheap base is `CV_CONF=0.05`, `CV_IOU=0.70`, `CV_CROSS_CLASS_NMS_IOU=0`
(`0.9237` local mAP). Adding a down-weighted TTA rescue pass improves local mAP
to `0.9270` when enabled only on dense images (`CV_SECOND_MIN_DETECTIONS=7`),
or `0.9283` when enabled on all images. The current `cv/Dockerfile` bakes the
adaptive version.

Then rebuild with a new tag.

Cross-class NMS remains available as `CV_CROSS_CLASS_NMS_IOU`, but avoid lower
thresholds unless a sweep proves they help the exact checkpoint being submitted.
For the earlier `ruiyang-v1` saved predictions, offline scoring was:

```text
cross_nms=0.00  mAP=0.9125  small=0.7112
cross_nms=0.90  mAP=0.9079  small=0.7112
cross_nms=0.95  mAP=0.9110  small=0.7112
cross_nms=0.97  mAP=0.9124  small=0.7112
```

That makes `0` the safest local choice; `0.97` is a nearly neutral optional
cloud A/B, while `0.90` and `0.95` should not be baked by default.

## 6. Final smoke test and submit

```bash
til build cv cv-yolo-v2-best
til test cv cv-yolo-v2-best
til submit cv cv-yolo-v2-best
```

Submit only if `til test` has `0` errors. A lower speed is acceptable if held-out
mAP improves materially: CV's task score is approximately `0.75 * mAP + 0.25 *
speed`.

Observed v2-best results:

```text
Ultralytics val:            mAP50-95 0.920 on 500-val split
Hard held-out HTTP eval:    mAP50-95 0.8589, mAP50 0.9414, small AP 0.5596
Full local til test:        mAP50-95 0.8839, 0 errors
Official hidden eval:       score 0.549, speed 0.960, 0 / 500 errors
```

## 7. RT-DETR architecture candidate

This is the current highest-information next run because v8s/v11m recipe tweaks
have not transferred to cloud. Train:

```bash
bash training/cv/train_rtdetr.sh
```

Then build and sweep through the same HTTP service:

```bash
cp /home/jupyter/cv_runs/til-rtdetr-l-896-hard-v1/weights/best.pt cv/models/best.pt
til build cv cv-rtdetr-l-v1

python training/cv/sweep_cv_http.py \
  --image melanie-minions-cv:cv-rtdetr-l-v1 \
  --model-family rtdetr \
  --data-dir /home/jupyter/novice/cv \
  --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \
  --out-dir /home/jupyter/cv_eval_sweeps/rtdetr-l-v1 \
  --conf 0.05,0.10,0.20,0.30 \
  --iou 0.50,0.60,0.70 \
  --imgsz 896,1024,1280 \
  --augment 0 \
  --cross-class-nms-iou 0,0.97 \
  --rtdetr-eval-idx 3,5 \
  --rtdetr-num-queries 100,300
```

If submitting an RT-DETR checkpoint, keep `CV_MODEL_FAMILY=rtdetr` baked into
the image or passed at runtime. The serving code can auto-fallback if YOLO
loading fails, but setting the family explicitly also enables the RT-DETR
decoder/query controls.

## What to look for

- If held-out AP is strong but official stays low: hidden distribution is still
  different. Prefer stronger augmentation, `yolov8m`, and harder splits.
- If small-object AP is weak: try `imgsz=896`, lower `CV_CONF`, and keep
  `max_det=100`.
- If a few classes are near zero: use class-balanced sampling or targeted
  augmentation for those categories before increasing model size.
- If `til test` errors: fix schema/container startup first; never submit a
  crashing image.
