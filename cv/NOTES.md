# CV — notes & history

Last updated: 14 May 2026 14:10 SGT

Per-task working log for CV (object detection). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#cv).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`cv-yolo-v2-best` — official 0.549 / 0.960 (14 May 14:00 SGT, 0 of 500 errors).**

YOLOv8s retrained at `imgsz=768` on a non-leaky hard split, then selected through
the actual Docker HTTP path. Full hard held-out HTTP eval scored `mAP50-95 =
0.8589`; full local `til test` scored `0.8839`; official hidden eval scored
`0.549`. Serving/schema is clean: `0 / 500` errors and speed stayed high.

This is +0.147 official accuracy over `cv-yolo-ft-v1` (`0.402 / 0.963`) with
essentially unchanged speed.

## What our model runs on

Current branch implementation: a schema-safe Ultralytics YOLO service in
[src/cv_manager.py](src/cv_manager.py), packaged by [Dockerfile](Dockerfile).

```python
def cv(image_bytes: bytes, key=None) -> list[dict]:
    # Decode robustly, run YOLO, convert xyxy -> official LTWH.
    # Any decode/model failure returns [] and logs key/byte size.
```

Docker now expects the trained checkpoint at `/workspace/models/cv/best.pt`,
copied from local `cv/models/best.pt` during build. `CV_CATEGORY_MAP` is an
identity `0..17` map because the fine-tuned model was trained directly on the
official TIL label order. The v2-best image uses tuned inference defaults:
`CV_CONF=0.25`, `CV_IOU=0.50`, `CV_IMGSZ=768`.

## Submission history

```text
Tag              Submitted       Score   Speed   Errors    Notes
latest           12/05 03:52     0.000   0.981   4 / 500   Empty-detection baseline, 4 inputs erroring
yolo-til-map-v2  14/05 01:56     0.044   0.961   0 / 500   YOLOv8n + sparse COCO→TIL map; clean serving, weak domain fit
cv-yolo-ft-v1    14/05 03:53     0.402   0.963   0 / 500   YOLOv8s fine-tuned on 18 TIL labels; local mAP50-95 0.885
cv-yolo-v2-best  14/05 14:00     0.549   0.960   0 / 500   YOLOv8s 768 hard-split retrain + tuned inference; local til test 0.8839, hard held-out 0.8589
```

## Detailed timeline

### cv-yolo-v2-best (14 May 14:00) — hard split + 768 inference tuning

- Created non-leaky hard splits from `/home/jupyter/novice/cv/annotations.json`:
  `4000` train images, `500` val images, `500` test images. The hard test split
  had `3334` boxes, denser than train and intentionally useful for stress eval.
- Trained `yolov8s.pt` for 80 epochs at `imgsz=768`, `batch=12`, `device=0`,
  with stronger augmentation (`close_mosaic=10`, `mixup=0.10`, `degrees=5`,
  `scale=0.60`). Training took `2.329` hours on the Workbench T4.
- Final Ultralytics validation on the 500-image val split: precision `0.984`,
  recall `0.961`, `mAP50=0.985`, `mAP50-95=0.920`.
- Default-container hard held-out HTTP eval (`CV_CONF=0.25`, `CV_IOU=0.70`,
  `CV_IMGSZ=640`) scored `0.8043` mAP50-95.
- HTTP sweep on 100 hard-test images found `imgsz=768` was the main win. Best
  smoke setting was `CV_CONF=0.25`, `CV_IOU=0.50`, `CV_IMGSZ=768`, scoring
  `0.8652` mAP50-95 on the 100-image slice.
- Full hard held-out HTTP confirmation at those settings scored `0.8589`
  mAP50-95, `0.9414` mAP50, small-object AP `0.5596`, medium AP `0.7557`,
  large AP `0.8911`.
- Full local `til test cv cv-yolo-v2-best`: `mAP@.5:.05:.95 = 0.8839`, `0`
  errors, `1250/1250` batches in `13:59`.
- Official hidden eval: `0.549 / 0.960`, `0 / 500` errors. New high score:
  +0.147 official accuracy vs `cv-yolo-ft-v1`.

Interpretation: the non-leaky held-out eval was directionally useful even though
the hidden gap is still large (`0.8589 → 0.549`). The biggest confirmed lever was
matching inference `imgsz=768` to training. Remaining weakness is likely hidden
small-object and aircraft-subclass distribution shift, not output format.

### cv-yolo-ft-v1 (14 May 03:53) — trained 18-class YOLOv8s

- Converted `/home/jupyter/novice/cv/annotations.json` into an Ultralytics YOLO
  dataset with `4500` train images, `500` validation images, `16620` train boxes,
  and `1881` validation boxes.
- Trained `yolov8s.pt` for 60 epochs at `imgsz=640`, `batch=16`, `device=0` on a
  Tesla T4. Training completed in `1.297` hours.
- Final validation from Ultralytics: precision `0.964`, recall `0.936`,
  `mAP50=0.975`, `mAP50-95=0.905`.
- Docker `til test` on the full local novice CV set: `mAP@.5:.05:.95 = 0.885`,
  `mAP50 = 0.951`, no errors, `1250/1250` batches in `08:30`.
- Official hidden eval: `0.402 / 0.963`, `0 / 500` errors.

Interpretation: not an LTWH/output-format failure. A bbox-format bug would likely
score near `0.0` or cause result-loading/evaluation errors. The model improved
`0.044 → 0.402` with unchanged speed and no errors, proving the schema and
`xyxy → LTWH` adapter are working. The local→official gap (`0.885 → 0.402`) is
more likely hidden distribution shift, harder images/small objects, label/domain
differences, or confidence/recall behavior.

### yolo-baseline (14 May) — pretrained detector + robust fallback

Implemented the notes plan:

- Wrapped image decode and model inference so bad images return `[]` instead
  of erroring the request. `cv_server.py` now passes the evaluator `key` into
  the manager for debug logs.
- Added `PIL.ImageOps.exif_transpose(...).convert("RGB")` so grayscale/RGBA/EXIF
  oddities normalize before inference.
- Added Ultralytics `yolov8n.pt` inference loaded once in `CVManager.__init__`.
- Converts Ultralytics `xyxy` boxes to official LTWH `[l, t, w, h]`, clamps boxes
  to image bounds, drops invalid zero-area boxes, and emits plain Python
  `float`/`int` values.
- Added default YOLO-index to custom TIL category-id mapping. Set
  `CV_CATEGORY_MAP` to either a JSON dict like `{"4": 3}` or a path to that JSON
  file when tuning class mappings.
- Added `pillow` and `ultralytics` to [requirements.txt](requirements.txt), and
  Docker build now pre-caches `yolov8n.pt`.

Local limitation: this worktree does not contain `/home/jupyter/<track>/cv`.
Confirm the actual category IDs on GCP before trusting the score:

```bash
python - <<'PY'
import json, os
track = os.environ.get("TEAM_TRACK", "novice")
with open(f"/home/jupyter/{track}/cv/annotations.json") as f:
    ann = json.load(f)
print(ann.get("categories", [])[:20])
PY
```

Observed category IDs are custom:

```text
0 cargo aircraft, 1 commercial aircraft, 2 drone, 3 fighter jet,
4 fighter plane, 5 helicopter, 6 light aircraft, 7 missile, 8 truck,
9 car, 10 tank, 11 bus, 12 van, 13 cargo ship, 14 yacht,
15 cruise ship, 16 warship, 17 sailboat
```

Default mapping is now sparse and approximate:

```text
YOLO car      -> 9 car
YOLO airplane -> 1 commercial aircraft
YOLO bus      -> 11 bus
YOLO truck    -> 8 truck
YOLO boat     -> 13 cargo ship
```

This is only a baseline: COCO YOLO cannot distinguish drone/fighter/cargo/light
aircraft, missile/tank/van, or specific ship subclasses without fine-tuning.

### latest (12 May 03:52) — submission plumbing only

Just a sanity-check submission to confirm the service starts, the endpoint
accepts requests, the JSON shape is correct. No detection logic yet. Scored
0.000 as expected.

## What needs doing

In priority order (per [../SUMMARY.md "CV"](../SUMMARY.md)):

### 1. Tier 1 + Tier 2 push to ~0.70 (in progress, May 14)

Code-side changes already in this branch:

- `cv_manager.py` now emits a real `score` field per detection (Ultralytics
  confidence). Local `test/test_cv.py` hard-codes `score=1.0` so this is a no-op
  locally but may help official scoring if it consumes confidences.
- `cv_manager.py` reads `CV_AUGMENT` and `CV_HALF` envs. `CV_AUGMENT=1` enables
  Ultralytics flip+multi-scale TTA. `CV_HALF=1` runs FP16 on GPU.
- Dockerfile defaults flipped to `CV_AUGMENT=1`, `CV_HALF=1`. Resolution stays
  `imgsz=768` until the sweep confirms 896/1024 pays off; `CV_CONF=0.25` and
  `CV_IOU=0.50` unchanged so v2-best behavior is recoverable by env override.
- `training/cv/sweep_cv_http.py` extends conf range up to 0.60 (the test
  evaluator pins all detection scores to 1.0 so over-detection hurts), adds
  imgsz=1024 and a 0/1 `--augment` axis.
- `training/cv/train_v3.sh` wraps a Tier 2 YOLOv11m@1024 retrain with
  copy-paste/mosaic for small-object recall.

Tier 1 sweep partial results (May 14, in-flight on Workbench, first 8/144 runs
on `cv-yolo-v2-tier1` against `annotations_test.json`):

```text
[1] conf=0.20 iou=0.45 imgsz=768  aug=0   mAP=0.9014  small=0.647
[2] conf=0.20 iou=0.45 imgsz=768  aug=1   mAP=0.8986  small=0.628
[3] conf=0.20 iou=0.45 imgsz=896  aug=0   mAP=0.9016  small=0.741
[4] conf=0.20 iou=0.45 imgsz=896  aug=1   mAP=0.9042  small=0.746  ← best so far
[5] conf=0.20 iou=0.45 imgsz=1024 aug=0   mAP=0.8355  small=0.586  ← regression
[6] conf=0.20 iou=0.45 imgsz=1024 aug=1   mAP=0.8727  small=0.608  ← regression
[7] conf=0.20 iou=0.50 imgsz=768  aug=0   mAP=0.9014  (iou=0.45 vs 0.50 ≈ tie at 768)
```

Findings already actionable:

- `imgsz=1024` is **actively bad** with v2-best (768-trained) weights —
  resolution mismatch. Drop from sweep, rely on Tier 2's 1024-retrained model.
- `imgsz=896 + aug=1` is the leader with notably stronger small-object AP
  (0.746 vs 0.647 at 768). Local lift +0.045 over v2-best's 0.8589.
- Confidence/IoU axes still need to clear before locking the bake.

Open work (run on Workbench):

- Finish (or narrow) Tier 1 sweep, bake winning envs into Dockerfile, run
  `til test cv cv-yolo-v2-tier1-best` and `til submit`.
- Then run `bash training/cv/train_v3.sh` (~4-5h on T4) for Tier 2; do not run
  concurrent with the sweep on the same T4 (memory contention slows both).

### 2. If revisiting CV, improve hidden-distribution generalization

The next CV score gap is not plumbing. Candidate A/Bs:

- Try `yolov8m.pt` at `imgsz=768` if time and speed budget allow.
- Add class-balanced or aircraft-heavy sampling for rare/small classes.
- Add stronger small-object augmentation/cropping. Hard held-out small AP is
  improved (`0.5596`) but still the weakest area bucket.
- Keep HTTP inference sweeps, but current best is already `CV_CONF=0.25`,
  `CV_IOU=0.50`, `CV_IMGSZ=768`.

Do not spend time on COCO→TIL mapping guesses; the fine-tuned model already uses
the official 18-class label order.

### 3. Use the non-leaky eval tools before the next submit

Implemented on 14 May:

- [../training/cv/prepare_yolo_dataset.py](../training/cv/prepare_yolo_dataset.py)
  now writes train/val/test YOLO splits plus split-specific COCO annotations.
- [../training/cv/eval_cv_http.py](../training/cv/eval_cv_http.py) evaluates the
  running Docker HTTP service on a held-out split and prints global, area, and
  per-class AP.
- [../training/cv/sweep_cv_http.py](../training/cv/sweep_cv_http.py) restarts the
  Docker service across `CV_CONF`, `CV_IOU`, and `CV_IMGSZ` sweeps and ranks the
  held-out mAP results.

This fixes the main local-eval blind spot: the previous `0.885` local Docker mAP
was measured on the full local CV set after training on 90% of it. Future CV
model selection should use `/home/jupyter/cv_yolo_dataset/coco/annotations_test.json`
instead of full-set `til test` mAP.

### 4. Output-format checks to preserve

- YOLO training labels are normalized center-XYWH; that is only the training
  format.
- API output must stay pixel LTWH `[left, top, width, height]`.
- `cv_manager.py` currently converts Ultralytics `xyxy` to LTWH and clamps to
  image bounds. Keep this adapter unchanged unless local `til test` catches a
  regression.
- `cv_manager.py` emits a `score` field per detection (Ultralytics conf). The
  official `cv/README.md` schema does not list `score`, but `test/test_cv.py`
  silently appends `"score": 1.0` regardless. Including the real confidence is
  free upside if the official cloud evaluator consumes it, and a no-op
  otherwise.

### 5. Gotcha: `test_cv.py` pins `"score": 1.0` on every detection

`test/test_cv.py` (the local evaluator that powers `til test`) hardcodes
`score=1.0` on every box before running pycocotools. Standard COCO mAP
integrates the precision-recall curve over confidences; pinning all scores to
1.0 turns mAP into a yes/no precision metric where every false positive at
score 1.0 directly hurts you, regardless of underlying conf.

This is why the optimal `CV_CONF` is **higher** than the COCO-default 0.001:
lower conf adds FPs as score-1.0 ties and tanks precision. We cannot
unilaterally change the evaluator (it's competition scaffolding), so the
best we can do is sweep conf upward in the 0.20-0.60 range and pick the F1
optimum on the held-out hard split. Whether the official cloud evaluator does
the same or actually consumes our `score` field is not knowable without
submitting, but the env-driven design lets us pivot either way.

## State

CV official score is now `0.549 / 0.960` with `0 / 500` errors. This is a real
ML improvement over `cv-yolo-ft-v1` (`0.402 / 0.963`) and materially raises the
estimated blended qualifier score. Further CV gains are possible, but NLP/AE
probably have better ROI unless a quick `yolov8m` or targeted small-object /
aircraft A/B is cheap.

## Reproducibility / pointers

- Manager source: [src/cv_manager.py](src/cv_manager.py)
- HTTP server (don't edit): [src/cv_server.py](src/cv_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#cv](../SUMMARY.md)
