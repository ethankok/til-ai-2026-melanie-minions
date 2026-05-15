# CV — notes & history

Last updated: 15 May 2026 12:00 SGT

Per-task working log for CV (object detection). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#cv).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`cv-yolo-v2-tier1-best` — official 0.556 / 0.956 (14 May 17:10 SGT, 0 of 500 errors). STILL ON LEADERBOARD.**

v11m@1024 (`cv-yolo11m-v3-pre`) regressed to 0.376/0.955 on 15 May. Tier1 stays
as the active CV image via highest-score retention.

**Next candidate: v11m@1280 inference** — hard held-out 0.9141 (vs tier1's 0.9049,
+0.009). Speed with TTA is ~0.60 (too slow). Need `aug=0` sweep to check if
accuracy holds without TTA before deciding to ship.

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
Tag                    Submitted       Score   Speed   Errors    Notes
latest                 12/05 03:52     0.000   0.981   4 / 500   Empty-detection baseline, 4 inputs erroring
yolo-til-map-v2        14/05 01:56     0.044   0.961   0 / 500   YOLOv8n + sparse COCO→TIL map; clean serving, weak domain fit
cv-yolo-ft-v1          14/05 03:53     0.402   0.963   0 / 500   YOLOv8s fine-tuned on 18 TIL labels; local mAP50-95 0.885
cv-yolo-v2-best        14/05 14:00     0.549   0.960   0 / 500   YOLOv8s 768 hard-split retrain + tuned inference; local til test 0.8839, hard held-out 0.8589
cv-yolo-v2-tier1-best  14/05 17:10     0.556   0.956   0 / 500   v2-best weights + TTA + imgsz=896 + iou=0.60 + score field; hard held-out 0.9049, local til test 0.8505. NEW HIGH (+0.007); STILL ON LEADERBOARD.
cv-yolo11m-v3-pre      15/05 11:34     0.376   0.955   0 / 500   YOLOv11m@1024 fully trained 120ep; val 0.937, hard held-out 0.8673. REGRESSED -0.180; matched-imgsz lost to v8s+upscaled inference.
cv-yolo11m-v3-1280     (not yet submitted)  —   —     —         Hard held-out 0.9141 (aug=1, imgsz=1280); small AP 0.742. +0.047 vs 1024. Speed with TTA ~0.60 (too slow). aug=0 sweep needed before deciding to ship.
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

### 1. Tier 1 + Tier 2 push to ~0.70 (Tier 1 SHIPPED at 0.556, Tier 2 in progress)

Tier 1 result (14 May 17:10): `cv-yolo-v2-tier1-best` shipped at
**0.556 / 0.956**, +0.007 over v2-best. Same weights, env-only changes:

- `CV_AUGMENT=1` (Ultralytics flip+multi-scale TTA) — wins on small/dense scenes
- `CV_HALF=1` (FP16 inference) — speed-neutral
- `CV_IMGSZ=896` — best resolution at the imgsz vs accuracy plateau
- `CV_IOU=0.60`, `CV_CONF=0.20` — picked from sweep
- `score` field emitted per detection (Ultralytics conf, not pinned 1.0)

Sweep top results (38 of 144 runs before terminating; the trend was clear):

```text
mAP=0.9049  conf=0.20 iou=0.60 imgsz=896 aug=1   ← shipped
mAP=0.9044  conf=0.20 iou=0.55 imgsz=896 aug=1
mAP=0.9042  conf=0.20 iou=0.45 imgsz=896 aug=1
mAP=0.9041  conf=0.25 iou=0.45 imgsz=896 aug=1
mAP=0.9016  conf=0.20 iou=0.45 imgsz=896 aug=0
mAP=0.8727  conf=0.20 iou=0.45 imgsz=1024 aug=1  ← imgsz=1024 dead with 768-trained weights
mAP=0.8355  conf=0.20 iou=0.45 imgsz=1024 aug=0
```

Confirmed:
- `imgsz=896 + aug=1` is the regime; `imgsz=1024` regresses by 0.07.
- `iou` plateau in 0.45-0.60 (Δ < 0.001).
- `conf=0.20` slightly beats `conf=0.25`.
- Local→official gap stayed wide (0.9049 → 0.556 = 0.349). Tier 2 is needed
  for any meaningful jump toward 0.70.

Tier 2 status (SHIPPED & REGRESSED, May 15):

- Trainer: `training/cv/train_v3.sh`, YOLOv11m @ imgsz=1024, batch=6, AdamW
  cos_lr, copy_paste=0.30 + mosaic=1.0 + mixup=0.15.
- Wallclock estimate revised: original 4-5h was wrong. YOLOv11m@1024 actually
  runs ~6:30/epoch on T4, so 120 epochs is ~13h. v8s@768 was 80 epochs in 2.3h
  for reference; m vs s + 768 vs 1024 + 80 vs 120 epochs compound to ~5.6x.
- First run died at epoch 58/120 from CUDA OOM caused by docker squatter
  containers from the earlier sweep eating ~10GB of VRAM. `best.pt` was saved
  (epoch 56, val mAP50-95 = 0.895).
- Resumed from `last.pt` and ran to completion at 120 epochs (~7h after
  resume). Final Ultralytics val mAP50-95 = **0.937**, mAP50 = **0.994** on
  the 500-image val split. Per-class lowest was `warship` at 0.862, top was
  `car` at 0.988.
- Sweep on the fully-trained `best.pt` against the hard test split topped at
  `mAP50-95 = 0.8673` (`conf=0.001 iou=0.70 imgsz=1024 aug=1`). That is
  **0.038 below v8s tier1's hard held-out 0.9049**. Small-object AP was the
  killer: v11m `0.587` vs tier1 `0.746` on the hard test split.
- Submitted as `cv-yolo11m-v3-pre` (15/05 11:34): **0.376 / 0.955**, 0/500
  errors. **REGRESSED -0.180 vs tier1's 0.556**. Leaderboard keeps the
  higher score, so tier1 stays as the active CV image.

Post-mortem — why v11m lost:

1. **Resolution mismatch.** v8s tier1 was trained at 768 and inferenced at
   896 (UP from training); the upscaling helped small objects (+0.04 in the
   v8s sweep). v11m was trained at 1024 and inferenced at 1024 (matched).
   We never tested v11m at imgsz=1280, which would likely lift it.
2. **Small-object hit comes from high-IoU bins.** v11m's mAP50 was 0.997
   (essentially perfect detection), so the loss is in the high-IoU
   precision bins (≥0.75) — exactly what higher inference resolution helps.
3. **Bigger model + matched-imgsz < smaller model + upscaled-imgsz** on
   this dataset's small-object distribution. Counter-intuitive but
   reproducible: hidden eval correlated tightly with hard held-out
   (0.8673 → 0.376), so the gap diagnosis is solid.
4. **Local→official gap stayed at ~0.49** (worse than tier1's 0.349). The
   hidden eval is even harsher on bbox precision than the hard held-out
   suggests.

Open lever NOT yet tested: imgsz=1280 inference on v11m. Speed cost
~30-35ms/img × 500 = ~18s, still 0.99 speed. Worth a sweep before fully
parking v11m.

**imgsz=1280 sweep result (15 May, after v3-pre submission):**

```text
mAP=0.9141  conf=0.001 iou=0.70 imgsz=1280 aug=1   ← best
mAP=0.9140  conf=0.05  iou=0.70 imgsz=1280 aug=1
mAP=0.9140  conf=0.10  iou=0.70 imgsz=1280 aug=1
mAP=0.9136  conf=0.001 iou=0.50 imgsz=1280 aug=1
(all imgsz=1024 rows: 0.8673 max — confirmed dead)
```

Hard held-out 0.9141 at imgsz=1280 (+0.047 vs 1024, +0.009 vs tier1's 0.9049).
Small AP recovered to 0.742 (vs tier1's 0.746 — essentially matched).

**Speed problem:** aug=1 at imgsz=1280 runs at ~1.04s/img → 520s for 500 images
→ speed score ~0.71. Blended: 0.75×0.60 + 0.25×0.71 = **0.628** vs tier1's
0.75×0.556 + 0.25×0.956 = **0.656**. TTA is killing the blended score.

**Next step: aug=0 sweep at imgsz=1280.** Without TTA, inference is ~0.35s/img
→ 175s → speed ~0.90. If aug=0 holds mAP ≥ 0.88, blended would be
0.75×0.58 + 0.25×0.90 = **0.660** — beats tier1. Run:

```bash
python training/cv/sweep_cv_http.py \
  --image melanie-minions-cv:cv-yolo11m-v3-pre \
  --data-dir /home/jupyter/novice/cv \
  --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \
  --out-dir /home/jupyter/cv_eval_sweeps/v11m-1280-noaug \
  --conf 0.001,0.05,0.10 \
  --iou 0.50,0.70 \
  --imgsz 1280 \
  --augment 0
# 6 runs * ~50s = ~5 min
```

The sweep script now records per-run wall-clock time and prints an estimated
`0.75*mAP + 0.25*speed` blend extrapolated to 500 images / 30 minutes. Use the
"Top estimated blended results" block, not just the mAP ranking, when deciding
whether v11m@1280-noaug is worth submitting.

If revisiting CV, the actual ROI levers are now (in order):
- Try the v11m@1280 sweep and re-submit if it clears 0.91 hard held-out.
- Stop training and accept tier1's 0.556 as the CV high. CV is 20% of
  qualifier; AE and NLP have higher ROI per per-team-member-hour.

If the resume crashes, do NOT restart from epoch 0. Resume again from
`last.pt`. Ultralytics handles the cosine schedule continuation correctly.

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
