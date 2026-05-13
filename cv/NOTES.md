# CV — notes & history

Last updated: 13 May 2026

Per-task working log for CV (object detection). For input/output spec see
[README.md](README.md). For submission history across all tasks see
[../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`latest` — official 0.000 / 0.981 (12 May 03:52 SGT, 4 of 500 errors).**

Empty-detection baseline. Service starts, endpoint returns valid-shape JSON,
but predicts no objects → 0.000 accuracy. The `4 / 500` errors are the
priority before any model work — even a working detector won't help if
schema bugs cause errors on some inputs.

## What our model runs on

Nothing yet. Baseline manager returns `[]` (empty detection list) for every
image. From [src/cv_manager.py](src/cv_manager.py):

```python
def cv(image_bytes: bytes) -> list[dict]:
    return []   # no detections
```

Valid JSON shape, scores 0.

## Submission history

```text
Tag       Submitted          Score   Speed   Errors    Notes
latest    12/05 03:52        0.000   0.981   4 / 500   Empty-detection baseline, 4 inputs erroring
```

## Detailed timeline

### latest (12 May 03:52) — submission plumbing only

Just a sanity-check submission to confirm the service starts, the endpoint
accepts requests, the JSON shape is correct. No detection logic yet. Scored
0.000 as expected.

## What needs doing

In priority order (per [../SUMMARY.md "CV"](../SUMMARY.md)):

### 1. Fix the `4 / 500` errors first

We don't yet know what the 4 erroring inputs are. Possibilities:
- Corrupted / non-JPEG image bytes that `PIL.Image.open()` chokes on
- Edge-case image dimensions or color modes (grayscale, RGBA)
- Request payload variations the manager doesn't handle

Cheapest diagnostic: wrap the manager body in a `try/except` that returns
`[]` on any decode error AND logs the input key/size. Then rebuild, submit,
and check the debug log via the submission's Debug URL.

A 4-error-free submission with `[]` predictions would already be a "clean"
baseline to compare model improvements against.

### 2. Drop in a pretrained detector

Default choice per SUMMARY.md is **Ultralytics YOLOv8/v11** (easiest), or
**RT-DETR** if accuracy beats speed. Pretrained COCO weights first; only
fine-tune if categories don't match the eval's `category_id` set.

Wire-up:
- Load model once in `CVManager.__init__`
- In `cv(image_bytes)`:
  - `PIL.Image.open(io.BytesIO(image_bytes))`
  - Run detector
  - **Convert xyxy → xywh** (COCO style: top-left + width/height, NOT corners). The output schema is `[x, y, w, h]` per `category_id`. Getting this wrong scores 0 with a working model.
  - Apply confidence threshold (start ~0.25, tune)
- Add `ultralytics`, `pillow` to [requirements.txt](requirements.txt)
- Use a CUDA base image in [Dockerfile](Dockerfile) if eval has GPU access (check the wiki)

### 3. Class mapping is the actual work

The eval has its own `category_id` set. The COCO IDs from YOLO won't match
directly. There must be a lookup table from the dataset metadata — find it
in the training images folder or wiki, build a `coco_id → eval_id` dict,
apply at inference time. **Wrong class mapping scores 0 even with perfect
boxes.**

### 4. Stretch (post-baseline)

- Fine-tune on the provided training set if categories diverge significantly from COCO.
- TTA (test-time augmentation: flip + scale ensemble) if speed budget allows.
- NMS tuning per class.

## State

CV is at 0.000 official. **Any working detector + correct class mapping is a
huge raw-score jump** (potentially 0.5–0.7 for a vanilla pretrained
YOLOv8-on-COCO if classes overlap well). Given the 20% qualifier weight,
even a mediocre CV submission contributes meaningfully more than further
ASR optimization.

## Reproducibility / pointers

- Manager source: [src/cv_manager.py](src/cv_manager.py)
- HTTP server (don't edit): [src/cv_server.py](src/cv_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#cv](../SUMMARY.md)
