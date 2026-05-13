# CV — notes & history

Last updated: 14 May 2026

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

Current branch implementation: a schema-safe Ultralytics YOLO baseline in
[src/cv_manager.py](src/cv_manager.py).

```python
def cv(image_bytes: bytes, key=None) -> list[dict]:
    # Decode robustly, run YOLO when available, convert xyxy -> xywh.
    # Any decode/model failure returns [] and logs key/byte size.
```

Default model is `yolov8n.pt`, cached during Docker build so runtime does not
depend on network access. Outputs use standard COCO category IDs by default
(`YOLO class index -> COCO category_id`), with `CV_CATEGORY_MAP` available as a
JSON override if the Workbench annotations use a different ID space.

## Submission history

```text
Tag       Submitted          Score   Speed   Errors    Notes
latest    12/05 03:52        0.000   0.981   4 / 500   Empty-detection baseline, 4 inputs erroring
```

## Detailed timeline

### yolo-baseline (14 May) — pretrained detector + robust fallback

Implemented the notes plan:

- Wrapped image decode and model inference so bad images return `[]` instead
  of erroring the request. `cv_server.py` now passes the evaluator `key` into
  the manager for debug logs.
- Added `PIL.ImageOps.exif_transpose(...).convert("RGB")` so grayscale/RGBA/EXIF
  oddities normalize before inference.
- Added Ultralytics `yolov8n.pt` inference loaded once in `CVManager.__init__`.
- Converts Ultralytics `xyxy` boxes to COCO-style `[x, y, w, h]`, clamps boxes
  to image bounds, drops invalid zero-area boxes, and emits plain Python
  `float`/`int` values.
- Added default YOLO-index to COCO-category-id mapping. If Workbench
  `annotations.json` uses different IDs, set `CV_CATEGORY_MAP` to either a JSON
  dict like `{"0": 1, "1": 2}` or a path to that JSON file.
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

If those IDs are standard COCO IDs, no override is needed.

### latest (12 May 03:52) — submission plumbing only

Just a sanity-check submission to confirm the service starts, the endpoint
accepts requests, the JSON shape is correct. No detection logic yet. Scored
0.000 as expected.

## What needs doing

In priority order (per [../SUMMARY.md "CV"](../SUMMARY.md)):

### 1. Fix the `4 / 500` errors first

Status: implemented defensive fallbacks, pending Workbench submission to confirm
the errors are gone. We don't yet know what the 4 erroring inputs are.
Possibilities:
- Corrupted / non-JPEG image bytes that `PIL.Image.open()` chokes on
- Edge-case image dimensions or color modes (grayscale, RGBA)
- Request payload variations the manager doesn't handle

Diagnostic now in place: manager returns `[]` on decode/model errors and logs
the input `key` plus byte size. Rebuild, submit, and check the Debug URL if
errors remain.

A 4-error-free submission with `[]` predictions would already be a "clean"
baseline to compare model improvements against.

### 2. Drop in a pretrained detector

Status: implemented with **Ultralytics YOLOv8n**. Pretrained COCO weights first;
only fine-tune if categories don't match the eval's `category_id` set.

Wire-up:
- Load model once in `CVManager.__init__`
- In `cv(image_bytes)`:
  - `PIL.Image.open(io.BytesIO(image_bytes))`
  - Run detector
  - **Convert xyxy → xywh** (COCO style: top-left + width/height, NOT corners). The output schema is `[x, y, w, h]` per `category_id`. Getting this wrong scores 0 with a working model.
  - Apply confidence threshold (start ~0.25, tune)
- Added `ultralytics`, `pillow` to [requirements.txt](requirements.txt)
- [Dockerfile](Dockerfile) already uses an NVIDIA PyTorch base and now caches
  `yolov8n.pt` during build.

### 3. Class mapping is the actual work

Status: default mapping is standard COCO. Still verify against Workbench
`annotations.json`. If the eval has its own `category_id` set, provide
`CV_CATEGORY_MAP` as a JSON dict/list and rebuild. **Wrong class mapping scores
0 even with perfect boxes.**

### 4. Stretch (post-baseline)

- Fine-tune on the provided training set if categories diverge significantly from COCO.
- TTA (test-time augmentation: flip + scale ensemble) if speed budget allows.
- NMS tuning per class.

## State

CV official score is still last-known 0.000 until this branch is built and
submitted. The local implementation should now produce non-empty detections
when the model loads and should return clean empty lists for bad inputs.

## Reproducibility / pointers

- Manager source: [src/cv_manager.py](src/cv_manager.py)
- HTTP server (don't edit): [src/cv_server.py](src/cv_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#cv](../SUMMARY.md)
