# CV — notes & history

Last updated: 14 May 2026

Per-task working log for CV (object detection). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#cv).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`yolo-til-map-v2` — official 0.044 / 0.961 (14 May 01:56 SGT, 0 of 500 errors).**

Pretrained YOLOv8n + custom sparse COCO→TIL mapping. This fixed the old
`4 / 500` errors and confirms the service/schema/fallback plumbing is clean.
Score is still low because COCO YOLO cannot distinguish the 18 TIL-specific
military/vehicle/ship subclasses. Next step: fine-tune a custom detector on the
provided CV annotations, not more mapping guesses.

## What our model runs on

Current branch implementation: a schema-safe Ultralytics YOLO baseline in
[src/cv_manager.py](src/cv_manager.py).

```python
def cv(image_bytes: bytes, key=None) -> list[dict]:
    # Decode robustly, run YOLO when available, convert xyxy -> xywh.
    # Any decode/model failure returns [] and logs key/byte size.
```

Default model is `yolov8n.pt`, cached during Docker build so runtime does not
depend on network access. Outputs use the custom Workbench category IDs by
default for the COCO classes YOLO can see, with `CV_CATEGORY_MAP` available as a
JSON override for further tuning.

## Submission history

```text
Tag              Submitted       Score   Speed   Errors    Notes
latest           12/05 03:52     0.000   0.981   4 / 500   Empty-detection baseline, 4 inputs erroring
yolo-til-map-v2  14/05 01:56     0.044   0.961   0 / 500   YOLOv8n + sparse COCO→TIL map; clean serving, weak domain fit
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

### 1. Fix the `4 / 500` errors first

Status: fixed by `yolo-til-map-v2` (`0 / 500` errors officially). The robust
manager fallback and RGB/EXIF normalization did their job. Keep this serving
path as the safe baseline while changing model weights.

### 2. Pretrained detector baseline

Status: implemented with **Ultralytics YOLOv8n**. It is useful only as a
serving/sanity baseline: official score `0.044` proves COCO labels are too
mismatched for the 18-class target list.

Wire-up:
- Load model once in `CVManager.__init__`
- In `cv(image_bytes)`:
  - `PIL.Image.open(io.BytesIO(image_bytes))`
  - Run detector
  - **Convert model boxes → LTWH** (official top-left + width/height, NOT corners and NOT center-XYWH). The output schema is `[l, t, w, h]` per `category_id`. Getting this wrong scores 0 with a working model.
  - Apply confidence threshold (start ~0.25, tune)
- Added `ultralytics`, `pillow` to [requirements.txt](requirements.txt)
- [Dockerfile](Dockerfile) already uses an NVIDIA PyTorch base and caches
  `yolov8n.pt` during build.

### 3. Class mapping was the baseline blocker; fine-tuning is now the work

Status: default mapping targets the official 18 labels. It is intentionally
sparse because the pretrained COCO model only has broad `car/bus/truck/airplane/boat`
classes. `CV_CATEGORY_MAP` can still override mappings, but the official
mAP@.5:.05:.95 target requires a detector trained on the TIL category list.
**Wrong class mapping scores 0 even with perfect boxes; correct mapping alone
is still insufficient when the detector cannot see the target subclasses.**

### 4. Stretch (post-baseline)

- Fine-tune on the provided training set if categories diverge significantly from COCO.
- TTA (test-time augmentation: flip + scale ensemble) if speed budget allows.
- NMS tuning per class.

## State

CV official score is now `0.044 / 0.961` with `0 / 500` errors. That is a clean
serving baseline but a weak detector. Next high-ROI work is machine learning:
convert `/home/jupyter/novice/cv/annotations.json` to YOLO format, fine-tune a
YOLO model for the 18 TIL labels, copy the best checkpoint into `cv/models/`, set
`CV_MODEL_PATH` to that checkpoint in the Docker image, then `til build/test/submit`
with a tag like `cv-yolo-ft-v1`.

## Reproducibility / pointers

- Manager source: [src/cv_manager.py](src/cv_manager.py)
- HTTP server (don't edit): [src/cv_server.py](src/cv_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#cv](../SUMMARY.md)
