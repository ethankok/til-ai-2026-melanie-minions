# CV — notes & history

## 24 May 2026 — Dockerfile bumped to CV_IMGSZ=1280

Commit `543516e` flipped `cv/Dockerfile` from `CV_IMGSZ=1024` to
`CV_IMGSZ=1280` so the next submission of the yolo11l-1280-alldata-final-v2
weights actually inferences at the resolution the model was trained at.
The already-shipped v2 row below was submitted at `CV_IMGSZ=1024` and has
not been re-submitted yet — when the next `til submit cv` goes out, it
will be at native 1280.

## yolo11l-1280-alldata-final-v2 — all-data fine-tuned at imgsz=1280 (SHIPPED, pending score)

**Cloud `[pending] / [pending]`, local test `0.988` mAP on novice validation set (small AP: 0.793, medium: 0.974, large: 0.997).**
- Fine-tuned YOLO11l from the `yolo11l-1024-alldata-final-v1` checkpoint for 36 epochs natively at `imgsz=1280`.
- Trained on the complete dataset (`cv_yolo_dataset_all`) to maximize representation.
- Submitted at `CV_IMGSZ=1024` (Dockerfile was not bumped before this submission). Dockerfile is now at 1280 for the next submission — see top note.

## yolo11l-896-plusval-v1 — official 0.640 / 0.954 (accuracy high)

**Cloud `0.640 / 0.954` official (22 May).**
- The plusval recipe folds the old val split back into training while keeping the hard test split for sanity (old train had only 34 cruise ships / 52 warships / 56 yachts vs ship-heavy val+test).
- Served at `CV_IMGSZ=896`.

20 May update: `cv/src/cv_manager.py` now has an opt-in Hugging Face OWLv2
zero-shot detector path behind `CV_MODEL_FAMILY=owlv2`. It maps the 18 official
TIL category prompts to LTWH output, supports custom `CV_OWLV2_PROMPTS`, and
keeps the YOLO path as the Docker default until Workbench `til test` /
submission numbers prove OWLv2 helps. For offline evaluator runs, prefetch/bake
the model cache during Docker build or point `CV_OWLV2_MODEL_ID` at a copied
local model directory, then keep `CV_OWLV2_LOCAL_FILES_ONLY=1`.

Failed follow-up: `ry_v3_adaptive` tried low-conf base (`CV_CONF=0.05`,
`CV_IOU=0.70`, `CV_CROSS_CLASS_NMS_IOU=0`) plus an adaptive down-weighted TTA
rescue pass on dense images (`CV_SECOND_MIN_DETECTIONS=7`). It overfit the
held-out JSON: offline ensemble-lab scoring was 0.9270 mAP, but `til test`
fell to 0.8513 and official fell to 0.571 / 0.958 on 19 May 22:08 SGT.
Keep the second-pass code opt-in only; `cv/Dockerfile` is restored to `ry-v2`.

Next checkpoint path: `training/cv/build_final_yolo_dataset.py` plus
`training/cv/train_v5_plusval.sh`. This folds the old val split into training
while keeping the old hard test split for sanity. The reason is split imbalance:
old train has only 34 cruise ships / 52 warships / 56 yachts, while val+test are
much more ship-heavy and dense.

Previous 16 May note: **CV RE-PARKED at `cv-yolo-v2-tier1-best`
0.556/0.956 after Phase C.1 didn't transfer.** Augmented training (JPEG + native
tile crops) lifted hard held-out +0.04 (0.948 vs tier1's 0.905) but the
local→cloud gap WIDENED from 0.349 → 0.395 — augmentation overfit to the
small-object-dense distribution, didn't generalize. v8s/v11m family is now
confirmed at-ceiling around cloud 0.55-0.56. Hitting 0.7 would require either
a different architecture family (RT-DETR / YOLOv9) or accepting CV at this
number and reallocating to AE/NLP. Tier1 stays shipped.

Per-task working log for CV (object detection). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#cv).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`yolo11-optimized-v3` — high-precision + TTA at 1280 (SHIPPED, pending score)**
**Submitted: 24/05 (By [Your Name])**
- Rebuilt from the `yolo11l-1280-alldata-final-v2` checkpoint.
- **Fixed:** The v2 submission accidentally served at 1024px. The Dockerfile was already bumped to 1280px, so this build finally serves at native resolution.
- **Precision Tweak:** Bumped `CV_CONF` from 0.15 to 0.20 to prune false positives.
- **Recall Tweak:** Flipped `CV_AUGMENT` from 0 to 1 to enable Test-Time Augmentation (TTA). Expecting a speed hit but a significant mAP lift on hidden/small objects.

**`yolo11l-1280-alldata-final-v2` — submitted 23/05 20:55, score pending.**
Last scored high: **`yolo11l-896-plusval-v1` — official 0.640 / 0.954.**
**`yolo11l-1280-alldata-final-v2` — submitted 23/05 20:55, score pending.**
Last scored high: **`yolo11l-896-plusval-v1` — official 0.640 / 0.954.**

Dockerfile now serves at `CV_IMGSZ=1280` (bumped 24 May in `543516e` —
matches the v2 checkpoint's training resolution). The already-shipped v2
row above was submitted at 1024; the next submission will be at native
1280. Other serving knobs stay at the ruiyang-sweep defaults:

```text
CV_CONF=0.15
CV_IOU=0.55
CV_IMGSZ=1280     # was 896 for plusval-v1, 1024 for the first v2 submit
CV_AUGMENT=0
CV_CROSS_CLASS_NMS_IOU=0.97
CV_SECOND_PASS=0
```

Previous CV highs: `ry-v2` 0.608 / 0.961, `ruiyang-v1` 0.588 / 0.955, parked
v8s `cv-yolo-v2-tier1-best` 0.556 / 0.956.

Tier1 + Tier2 attempts (15-16 May) all regressed cloud or were blended-flat:

| Tag | Cloud | Hard held-out | Notes |
|---|---:|---:|---|
| `ry-v2` (live) | 0.608 / 0.961 | 0.9234 | ruiyang-v1 weights, conf=0.15 iou=0.55 imgsz=896 aug=0 cross_nms=0.97 |
| `ruiyang-v1` | 0.588 / 0.955 | 0.9125 | new high before core sweep, real-score HTTP eval |
| `cv-yolo-v2-tier1-best` (live) | 0.556 / 0.956 | 0.9049 | tier1 |
| `cv-yolo11m-v3-pre` (15/05) | 0.376 / 0.955 | 0.8673 | v11m@1024 matched-imgsz |
| `v11m-1280-noaug-v1` (16/05) | 0.474 / 0.949 | 0.9088 | v11m at 1280 aug=0 |
| `cv-augc1-v4` (16/05 15:28) | 0.553 / 0.962 | 0.948 | Phase C.1, mismatched Dockerfile config (imgsz=768) |
| `cv-augc1-v4-1280` (16/05 18:28) | 0.553 / 0.959 | 0.948 | Phase C.1, matched config (imgsz=1280) |
| v8s-1024 retrain (16/05) | NOT SUBMITTED | 0.8217-0.8370 | cp=0.40 toxic regression |

**Current read**: `ry-v2` is worth keeping. Inference-only gains within this
weight family may still exist, but the easy row has now been shipped. Material
next gains likely require a different architecture family or a fundamentally
different distribution recipe.

## Calibration learned (for any future CV submission)

```text
Backbone     Local→cloud gap on hard held-out
v8s          ~0.35  (tier1: 0.9049 → 0.556)
v11m         ~0.44  (v3-pre: 0.8673 → 0.376; v11m@1280: 0.9088 → 0.474)
```

The two v11m data points agree on absolute cloud (`0.376` vs `0.474`) and on gap
(`0.491` vs `0.435`). **The gap is a function of architecture generalization to
the hidden distribution, not of resolution.**

Implication: to beat tier1 cloud 0.556 with a v11m-class submission would need
hard held-out ≥ 0.99 (out of reach). v8s-class needs hard ≥ 0.91 (current tier1
is 0.9049). A different backbone family has an unknown gap; discovery requires
submitting, and that exploratory budget is spent.

**Speed projection was wrong on the cloud side**: cloud GPU is meaningfully faster
than the workbench T4. No-TTA inference at 1280 hits cloud speed 0.949, not the
0.85-0.92 we projected. Cloud speed is no longer a CV constraint for any
reasonable inference resolution.

## Phase C.1 — augmented training didn't transfer (16 May 19:00 SGT)

### Pipeline executed as planned

1. `training/cv/build_aug_dataset.py` produced ~3× train set: originals + JPEG-recompressed copies (q=40-85) + 1024×1024 native-resolution tile crops (≥0.5 visible-area filter).
2. `training/cv/train_v4.sh` trained v8s at imgsz=1024, scale=0.80, copy_paste=0.10, 80 epochs. Final val mAP50-95 ≈ 0.95.
3. Submitted twice — first at Dockerfile default (model trained at 1024, served at 768 → mismatch), then with matched config (imgsz=1280, conf=0.001, iou=0.7, aug=0 baked into ENV).

### Numbers across distributions

| Image set | augc1-v4 | tier1 | Δ |
|---|---:|---:|---:|
| Hard held-out 500 (sweep, small-object dense) | **0.948 / 0.779 small** | 0.905 / 0.746 small | **+0.043** |
| Full local 1250 (`til test`, easier/sparser) | 0.739 / 0.192 small | ~0.85 / high small | **-0.11** |
| **Cloud 500 (hidden)** | **0.553 / 0.959** | 0.556 / 0.956 | **-0.003** |

### Interpretation

- **Training worked on its target.** +0.04 hard held-out is a real improvement, exactly what JPEG + tile-crop augmentation was supposed to deliver.
- **Cloud distribution is closer to full-local than to hard.** Augmentation taught the model to crush dense-small-object scenes at the cost of sparse easy scenes; cloud has more of the latter so wins/losses cancel.
- **Gap WIDENED**: tier1 gap was 0.349 (0.905 → 0.556). augc1 gap is 0.395 (0.948 → 0.553). Augmenting on hard distribution *specialized* further from cloud, not closer.
- **`til test` 0.739 with small AP 0.192 is partly a measurement artifact**: at `CV_CONF=0.001`, test_cv.py pins all scores to 1.0 before pycocotools, flooding low-confidence detections that drown the PR curve on sparse easy images. Sweep's pycocotools at the same env reports 0.948 because it uses real conf scores. Cloud uses something closer to the sweep's behavior (cloud=0.553, not crashed), so the +0.04 hard-set gain just didn't show up.

### Workbench command (for reference if revisiting)

```bash
# 1. Build augmented dataset (~5-10 min)
python training/cv/build_aug_dataset.py \
  --in-dir /home/jupyter/cv_yolo_dataset \
  --out-dir /home/jupyter/cv_yolo_dataset_augc1 \
  --crops-per-image 1 --jpeg-per-image 1

# 2. Train (~6h on T4)
bash training/cv/train_v4.sh

# 3. Deploy + sweep + submit
cp /home/jupyter/cv_runs/til-yolov8s-1024-augc1-v4/weights/best.pt cv/models/best.pt
til build cv cv-augc1-v4
python training/cv/sweep_cv_http.py \
  --image melanie-minions-cv:cv-augc1-v4 \
  --data-dir /home/jupyter/novice/cv \
  --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \
  --out-dir /home/jupyter/cv_eval_sweeps/augc1-v4 \
  --conf 0.001,0.05,0.20 --iou 0.50,0.60,0.70 \
  --imgsz 1024,1280 --augment 0,1
til submit cv cv-augc1-v4
```

## Phase C.0 — gap diagnostic (16 May 05:30 SGT)

`training/cv/gap_diagnose.py` re-encodes the hard held-out 500 in-memory under
each transform and re-scores tier1.

Premise: all five inside-the-box recipe levers (Pass A confusion, `CV_CONF`
sweep, tiled inference, v8s-1024, v11m@1280) are dead. The gap was *not* tested
against JPEG quality shift or scale/resolution shift between training set and
hidden eval.

### Results

Baseline this run: mAP **0.9049** / small AP 0.7463 (note: +0.01 mAP and +0.10
small AP higher than Pass A's earlier 0.8947/0.6434 — same image, same env vars
in theory; flagged as baseline anomaly below).

| Transform | Total mAP | Small AP | Medium AP | Large AP | Δ total | Δ small |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9049 | 0.7463 | 0.8245 | 0.9274 | — | — |
| jpeg-q70 | 0.8934 | 0.6168 | 0.8148 | 0.9194 | −0.011 | **−0.130** |
| jpeg-q50 | 0.8900 | 0.5972 | 0.8060 | 0.9156 | −0.015 | **−0.149** |
| jpeg-q30 | 0.8786 | 0.6495 | 0.7840 | 0.9027 | −0.026 | −0.097 |
| downsample-2x | 0.8928 | 0.6287 | 0.7783 | 0.9212 | −0.012 | **−0.118** |
| downsample-3x | 0.8612 | 0.6428 | 0.7182 | 0.8987 | −0.044 | −0.104 |
| jpeg50-down2 | 0.8776 | 0.5598 | 0.7418 | 0.9072 | −0.027 | **−0.187** |

### Interpretation

- Auto-verdict ("JPEG aug dead, resolution dead") was misleading — it thresholded total mAP at 0.05. Real signal: **small AP drops 10-19 points under every shift**, and Pass A established small AP is where the cloud loses. Both axes are real contributors to the cloud gap; just not visible in total-mAP rollup.
- Total-mAP drops sum to at most -0.07; cloud gap is 0.349. JPEG + resolution shifts account for **≤ 20%** of cloud gap.
- Remaining 80% is most likely **different scene content** (heavier small-object bias), not a transform of similar ones.
- Realistic cloud ceiling for Phase C augmentation alone: **0.58-0.63**, not 0.70. To hit 0.70 also need backbone change (RT-DETR / YOLOv9) or wholesale distribution-shift recipe.

### Workbench command

```bash
docker run -d --rm --name cv-tier1 -p 5002:5002 \
  -e CV_MODEL_PATH=/workspace/models/cv/best.pt \
  -e CV_CONF=0.20 -e CV_IOU=0.60 -e CV_IMGSZ=896 \
  -e CV_AUGMENT=1 -e CV_HALF=1 \
  --gpus all melanie-minions-cv:cv-yolo-v2-tier1-best

python training/cv/gap_diagnose.py \
  --data-dir /home/jupyter/novice/cv \
  --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \
  --out /home/jupyter/cv_eval_sweeps/gap-diagnose.json
docker rm -f cv-tier1
```

Runtime ~35 min all 7 transforms; `--transforms baseline,jpeg-q50,downsample-2x`
is a ~15-min smoke.

### Baseline anomaly (non-blocking)

Pass A on 16 May 02:30 SGT recorded tier1 hard held-out 0.8947/0.6434 small.
Phase C.0 diagnostic at 05:30 SGT recorded 0.9049/0.7463 small under same env
vars + image. +0.10 small AP "for free" with no model change. Possible causes:
`cv/models/best.pt` replaced between runs (v8s-1024 or v11m weights briefly
substituted); `CV_TILE_MODE` default changed with tiled-inference patch;
pycocotools / PIL version diff. Worth `sha256sum cv/models/best.pt` against the
original tier1 checksum if revisiting.

## Pass A — hard held-out failure analysis (15-16 May)

Ran `eval_cv_http.py` on `cv-yolo-v2-tier1-best` (`CV_CONF=0.20`, `CV_IOU=0.60`,
`CV_IMGSZ=896`, `CV_AUGMENT=1`, `CV_HALF=1`) against the hard held-out 500-image
/ 3334-box test split:

```text
mAP50-95: 0.8947
mAP50:    0.9954
mAP75:    0.9792
small:    0.6434     ← THE bottleneck (-0.25 below total)
medium:   0.8506
large:    0.9128
```

Per-class loss sorted by `(1-AP) * box_count`: top losers (cargo ship, fighter
jet, helicopter, commercial aircraft, warship) are medium-frequency classes with
AP 0.86-0.88, not rare classes whose AP collapses. **Not a class-imbalance story.**

Confusion at IoU ≥ 0.5, conf ≥ 0.20:

```text
ground-truth boxes: 3334
matched:            3333 (class-correct 3317, class-wrong 16, FN 1)
false-positives:    176
```

**16/3334 = 0.48% class confusion.** Aircraft subclass confusion dead as a
hypothesis. The 176 FPs are almost all in conf < 0.80 (see below).

FP/TP by score bucket (IoU ≥ 0.5):

```text
score        tp       fp    precision
0.20-0.30     1       35     2.8%
0.30-0.40     4       22    15.4%
0.40-0.50     1       28     3.4%
0.50-0.60     3       20    13.0%
0.60-0.70     6       23    20.7%
0.70-0.80    18       19    48.7%
0.80-0.90   380       19    95.2%
0.90-1.00  2917       13    99.6%
```

97% of TPs are conf ≥ 0.80. Low-conf buckets are mostly FPs.

### CV_CONF sweep — Lever 1 (free FP cleanup) is dead

Hypothesis: raising `CV_CONF` should drop most FPs at low cost in TPs and lift
mAP. **Falsified** — mAP drops monotonically:

```text
CV_CONF=0.20  mAP50-95=0.8947   ← current shipped
CV_CONF=0.40  mAP50-95=0.8929   -0.0018
CV_CONF=0.60  mAP50-95=0.8909   -0.0038
CV_CONF=0.70  mAP50-95=0.8886   -0.0061
CV_CONF=0.80  mAP50-95=0.8854   -0.0093
```

pycocotools mAP integrates the PR curve at each IoU. Cutting low-conf
detections amputates the high-recall tail. Even precision-2.8% predictions
contribute via the recall axis. Hidden cloud evaluator probably consumes our
`score` field and/or also runs an integrated PR curve.

### Worst-image eyeballing

`1992.jpg`, `3919.jpg`, `4853.jpg` (worst 3 by per-image loss): photo-composited
backgrounds (mountain valley, urban riverside, forest/lake) with cutout/3D
objects pasted at wildly varying scales — some aircraft 25-50 px wide on
1920×1080 native. **Small-AP gap is genuinely about pixel-scale localization at
high IoU**, not memorization. At imgsz=896, a 30px-native object becomes ~14px
in the model's input — below the resolution at which YOLOv8s anchors and stride
can localize tightly enough to clear IoU ≥ 0.75.

## Tiled inference A/B — Lever 2 is dead (16 May)

Implemented in `cv/src/cv_manager.py` behind `CV_TILE_MODE` (default `off`).
Modes `2x2`, `2x1`, `3x2` crop the 1920×1080 image into overlapping tiles
(`CV_TILE_OVERLAP=0.20` default), run detector at `CV_TILE_IMGSZ=768` per tile,
optionally also full image at `CV_IMGSZ=896` (`CV_TILE_FULL_PASS=1` default),
drop boxes touching internal tile edges (`CV_TILE_EDGE_MARGIN=4`), merge with
class-aware NMS at `CV_TILE_MERGE_IOU=0.50`.

Off-mode hard held-out is bit-identical to prior tier1 number (0.8947) — sanity
confirmed.

| Mode | Total mAP | Small | Medium | Large | Tile passes |
|---|---:|---:|---:|---:|---:|
| `off`  | 0.8947 | 0.6434 | 0.8506 | 0.9128 | 1 |
| `2x2`  | 0.8993 | 0.6139 | 0.8475 | 0.9185 | 5 |
| `2x1`  | 0.8771 | 0.5929 | 0.8329 | 0.8925 | 3 |
| `3x2`  | 0.9009 | 0.6255 | **0.8728** | 0.9180 | 7 |
| `3x2` em=0 ov=0.30 | similar pattern (no small-AP recovery) | — | — | — | 7 |

Read:

- **Small AP regressed in every tiled mode.** Hypothesis (tiling rescues small-object recall) was wrong on this dataset. `edge-margin=0` test showed it's not the 4px filter dropping legitimate small detections; it's how the model handles partial objects within tile crops.
- **3x2 lift is real but medium-object lift.** medium AP +0.022 (0.8506 → 0.8728) explains nearly the entire +0.006 total. 3x2 tiles are 738×600, so a 60px native object becomes ~80px effective at imgsz=768 — exactly the medium bucket.

Speed math (was conservative; cloud GPU faster than estimated):

```text
Current shipped : 0.75 * 0.556 + 0.25 * 0.956 = 0.656
3x2 best case   : 0.75 * 0.580 + 0.25 * 0.870 = 0.653  (+0.024 cloud accuracy assumed)
3x2 likely case : 0.75 * 0.560 + 0.25 * 0.870 = 0.638  (cloud accuracy flat)
```

**Decision**: don't ship tiled inference on tier1 weights. Patch stays merged
but disabled by default. Retested all four tile modes against v8s-1024 weights
when retrain finished — no candidate improved on tier1. Lever 2 dead.

## v8s-1024 retrain (16 May, NOT submitted)

3.8 hr on T4 via `training/cv/train_v4.sh`:

- `model=yolov8s.pt`, `imgsz=1024` (vs tier1's 768), `epochs=80`, `batch=10`
- `copy_paste=0.40` (vs tier1's 0.10), everything else matched tier1

Final Ultralytics val (500-image split): `mAP50-95 0.879` (vs tier1's 0.920 on
same split — real val regression of -0.041).

Hard held-out across three inference modes:

| Inference mode | Total mAP | Small | Medium | Large |
|---|---:|---:|---:|---:|
| `imgsz=1024 aug=1` | 0.8217 | 0.5327 | 0.7591 | 0.8425 |
| `imgsz=1280 aug=0` | 0.8370 | 0.5168 | 0.7594 | 0.8550 |
| `imgsz=1024 + tile=3x2` | 0.8361 | 0.5058 | 0.7953 | 0.8507 |

vs **tier1 baseline** (0.8947/0.6434/0.8506/0.9128): total regressed -0.058 to
-0.073; **small AP regressed -0.110 to -0.138 — opposite of hypothesis** (the
retrain was supposed to lift small AP via higher resolution; it crashed it).

Most likely cause: `copy_paste=0.40` was too aggressive on this dataset, which
is *already* a carefully composed copy-paste. Aggressive training-time
copy-paste slaps random instance crops onto random training images, shifting
training distribution away from eval. **Key lesson for any future v8s retrain:**

```text
copy_paste     dataset behavior
0.0 - 0.10     safe; tier1's recipe
0.20 - 0.30    untested; v3 used 0.30 (also regressed, confounded with v11m + matched-imgsz)
0.40           toxic; this run
```

Not submitted.

## v11m@1280 submission analysis

Submitted as `v11m-1280-noaug-v1` 16/05 04:04. Sweep top results:

```text
       conf=0.001  conf=0.05  conf=0.10  conf=0.20
iou=0.50  0.9080     0.9075     0.9075     0.9075
iou=0.70  0.9088     0.9083     0.9083     0.9083  ← best row
```

Per-area on best row (`conf=0.001 iou=0.70 imgsz=1280 aug=0`):

```text
small:  0.7168
medium: 0.8594
large:  0.9273
```

Best-row local→cloud:

| Metric | Local hard | Cloud |
|---|---:|---:|
| total mAP | 0.9088 | 0.474 |
| speed | (proj 0.85-0.92) | 0.949 |

Cloud observations:

- **Speed beat projection.** Projected 0.85-0.92, cloud delivered 0.949. Cloud GPU faster than workbench T4; no-TTA at 1280 doesn't bottleneck.
- **Accuracy gap structurally wider than v8s.** v11m@1280 gap = 0.435 vs v11m@1024 gap = 0.491 vs v8s tier1 gap = 0.349. v11m's gap is wider regardless of resolution.

`til test cv v11m-1280-noaug-v1` printed local full-set:

```text
mAP@.5:.05:.95: 0.812
small AP: 0.144   ← collapsed on easy-but-numerous full-set images
medium AP: 0.701
large AP: 0.878
```

Different signal from hard held-out: full set is dominated by easy images where
v11m@1280 over-fit (small AP collapsed to 0.144). Hard held-out is small-object
heavy → v11m@1280 looks great. Full set looks worse because the model is
mis-calibrated on common easy cases. **The hard split was the better proxy for
cloud direction; the full set was the better proxy for cloud absolute level.**

## v11m@1280 sweep (15 May, before v11m-1280-noaug-v1 submission)

aug=1 sweep at imgsz=1280 after v3-pre submission:

```text
mAP=0.9141  conf=0.001 iou=0.70 imgsz=1280 aug=1   ← best
mAP=0.9140  conf=0.05  iou=0.70 imgsz=1280 aug=1
mAP=0.9140  conf=0.10  iou=0.70 imgsz=1280 aug=1
mAP=0.9136  conf=0.001 iou=0.50 imgsz=1280 aug=1
(all imgsz=1024 rows: 0.8673 max — confirmed dead)
```

Hard held-out 0.9141 at imgsz=1280 (+0.047 vs 1024, +0.009 vs tier1's 0.9049).
Small AP recovered to 0.742 (vs tier1's 0.746 — essentially matched).

Speed problem with aug=1: ~1.04s/img → 520s for 500 → speed score ~0.71.
Blended: `0.75×0.60 + 0.25×0.71 = 0.628` vs tier1's `0.656`. TTA killed
blended → switched to aug=0 sweep (results in section above), which became
the `v11m-1280-noaug-v1` submission.

## Submission history

```text
Tag                    Submitted       Score   Speed   Errors    Notes
latest                 12/05 03:52     0.000   0.981   4 / 500   Empty-detection baseline, 4 inputs erroring
yolo-til-map-v2        14/05 01:56     0.044   0.961   0 / 500   YOLOv8n + sparse COCO→TIL map; clean serving, weak domain fit
cv-yolo-ft-v1          14/05 03:53     0.402   0.963   0 / 500   YOLOv8s fine-tuned on 18 TIL labels; local mAP50-95 0.885
cv-yolo-v2-best        14/05 14:00     0.549   0.960   0 / 500   YOLOv8s 768 hard-split retrain + tuned inference; til test 0.8839, hard 0.8589
cv-yolo-v2-tier1-best  14/05 17:10     0.556   0.956   0 / 500   v2-best weights + TTA + imgsz=896 + iou=0.60 + score field; hard 0.9049, til test 0.8505. NEW HIGH (+0.007); STILL ON LEADERBOARD.
cv-yolo11m-v3-pre      15/05 11:34     0.376   0.955   0 / 500   YOLOv11m@1024 fully trained 120ep; val 0.937, hard 0.8673. REGRESSED -0.180; matched-imgsz lost to v8s+upscaled.
v11m-1280-noaug-v1     16/05 04:04     0.474   0.949   0 / 500   Same v11m weights, imgsz=1280 aug=0. Hard 0.9088. REGRESSED -0.082; v11m gap structurally wider.
cv-augc1-v4            16/05 15:28     0.553   0.962   0 / 500   Phase C.1, mismatched Dockerfile config. Tied tier1 by luck.
cv-augc1-v4-1280       16/05 18:28     0.553   0.959   0 / 500   Phase C.1, matched config. Hard 0.948. Tier1 stays.
yolo11l-896-plusval-v1 22/05 23:45     0.640   0.954   0 / 500   YOLO11l on plusval data. Served at imgsz=896. Accuracy high.
yolo11l-1280-alldata-final-v2 23/05 20:55 [pending] [pending] 0 / 500 Fine-tuned YOLO11l on all data natively at 1280px. Local mAP: 0.988.
```

## Detailed timeline (early submissions)

### cv-yolo-v2-best (14 May 14:00) — hard split + 768 inference tuning

- Created non-leaky hard splits from `/home/jupyter/novice/cv/annotations.json`: 4000 train / 500 val / 500 test. Hard test split had 3334 boxes, denser than train and intentionally useful for stress eval.
- Trained `yolov8s.pt` 80 epochs at imgsz=768, batch=12, device=0, with stronger aug (close_mosaic=10, mixup=0.10, degrees=5, scale=0.60). 2.329 hr on T4.
- Ultralytics val on 500-image val split: P 0.984, R 0.961, mAP50 0.985, mAP50-95 0.920.
- Default-container hard held-out HTTP eval (`CV_CONF=0.25 CV_IOU=0.70 CV_IMGSZ=640`): 0.8043 mAP50-95.
- HTTP sweep on 100 hard-test images: imgsz=768 was the main win. Best smoke setting `CV_CONF=0.25 CV_IOU=0.50 CV_IMGSZ=768` scored 0.8652 on the 100-image slice.
- Full hard held-out HTTP confirmation at those settings: 0.8589 mAP50-95, 0.9414 mAP50, small AP 0.5596, medium 0.7557, large 0.8911.
- Full local `til test cv cv-yolo-v2-best`: 0.8839 mAP@.5:.05:.95, 0 errors, 1250/1250 batches in 13:59.
- Official: **0.549 / 0.960**, 0/500 errors. +0.147 official accuracy vs `cv-yolo-ft-v1`.

Interpretation: non-leaky held-out eval was directionally useful even though hidden gap is still large (0.8589 → 0.549). Biggest confirmed lever: matching inference imgsz=768 to training. Remaining weakness: hidden small-object and aircraft-subclass distribution shift, not output format.

### cv-yolo-ft-v1 (14 May 03:53) — trained 18-class YOLOv8s

- Converted `/home/jupyter/novice/cv/annotations.json` into Ultralytics YOLO dataset: 4500 train / 500 val images, 16620 train / 1881 val boxes.
- Trained `yolov8s.pt` 60 epochs at imgsz=640, batch=16. 1.297 hr on T4.
- Ultralytics val: P 0.964, R 0.936, mAP50 0.975, mAP50-95 0.905.
- `til test` full local novice CV set: mAP@.5:.05:.95 0.885, mAP50 0.951, 0 errors, 1250 batches in 08:30.
- Official: **0.402 / 0.963**, 0/500 errors.

Interpretation: not an LTWH/output-format failure. A bbox-format bug would score near 0.0 or cause result-loading errors. Score improved 0.044 → 0.402 with unchanged speed and no errors, proving schema and `xyxy → LTWH` adapter work. Local→official gap (0.885 → 0.402) is hidden distribution shift / harder images / small objects / label-domain differences, not plumbing.

### yolo-baseline / yolo-til-map-v2 (14 May) — pretrained detector + robust fallback

Implemented plumbing:

- Wrapped image decode + model inference so bad images return `[]` instead of erroring. `cv_server.py` passes evaluator `key` to manager for debug logs.
- Added `PIL.ImageOps.exif_transpose(...).convert("RGB")` so grayscale/RGBA/EXIF oddities normalize before inference.
- Added Ultralytics `yolov8n.pt` loaded once in `CVManager.__init__`.
- Converts xyxy → official LTWH `[l,t,w,h]`, clamps boxes to image bounds, drops zero-area boxes, emits plain Python `float`/`int`.
- Added default YOLO-index to TIL category-id mapping. Set `CV_CATEGORY_MAP` to a JSON dict (`{"4": 3}`) or path to JSON file when tuning mappings.

Official TIL category IDs (confirmed on GCP):

```text
0 cargo aircraft, 1 commercial aircraft, 2 drone, 3 fighter jet,
4 fighter plane, 5 helicopter, 6 light aircraft, 7 missile, 8 truck,
9 car, 10 tank, 11 bus, 12 van, 13 cargo ship, 14 yacht,
15 cruise ship, 16 warship, 17 sailboat
```

Default sparse mapping:

```text
YOLO car      -> 9 car
YOLO airplane -> 1 commercial aircraft
YOLO bus      -> 11 bus
YOLO truck    -> 8 truck
YOLO boat     -> 13 cargo ship
```

Only a baseline — COCO YOLO cannot distinguish drone / fighter / cargo / light aircraft, missile / tank / van, or specific ship subclasses without fine-tuning.

### latest (12 May 03:52) — submission plumbing only

Sanity-check: service starts, endpoint accepts requests, JSON shape correct. No detection logic. Scored 0.000 as expected.

## Tier 1 sweep — picking the 0.9049 row

After v2-best (0.549 cloud), env-only changes to find tier1 (same weights):

- `CV_AUGMENT=1` (Ultralytics flip+multi-scale TTA) — wins on small/dense scenes
- `CV_HALF=1` (FP16 inference) — speed-neutral
- `CV_IMGSZ=896` — best resolution at imgsz vs accuracy plateau
- `CV_IOU=0.60`, `CV_CONF=0.20` — picked from sweep
- `score` field emitted per detection (Ultralytics conf, not pinned 1.0)

Sweep top rows (38 of 144 before terminating — trend was clear):

```text
mAP=0.9049  conf=0.20 iou=0.60 imgsz=896 aug=1   ← shipped
mAP=0.9044  conf=0.20 iou=0.55 imgsz=896 aug=1
mAP=0.9042  conf=0.20 iou=0.45 imgsz=896 aug=1
mAP=0.9041  conf=0.25 iou=0.45 imgsz=896 aug=1
mAP=0.9016  conf=0.20 iou=0.45 imgsz=896 aug=0
mAP=0.8727  conf=0.20 iou=0.45 imgsz=1024 aug=1  ← imgsz=1024 dead with 768-trained weights
mAP=0.8355  conf=0.20 iou=0.45 imgsz=1024 aug=0
```

Confirmed: `imgsz=896 + aug=1` is the regime; iou plateau 0.45-0.60 (Δ<0.001);
conf=0.20 slightly beats 0.25. Local→official gap 0.9049 → 0.556 = 0.349.

## v11m@1024 (Tier 2, May 15) — full training history

- Trainer: `training/cv/train_v3.sh`. YOLOv11m @ imgsz=1024, batch=6, AdamW cos_lr, copy_paste=0.30 + mosaic=1.0 + mixup=0.15.
- Wallclock estimate revised: original 4-5h was wrong. YOLOv11m@1024 runs ~6:30/epoch on T4 → 120 epochs ≈ 13h. v8s@768 was 80 epochs in 2.3h. Compound: m vs s + 768 vs 1024 + 80 vs 120 = ~5.6x.
- First run died at epoch 58/120 from CUDA OOM caused by docker squatter containers from earlier sweep eating ~10GB VRAM. `best.pt` saved (epoch 56, val mAP50-95 0.895).
- Resumed from `last.pt`, ran to completion at 120 epochs (~7h after resume). Final Ultralytics val mAP50-95 **0.937**, mAP50 **0.994** on 500-image val split. Per-class low: `warship` 0.862; high: `car` 0.988.
- Sweep on fully-trained `best.pt` against hard test split: max mAP50-95 0.8673 at `conf=0.001 iou=0.70 imgsz=1024 aug=1`. **0.038 below v8s tier1's 0.9049.** Small-object AP: v11m 0.587 vs tier1 0.746.
- Submitted `cv-yolo11m-v3-pre` (15/05 11:34): **0.376 / 0.955**, 0/500 errors. **REGRESSED -0.180 vs tier1 0.556.**

Post-mortem:

1. **Resolution mismatch.** v8s tier1 trained at 768 / inferenced at 896 (UP); upscaling helped small objects (+0.04 in v8s sweep). v11m trained at 1024 / inferenced at 1024 (matched). v11m@1280 (later sweep) lifted hard to 0.9141 but didn't transfer.
2. **Small-object hit from high-IoU bins.** v11m mAP50 = 0.997 (essentially perfect detection); loss is in high-IoU precision bins (≥0.75) — exactly what higher inference resolution helps.
3. **Bigger model + matched-imgsz < smaller model + upscaled-imgsz** on this dataset. Counter-intuitive but reproducible.
4. **Local→official gap structurally wider** (0.491 vs tier1's 0.349). Hidden eval harsher on bbox precision than hard held-out suggests.

## What we run

Schema-safe Ultralytics YOLO service in [src/cv_manager.py](src/cv_manager.py),
packaged by [Dockerfile](Dockerfile):

```python
def cv(image_bytes: bytes, key=None) -> list[dict]:
    # Decode robustly, run YOLO, convert xyxy -> official LTWH.
    # Any decode/model failure returns [] and logs key/byte size.
```

Docker expects the trained checkpoint at `/workspace/models/cv/best.pt`, copied
from local `cv/models/best.pt` during build. `CV_CATEGORY_MAP` is identity 0..17
(fine-tuned model was trained on the official TIL label order). Shipped
inference defaults baked into the image vary by tag (see tier1 sweep section).

## `test_cv.py` pins `score: 1.0` — gotcha

`test/test_cv.py` (the local evaluator that powers `til test`) hardcodes
`score=1.0` on every box before running pycocotools. Standard COCO mAP
integrates PR over confidences; pinning all scores to 1.0 turns mAP into a
yes/no precision metric where every FP at score 1.0 directly hurts you.

This is why the optimal `CV_CONF` is **higher** than COCO-default 0.001: lower
conf adds FPs as score-1.0 ties and tanks precision. We can't unilaterally
change the evaluator (it's competition scaffolding); best we can do is sweep
conf up in 0.20-0.60 and pick F1 optimum on hard held-out. Whether cloud
evaluator does the same or consumes our `score` field is not knowable without
submitting, but env-driven design lets us pivot either way.

## What CV needs to hit 0.7 (left for later if revisited)

None have high confidence under our timeline:

1. **RT-DETR-L or YOLOv9c transfer.** Different backbone family with unknown local→cloud gap. ~6h GPU per train. Could land 0.45 or 0.65 — high variance, info-positive.
2. **Conservative re-augmentation.** Drop tile crops, keep only JPEG. Recovers full-local performance. Realistic cloud 0.55-0.58. Doesn't hit 0.7.
3. **Wholesale distribution-shift recipe** (synthetic photo-composition, harder mosaic, train at native-resolution multi-scale). Half day of dataloader work + 6-8h training. Plausible but not credible at deadline.

Default: **stay parked at tier1.** Submission slots are uncapped so any of the
above can be tried in parallel without risk to leaderboard — but should be
re-evaluated against AE/NLP marginal-hour ROI before kicking off.

## Output-format checks to preserve

- YOLO training labels are normalized center-XYWH; that is only the training format.
- API output must stay pixel LTWH `[left, top, width, height]`.
- `cv_manager.py` converts Ultralytics xyxy → LTWH and clamps to image bounds. Keep this adapter unchanged unless local `til test` catches a regression.
- `cv_manager.py` emits a `score` field per detection (Ultralytics conf). Official `cv/README.md` schema doesn't list `score`, but `test/test_cv.py` silently appends `score=1.0` regardless. Including real confidence is free upside if cloud evaluator consumes it, no-op otherwise.

## Reproducibility / pointers

- Manager source: [src/cv_manager.py](src/cv_manager.py)
- HTTP server (don't edit): [src/cv_server.py](src/cv_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#cv](../SUMMARY.md)
- Training scripts: [../training/cv/](../training/cv/) — `prepare_yolo_dataset.py`, `build_aug_dataset.py`, `train_v3.sh`, `train_v4.sh`, `eval_cv_http.py`, `sweep_cv_http.py`, `analyze_cv_failures.py`, `gap_diagnose.py`
