# CV — notes & history

Last updated: 16 May 2026 06:00 SGT — **CV UN-PARKED. Phase C.1 ready to
train.** Diagnostic ran on tier1 + hard held-out; verdict is that
**both JPEG quality and resolution shifts hit small AP hard** (-0.13 to
-0.19 small AP under various transforms) even though the total mAP
drops are modest (-0.01 to -0.04). Since the cloud failure mode is
small-AP per Pass A, train-time augmentation along these axes is
hypothesis-aligned. `training/cv/build_aug_dataset.py` and
`training/cv/train_v4.sh` written; estimated cloud lift +0.03 to +0.07
(target cloud 0.58-0.63). 0.7 still requires more than Phase C.

Per-task working log for CV (object detection). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#cv).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Phase C — gap-shrink plan (16 May 04:50 SGT)

**Premise.** All five inside-the-box recipe levers are dead (Pass A confusion,
`CV_CONF` sweep, tiled inference, v8s-1024 cp=0.40, v11m@1280). Calibration
says v8s needs hard ≥ 1.05 and v11m needs ≥ 1.14 to reach cloud 0.70 at their
current gaps — both impossible. So **the gap itself has to shrink**.

The gap was *not* tested against:

- **JPEG quality shift** — hidden eval images may be lower-quality JPEGs than
  our near-lossless photo-composited training set.
- **Scale / resolution shift** — hidden images may be at a different native
  resolution, so the model's effective small-object pixel sizes differ.

**Phase C.0 — diagnostic** (`training/cv/gap_diagnose.py`).
Re-encode the hard held-out 500 images in-memory under each transform below
and re-score tier1. Reuses `score_predictions` from `eval_cv_http.py` so
numbers compare directly to the existing `0.8947` baseline.

| Transform | Tests |
|---|---|
| `baseline` | sanity, must reproduce 0.8947 |
| `jpeg-q70`, `jpeg-q50`, `jpeg-q30` | JPEG-quality shift |
| `downsample-2x`, `downsample-3x` | effective-resolution shift |
| `jpeg50-down2` | upper-bound on stacked effect |

Verdict logic:
- transform drops mAP ≥ 0.05 → matching train-time augmentation is high-EV
- transform barely moves mAP → that hypothesis is dead before training

**Phase C.1** — single augmentation justified by C.0. Retrain v8s @ imgsz=1024
on tier1's recipe + one new aug. ~4h GPU, ~30 LOC. Submit, observe gap delta.

**Phase C.2** — stack the second augmentation only if C.1 moved cloud.
Otherwise we've saved ~6h and skip to alternative hypotheses (eg. annotation
noise check, RT-DETR family transfer).

**Phase C.3** — HSV strength bump folds into C.2's recipe as a 1-line train
arg, not a separate phase.

**Honest target.** Realistic ceiling if both axes are real and both augs
stack favorably: cloud 0.60-0.65. Cloud 0.70 requires the gap to shrink from
0.35 → 0.20 (a 43% reduction) AND hard held-out to stay at 0.91 — that's the
upside path, not the median expectation.

**Submission economy.** Submission slots are uncapped and only the best
score counts on the leaderboard; ship and observe, don't pre-gate on local
proxies that have already proven unreliable.

How to run the diagnostic on Workbench:

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

Runtime: ~35 min for all 7 transforms; `--transforms baseline,jpeg-q50,downsample-2x`
for a ~15-min smoke.

### Diagnostic results (16 May 05:30 SGT)

Ran `gap_diagnose.py` on tier1 + the 500-image hard held-out. Baseline came
out at **mAP 0.9049 / small AP 0.7463** — note this is +0.01 mAP and +0.10
small AP higher than Pass A's earlier 0.8947/0.6434 baseline. Same image,
same env vars in theory; possibly different `cv/models/best.pt` checksum
or different `CV_TILE_MODE` default. Deltas below are valid (all measured
against the same baseline in this run); the absolute drift is flagged for
follow-up.

| Transform | Total mAP | Small AP | Medium AP | Large AP | Δ total | Δ small |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9049 | 0.7463 | 0.8245 | 0.9274 | — | — |
| jpeg-q70 | 0.8934 | 0.6168 | 0.8148 | 0.9194 | −0.011 | **−0.130** |
| jpeg-q50 | 0.8900 | 0.5972 | 0.8060 | 0.9156 | −0.015 | **−0.149** |
| jpeg-q30 | 0.8786 | 0.6495 | 0.7840 | 0.9027 | −0.026 | −0.097 |
| downsample-2x | 0.8928 | 0.6287 | 0.7783 | 0.9212 | −0.012 | **−0.118** |
| downsample-3x | 0.8612 | 0.6428 | 0.7182 | 0.8987 | −0.044 | −0.104 |
| jpeg50-down2 | 0.8776 | 0.5598 | 0.7418 | 0.9072 | −0.027 | **−0.187** |

The script's auto-verdict ("JPEG aug dead, resolution dead") was misleading
because it thresholded on total mAP at 0.05. The real signal is **small AP
drops by 10-19 points under every shift**, and Pass A established that
small AP is precisely where the cloud loses. So both axes are real
contributors to the cloud gap, just not visible in the total-mAP rollup.

**Honest interpretation:**
- The total-mAP drops sum to at most -0.07 across all axes; cloud gap is 0.349.
  So JPEG + resolution shifts account for **≤ 20%** of the cloud gap.
- The remaining 80% is most likely **different scene content** (different
  photos with a heavier small-object bias), not a transform of similar ones.
- Train-time augmentation will help, but the realistic cloud ceiling for
  Phase C alone is **0.58-0.63**, not 0.70. To hit 0.70 we also need a
  backbone change (RT-DETR / YOLOv9) or a wholesale distribution-shift recipe.

## Phase C.1 — augmented v8s @ 1024 retrain (16 May 06:00 SGT)

Single integrated train, not a phased C.1 → C.2 dance. Justification:
both diagnostic axes hit small AP at similar magnitudes; building one
augmented dataset that covers both is cheaper than two sequential runs.

**Pipeline:**

1. **Offline dataset expansion** — `training/cv/build_aug_dataset.py`.
   For each train image (~4000):
   - Keep the original.
   - Add a JPEG-recompressed copy at random quality in [40, 85].
   - Add one native-resolution 1024×1024 crop (random position biased toward
     a box center; ≥ 0.5 visible-area threshold; drop crops with no visible
     boxes).
   Val/test untouched. Output `/home/jupyter/cv_yolo_dataset_augc1/` (~3x train).
2. **Retrain** — `training/cv/train_v4.sh`. Changes vs tier1:
   - `imgsz=1024` (was 768) — high-res input
   - `scale=0.80` (was 0.60) — more aggressive small-object generation
   - Everything else matched to tier1: `yolov8s.pt`, 80 epochs, batch=8,
     AdamW + cos_lr, `mosaic=1.0`, `mixup=0.10`, `copy_paste=0.10`
     (the 16/05 v8s-1024 retrain confirmed cp=0.40 is toxic).

**Run on Workbench:**

```bash
# 1. Build the augmented dataset (~5-10 min)
python training/cv/build_aug_dataset.py \
  --in-dir /home/jupyter/cv_yolo_dataset \
  --out-dir /home/jupyter/cv_yolo_dataset_augc1 \
  --crops-per-image 1 \
  --jpeg-per-image 1

# 2. Train (~6h on T4)
bash training/cv/train_v4.sh

# 3. Deploy
cp /home/jupyter/cv_runs/til-yolov8s-1024-augc1-v4/weights/best.pt cv/models/best.pt
til build cv cv-augc1-v4

# 4. Sweep + submit best blended row
python training/cv/sweep_cv_http.py \
  --image melanie-minions-cv:cv-augc1-v4 \
  --data-dir /home/jupyter/novice/cv \
  --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \
  --out-dir /home/jupyter/cv_eval_sweeps/augc1-v4 \
  --conf 0.001,0.05,0.20 \
  --iou 0.50,0.60,0.70 \
  --imgsz 1024,1280 \
  --augment 0,1
til submit cv cv-augc1-v4
```

**Decision rule after submit:**
- Cloud > 0.556 → new tier1; iterate (try imgsz=1280 inference, more aug, etc.)
- Cloud ≈ 0.556 (within ±0.01) → Phase C ceiling, pivot to RT-DETR / YOLOv9
- Cloud < 0.546 → Phase C scope wrong; fall back, rethink.

Submission slots are uncapped per the rules (leaderboard keeps the higher
score), so submit and observe — don't pre-gate on local proxies that have
already proven unreliable.

### Baseline anomaly to verify (non-blocking)

Pass A on 16 May 02:30 SGT recorded tier1 hard held-out at 0.8947/0.6434
small. The 16 May 05:30 SGT diagnostic recorded 0.9049/0.7463 small under
the same env vars and same image. That's +0.10 small AP "for free" with no
model change. Possible explanations:

- `cv/models/best.pt` was replaced between the two runs (v8s-1024 or v11m
  weights briefly substituted in for tier1)
- `CV_TILE_MODE` got a default change with the tiled-inference patch
- pycocotools or PIL version diff changed evaluation/encoding behavior

If the +0.10 small AP is a real model/code state, it'd compound any cloud
lift from C.1. Worth confirming with `sha256sum cv/models/best.pt` against
the original tier1 checksum and a quick `git log -p cv/src/cv_manager.py`
since 14 May 17:10 SGT.

## Current shipped tag

**`cv-yolo-v2-tier1-best` — official 0.556 / 0.956 (14 May 17:10 SGT, 0 of 500 errors). STILL ON LEADERBOARD.**

Three tier1-killer attempts on 16 May all regressed cloud or were
blended-flat:

- `cv-yolo-v2-tier1-best` (live): cloud `0.556 / 0.956`, hard held-out `0.9049`
- `cv-yolo11m-v3-pre` (15/05): cloud `0.376 / 0.955`, hard held-out `0.8673`
- `v11m-1280-noaug-v1` (16/05): cloud `0.474 / 0.949`, hard held-out `0.9088`
- v8s-1024 retrain (16/05, NOT submitted): hard held-out `0.8217-0.8370`,
  was a clear local regression so tier1's leaderboard slot was protected.

Final read: **CV is parked**. We've now characterised the local→cloud gap
across model families and across resolution/TTA settings, and there is no
inference-only or training-only lever left that we haven't already tried.
Continuing CV would mean either a fundamentally different architecture
(SAHI-style proper tiling at training time, or a different detection
family) or class-balanced retraining with a cleaner augmentation recipe;
neither has a high-confidence path past tier1's `0.556` on cloud.

## Day-of summary (16 May)

What we ran, in order, and what each thing told us:

1. **Pass A failure analysis on tier1 weights** (hard held-out `0.9049`).
   Result: small AP `0.6434` is the entire mAP gap; class confusion is
   `16/3334 = 0.48%` — confirmed dead as hypothesis. Worst-image eyeballing
   of `1992.jpg`, `3919.jpg`, `4853.jpg` showed the dataset is photo-
   composited with cutout/3D objects pasted at wildly varying scales,
   small targets land at 25-50 px on 1920×1080 native (`~14 px` in the
   model's `896` input).
2. **`CV_CONF` sweep** (`0.20 → 0.40 → 0.60 → 0.70 → 0.80`). Result: mAP
   regressed monotonically (`0.8947 → 0.8854`). Cloud is consuming
   detections through the integrated PR curve — low-conf predictions
   contribute via the recall axis. **Lever 1 (free FP cleanup) is dead.**
3. **Tiled-inference patch + A/B** (`CV_TILE_MODE` env in `cv_manager.py`,
   modes `2x2 / 2x1 / 3x2`). Result: `3x2` lifts hard held-out by `+0.006`,
   driven entirely by medium AP (`0.8506 → 0.8728`); small AP *regressed*
   in every tiled mode. With 3× compute cost, blended tied or regressed.
   **Lever 2 (tiled inference) is dead** for tier1 weights.
4. **`v8s-1024` retrain** (3.8 hr on T4; `imgsz=1024`, `copy_paste=0.40`).
   Final Ultralytics val `mAP50-95 = 0.879` (vs tier1's `0.920` on the
   same 500-image val split). Hard held-out across 3 inference modes:
   `0.8217 / 0.8370 / 0.8361` — all clearly worse than tier1's `0.9049`.
   The aggressive `copy_paste=0.40` was likely the regressor (the dataset
   is *already* a careful copy-paste; aggressive training-time copy-paste
   slaps random instance crops onto random scenes, shifting the
   distribution away from eval). **Not submitted.** Lesson: `copy_paste`
   is destructively powerful on this dataset; v3's `0.30` may also have
   been a contributor to v3's regression.
5. **v11m@1280 aug=0 sweep + submission** (`v11m-1280-noaug-v1`).
   Hard held-out `0.9088` (`+0.014` over tier1). Submitted because the
   no-TTA speed math projected `~0.85-0.92` cloud speed and the local
   accuracy lift was real. **Cloud landed `0.474 / 0.949`** (`-0.082`
   accuracy vs tier1, `-0.007` speed). Tier1 stays on leaderboard via
   highest-score retention.

## v11m@1280 aug=0 submission analysis

```text
sweep results, conf x iou
       conf=0.001  conf=0.05  conf=0.10  conf=0.20
iou=0.50  0.9080     0.9075     0.9075     0.9075
iou=0.70  0.9088     0.9083     0.9083     0.9083  ← best row
```

Per-area on the best row (`conf=0.001 iou=0.70 imgsz=1280 aug=0`):

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

Cloud-side observations:

- **Speed beat my projection by a comfortable margin.** I projected
  `0.85-0.92` based on T4 wall-clock; cloud delivered `0.949`. The cloud
  GPU is meaningfully faster than the workbench T4, and no-TTA at 1280
  doesn't bottleneck on it. **Cloud speed is no longer a CV constraint
  for any reasonable inference resolution.**
- **Accuracy gap was wider than v8s.** Local→cloud gap for v11m@1280
  is `0.9088 - 0.474 = 0.435`. For v11m@1024 it was `0.491`. For v8s
  tier1 it's `0.349`. **v11m's gap is structurally wider than v8s's**;
  resolution helps it locally but doesn't close the cloud gap.
- **The 0.349 v8s gap held only for v8s** — calibration insight.
  Going forward, predict cloud accuracy from local hard held-out via
  separate gap estimates per backbone family.

`til test cv v11m-1280-noaug-v1` printed local full-set:

```text
mAP@.5:.05:.95: 0.812
small AP: 0.144   ← collapsed on the easy-but-numerous full-set images
medium AP: 0.701
large AP: 0.878
```

This is a different signal from the hard held-out: the full local set
is dominated by easy images that v11m@1280 over-fit on (small AP
collapsed to `0.144`). The hard held-out happens to be small-object
heavy, so v11m at 1280 looks great there; the full set looks much
worse because the model is now mis-calibrated on common easy cases.
The hard split was the better proxy for cloud direction; the full
set was the better proxy for cloud absolute level.

## Calibration learned (for any future CV submission)

```text
Backbone     Local→cloud gap on hard held-out
v8s          ~0.35  (tier1: 0.9049 → 0.556)
v11m         ~0.44  (v3-pre: 0.8673 → 0.376; v11m@1280: 0.9088 → 0.474)
```

The two v11m data points are within 0.06 of each other in absolute
cloud terms (`0.376` vs `0.474`) and the gap is roughly stable
(`0.491` vs `0.435`). The gap is a function of the architecture's
generalization to the hidden distribution, not of resolution.

**Implication**: to beat tier1 cloud `0.556`, a v11m-class submission
would need hard held-out ≥ `0.99`, which is essentially out of reach.
A v8s-class submission would need hard held-out ≥ `0.91` (current
tier1 is `0.9049`). A different backbone family would have an unknown
gap; the only way to discover it is to submit, and we've used up
that exploratory budget.

## Pass A — hard held-out failure analysis (15-16 May)

Ran the existing `eval_cv_http.py` on `cv-yolo-v2-tier1-best` (`CV_CONF=0.20`,
`CV_IOU=0.60`, `CV_IMGSZ=896`, `CV_AUGMENT=1`, `CV_HALF=1`) against the hard
held-out 500-image / 3334-box test split:

```text
mAP50-95: 0.8947
mAP50:    0.9954
mAP75:    0.9792
small:    0.6434     ← THE bottleneck (-0.25 below total)
medium:   0.8506
large:    0.9128
```

Per-class loss table sorted by `(1-AP) * box_count` showed the top losers
(cargo ship, fighter jet, helicopter, commercial aircraft, warship) are all
medium-frequency classes whose AP sits at 0.86-0.88, not rare classes whose
AP collapses. Confirmed: it is **not** a class-imbalance story.

Confusion at IoU >= 0.5, conf >= 0.20:

```text
ground-truth boxes: 3334
matched:            3333 (class-correct 3317, class-wrong 16, FN 1)
false-positives:    176
```

**16 / 3334 = 0.48% class confusion.** Aircraft subclass confusion is dead as
a hypothesis. The 176 FPs are almost all in conf < 0.80 (see below).

FP/TP by score bucket (IoU >= 0.5):

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

97% of TPs are conf >= 0.80. The low-conf buckets are mostly FPs.

### CV_CONF sweep (cleaning up FPs by raising conf)

Hypothesis: raising `CV_CONF` should drop most FPs at low cost in TPs and
lift mAP. **Falsified** — mAP drops monotonically:

```text
CV_CONF=0.20  mAP50-95=0.8947   ← current shipped
CV_CONF=0.40  mAP50-95=0.8929   -0.0018
CV_CONF=0.60  mAP50-95=0.8909   -0.0038
CV_CONF=0.70  mAP50-95=0.8886   -0.0061
CV_CONF=0.80  mAP50-95=0.8854   -0.0093
```

Read: pycocotools mAP integrates the precision-recall curve at each IoU.
Cutting low-conf detections doesn't just remove FPs, it amputates the
high-recall tail of the PR curve. Even precision-2.8% predictions are
contributing to the AP integral via the recall axis. This is the *opposite*
of what `test/test_cv.py`'s `score=1.0` pinning would suggest, and means
the hidden cloud evaluator probably consumes our `score` field and/or
also runs an integrated PR curve.

**Lever 1 (raise conf for free FP cleanup) is dead** locally. Could still
flip on cloud if cloud uses score-pinned mAP, but EV is small and the local
signal is clean enough that we shouldn't burn a submission slot on it.

### Class-balanced sampling — also dead

With 16/3334 class-wrong boxes, there is no class-imbalance story to fix.

### What the 25 worst images showed

Eyeballing 1992.jpg, 3919.jpg, 4853.jpg (worst 3 by per-image loss):
photo-composited backgrounds (mountain valley, urban riverside, forest/lake)
with cutout/3D objects pasted at wildly varying scales — some aircraft 25-50
pixels wide on 1920×1080 native. **Backgrounds vary; the small-AP gap is
genuinely about pixel-scale localization at high IoU**, not memorization.
At inference imgsz=896, a 30px-native object becomes ~14px in the model's
input, which is below the resolution at which YOLOv8s' anchors and stride
can localize tightly enough to clear IoU >= 0.75.

## Tiled inference A/B (16 May)

Implemented in `cv/src/cv_manager.py` behind `CV_TILE_MODE` (default `off`).
Modes added: `2x2`, `2x1`, `3x2`. Each mode crops the 1920×1080 image into
overlapping tiles (default `CV_TILE_OVERLAP=0.20`), runs the detector at
`CV_TILE_IMGSZ=768` per tile, optionally also runs the full image at
`CV_IMGSZ=896` (`CV_TILE_FULL_PASS=1` default), drops boxes touching internal
tile edges (`CV_TILE_EDGE_MARGIN=4` default), and merges with class-aware NMS
at `CV_TILE_MERGE_IOU=0.50`.

Off-mode hard held-out is bit-identical to the prior tier1 number (`0.8947`)
— sanity confirmed before trusting tile variants.

A/B against the same hard held-out split, tier1 weights:

| mode | total mAP | small | medium | large | tile passes |
|---|---:|---:|---:|---:|---:|
| `off`  | 0.8947 | 0.6434 | 0.8506 | 0.9128 | 1 |
| `2x2`  | 0.8993 | 0.6139 | 0.8475 | 0.9185 | 4 + 1 = 5 |
| `2x1`  | 0.8771 | 0.5929 | 0.8329 | 0.8925 | 2 + 1 = 3 |
| `3x2`  | 0.9009 | 0.6255 | **0.8728** | 0.9180 | 6 + 1 = 7 |
| `3x2` em=0 ov=0.30 | similar pattern (no clear small-AP recovery) | — | — | — | 7 |

Read:

- **Small-AP regressed in every tiled mode**. The hypothesis (tiling rescues
  small-object recall) was wrong on this dataset. Likely the edge-margin
  filter at 4px drops legitimate small detections that happen to land on
  internal tile cuts; lowering to 0 didn't recover small AP, suggesting it's
  also a property of how the model handles partial objects within tile crops.
- **The 3x2 lift is real but it's a medium-object lift.** medium AP +0.022
  (0.8506 → 0.8728) explains nearly the entire +0.006 total. 3x2 tiles are
  738×600, so a 60px native object becomes ~80px effective at imgsz=768 —
  exactly the medium bucket.

### Speed cost (back-of-envelope, T4)

This estimate was wrong on the cloud side. The cloud GPU is faster than
the T4 by enough that no-TTA inference at 1280 hits cloud speed `0.949`,
not the `0.85-0.92` we projected. The blended math below was conservative
in the wrong direction. (See "Calibration learned" above.)

Current shipped (1 forward pass with TTA ≈ 3 fwd passes) hits cloud
speed `0.956`. 3x2 with full + TTA = 7 forward passes ≈ 3× compute on T4.

```text
Current shipped : 0.75 * 0.556 + 0.25 * 0.956 = 0.656
3x2 best case   : 0.75 * 0.580 + 0.25 * 0.870 = 0.653  (+0.024 cloud accuracy assumed)
3x2 likely case : 0.75 * 0.560 + 0.25 * 0.870 = 0.638  (cloud accuracy flat)
```

Tied at best, regressed at worst on T4 numbers; in retrospect speed would
have been higher on cloud, but `3x2` still wouldn't have shipped because
the local +0.006 mAP would not have translated to a cloud accuracy lift.

### Decision (was)

Don't ship tiled inference on tier1 weights. Patch stays merged but
disabled by default. Re-tested all four tile modes against the v8s-1024
weights when the retrain finished — no candidate improved on tier1.

**Lever 2 (tiled inference) is dead.** The diagnostic value was real:
it confirmed the small-object failure mode is at-IoU localization, not
detection-recall, and ruled out a cheap inference fix.

## v8s-1024 retrain (16 May, NOT submitted)

3.8 hr on T4 (`training/cv/train_v4.sh`):

- `model=yolov8s.pt`
- `imgsz=1024` (vs tier1's 768)
- `epochs=80`
- `batch=10`
- `copy_paste=0.40` (vs tier1's 0.10)
- everything else matched tier1

Final Ultralytics val (500-image split):

```text
all  Box(P 0.975  R 0.975  mAP50 0.989  mAP50-95 0.879)
```

Tier1 on the same val: `mAP50-95 0.920`. The retrain was already a real
val regression of `-0.041`.

Hard held-out across three inference modes (`/home/jupyter/cv_eval_sweeps/1024-debug/`):

| Inference mode | total mAP | small | medium | large |
|---|---:|---:|---:|---:|
| `imgsz=1024 aug=1` | 0.8217 | 0.5327 | 0.7591 | 0.8425 |
| `imgsz=1280 aug=0` | 0.8370 | 0.5168 | 0.7594 | 0.8550 |
| `imgsz=1024 + tile=3x2` | 0.8361 | 0.5058 | 0.7953 | 0.8507 |

vs **tier1 baseline** (`0.8947 / 0.6434 / 0.8506 / 0.9128`):

- Total mAP regressed `-0.058` to `-0.073`.
- Small AP regressed `-0.110` to `-0.138` — the *opposite* of the
  hypothesis. The retrain was supposed to lift small AP via higher
  resolution; it crashed it instead.

Most likely cause: `copy_paste=0.40` was too aggressive. The dataset is
*already* a carefully composed copy-paste. Aggressive training-time
copy-paste slaps random instance crops onto random training images,
shifting the training distribution away from the eval distribution.
This is a key lesson if anyone retrains v8s in the future:

```text
copy_paste     dataset behavior
0.0 - 0.10     safe; tier1's recipe
0.20 - 0.30    untested; v3 used 0.30 (also regressed, but with v11m
               + matched-imgsz, so confounded)
0.40           toxic; this run
```

Not submitted. CV is parked.

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
