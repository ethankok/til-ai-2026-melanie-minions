# CV — notes & history

> **How to read this file.** This is the CV working memory: what we've tried,
> what worked, what failed, and *why*, so we don't re-run dead ends. **Read the
> `## Read this first` digest below before doing any CV work** — it is the
> current, distilled state. Everything under the `## Detailed history (archive)`
> divider further down is the original per-session log, kept for raw numbers,
> sweep tables, and reproduction commands. The archive is intentionally
> redundant with the digest; consult it only for the specifics behind a claim.
> Authoritative spec: [README.md](README.md) +
> [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#cv).
> Cross-task scores: [../RESULTS.md](../RESULTS.md).
>
> _Digest last refreshed: 30 May 2026._

---

## Read this first

### Current state (30 May 2026)

- **Phase:** Qualifiers closed (window ended 24 May 23:59:59 SGT). Semifinals
  prep runs through **2026-06-10**. CV is 20% of the overall score.
- **Champion (current high by blended score):**
  **`yolo11l-1408` — 0.684 acc / 0.937 speed, blended 0.7473**, 0/500 errors
  (30 May). These are the same YOLO11l v1 weights (all-data, trained at
  `imgsz=1024`) served at **`CV_IMGSZ=1408`** — the upscale-at-inference lever
  had headroom past 1280; 1408 is the peak (see *What works*). Prior champion
  was `...-v1-img1280` at 0.671 / 0.950 (blended 0.7410).
- **Shipped Docker serving config** ([Dockerfile](Dockerfile)) — what goes to
  cloud now:
  - `CV_MODEL_FAMILY=auto`, weights at `/workspace/models/cv/best.pt`,
    `CV_CATEGORY_MAP` = identity `0..17`.
  - `CV_IMGSZ=1408`, `CV_CONF=0.20`, `CV_IOU=0.55`, `CV_AUGMENT=1`, `CV_HALF=1`,
    `CV_CROSS_CLASS_NMS_IOU=0.97`, `CV_SECOND_PASS=0`.
- **Headroom is small (~0.05).** The AE task has far more (~0.15). **If Semis
  time is scarce, AE wins the marginal-hour return on investment** — the
  architecture-family lever is now spent (see below), so remaining CV time goes
  to noise-robustness only.
- **Architecture-family question CLOSED (30 May):** RF-DETR (a DINOv2-backbone
  DETR detector) was trained and submitted — cloud **0.666/0.921, blended 0.7298,
  lost to champion 0.7410**. It *ties* the champion on accuracy with a narrower
  train→cloud gap than the old YOLO lineage, but a strong YOLO already hits the
  same ~0.67 cloud ceiling, so there is no win. **This confirms the ceiling is
  content shift, not backbone.** Champion stays shipped. See the resolved RF-DETR
  section below. **Next CV lever: noise-robustness only.**

### Input purification + multi-model ensembling — MERGED, DEFAULTS OFF (11 Jun)

The `sq` branch (input-purification defense + multi-model ensembling) was merged
to `main` on 11 Jun. **Both capabilities ship OFF by default — the served model
is byte-for-byte the pre-merge champion** (`CV_PURIFY=False`, single
`CV_MODEL_PATH=.../best.pt`). The merge is code-only; there is no behavior
change and nothing was re-submitted. What landed:

- **`_purify_image` (`CV_PURIFY`)** — optional resize → Gaussian blur → JPEG
  re-encode, with bounding-box coordinates rescaled back to original dimensions.
  Intended as a counter to the Finals Noise-task adversarial perturbation. Knobs:
  `CV_PURIFY_JPEG_QUALITY`, `CV_PURIFY_BLUR_SIGMA`, `CV_PURIFY_MIN_SCALE` (the
  committed "Strategy A" preset uses JPEG-q80 only; blur and scale are disabled).
- **Multi-model ensembling** — `CV_MODEL_PATH` accepts a comma-separated list;
  each model runs over tiles / full-image / second pass, and detections are pooled
  before NMS (naive concatenation, **not** weighted box fusion (WBF)).
- 3 mocked unit tests (`test/test_cv_purify.py`, `test/test_cv_ensemble.py`) pass.

**Not yet gated for deploy.** Before flipping `CV_PURIFY=True` or adding a
second model: the only candidate weight (`plusval-v1`) scored **0.640 — below the
champion's 0.671**, so the ensemble is unvalidated and carries ~2× latency
(Finals is per-batch 5 s speed / 10 s hard timeout — see
[../CLAUDE.md](../CLAUDE.md)). The bar set earlier in this file still stands: an
ensemble needs **≥ +0.025 cloud acc** to justify the latency cost. Gate on the
Workbench (cloud + on-hardware A/B) first. If a second model is added to
`CV_MODEL_PATH`, confirm its weight is staged in the image — a missing path
silently degrades to single-model (the loader filters `None`).

### The one problem that dominates CV: the local→cloud distribution shift

Every YOLO submission shows a large, stable gap between local hard-held-out mAP
and cloud accuracy. The gap is driven by **content shift (~80%), not backbone
capacity and not a fixable inference knob.** Diagnostics (Phase C.0): JPEG-quality
and resolution shifts together explain **≤20%** of the gap; the remaining ~80% is
*different scene content* (the cloud eval set is heavier on small-object /
photo-composited scenes than our training distribution). The gap is a function of
**architecture generalization to the hidden distribution, not resolution**:

```
Backbone     local→cloud gap on hard held-out
v8s          ~0.35   (tier1: 0.9049 → 0.556)
v11m         ~0.44   (0.8673 → 0.376 ; 0.9088 → 0.474)
RF-DETR-B    ~0.26   (0.921 → 0.666)   ← narrower gap, but same ~0.67 cloud ceiling
```

Implication: to beat the old tier-1 cloud score (0.556) with a **v11m**-class
model, you would need a hard-held-out score ≥ 0.99 (out of reach); a v8s-class
model needs ≥ 0.91. **The backbone-family lever has now been tested to exhaustion:
RF-DETR's DINOv2 ViT backbone generalizes better (gap 0.26 vs YOLO's 0.35–0.44)
yet still tops out at the same ~0.67 cloud accuracy as YOLO11l — so the ceiling
is the task's content distribution, not any architecture.** Neither more YOLO
sweeping nor a different detector family breaks through it.

### Calibration / measurement facts (trust these)

- **Speed is NOT a CV constraint.** The cloud GPU is much faster than the
  Workbench T4: no-TTA inference at `imgsz=1280` hits cloud speed **0.949** (we
  had projected 0.85–0.92). Spend speed budget freely if it buys cloud accuracy.
- **Blended break-even rule:** the blended score is `0.75·acc + 0.25·speed`, so
  a heavier inference path pays off only if **Δacc ≥ Δspeed / 3**. Gate every
  defense / ensemble decision on cloud blended score, not raw accuracy alone.
- **Two proxies, two jobs:** the **hard held-out split** predicts cloud
  *direction*; the **full local set** (`til test`) predicts cloud *absolute
  level*. A model can win one and lose the other (v11m@1280 is an example of
  this).
- **`test/test_cv.py` pins `score=1.0` on every box** before pycocotools (it is
  competition scaffolding; we cannot change it). This turns local mAP into a
  precision-sensitive metric, so the optimal `CV_CONF` is **higher** than the
  COCO default 0.001 — low confidence floods score-1.0 false positives and tanks
  the local number. Whether the cloud evaluator consumes our real `score` field
  is unknown; we emit it anyway (free upside, no-op otherwise).

### What works / keep doing

- **"Train small, serve big" — upscale-at-inference is the ONLY confirmed lever
  on the YOLO backbone.** v8s: train 768 / serve 896. YOLO11l: train 1024 /
  serve **1408** (the current champion). Reproducible across v8s / v11m / v11l.
  - **Peaks at ~1.375× train resolution, then turns over (30 May sweep).**
    YOLO11l (trained at 1024) cloud acc by serve resolution: 1280 → 0.671,
    **1408 → 0.684 (peak, blended 0.7473)**, 1536 → 0.650 (regressed on both
    cloud and local — full-set mAP 0.986→0.977, small AP 0.860→0.829; too far
    above training scale). The lever had real headroom past 1280 (a correction to
    the earlier "exhausted" read), but the curve is now mapped — **1408 is the
    peak; finer steps are inside cloud σ≈0.053.**
- **Corollary (counter-intuitive, reproducible): bigger model + matched imgsz
  LOSES to smaller model + upscaled imgsz** on this dataset. Native-1280
  training (v2) scored *worse* than 1024-trained weights served at 1280 (v1). Do
  not "fix" the upscale by training at the serve resolution.
  - ⚠️ This lever does **NOT** transfer to DETR — RF-DETR ties resolution to
    positional embeddings, so train and serve at the *same* resolution.
- **plusval / all-data recipe** (`build_final_yolo_dataset.py`,
  `train_v5_plusval.sh`): fold the old validation split back into training, keep
  the hard test split for sanity checks. Addresses ship-class imbalance (the old
  train set had only 34 cruise ships / 52 warships / 56 yachts). This recipe
  produced the champion weights.

### Dead ends — DO NOT REDO (each confirmed negative)

| Lever | What was tried | Result / why it failed |
|---|---|---|
| **TTA on YOLO11l** (`CV_AUGMENT=1` / optimized-v3) | Test-time augmentation for accuracy | Raw acc +0.018 but **blended flat-to-negative**; costs a full speed point. Dead on this distribution. |
| **Raise `CV_CONF` to clean false positives** | Cut low-confidence detections | mAP drops **monotonically** (0.8947 → 0.8854 as conf goes 0.20→0.80). The integrated PR curve needs the high-recall tail; even 2.8%-precision boxes help via recall. |
| **Tiled inference** (`CV_TILE_MODE` 2x2 / 2x1 / 3x2) | Crop into tiles to rescue small objects | **Small AP regressed in every mode**; only a minor *medium*-object lift. Net flat/negative on blended. Patch stays default-off. |
| **v8s-1024 retrain with `copy_paste=0.40`** | Higher-resolution retrain | **Toxic.** Small AP crashed −0.11 to −0.14 (the dataset is already a carefully composed copy-paste mix; 0.40 augmentation shifts the training distribution away from eval). Safe range is 0.0–0.10. |
| **v11m@1024 and @1280** | Bigger backbone | Regressed to cloud 0.376 / 0.474; gap structurally wider (~0.44). A bigger model didn't help; matched imgsz hurt. |
| **Augmented training** (Phase C.1: JPEG + native tile crops) | Train on shifted data to close the gap | Lifted hard held-out +0.04 but **WIDENED the cloud gap** 0.349 → 0.395 — the model specialized further to small-object-dense scenes; the cloud set is closer to easy/sparse. Cloud accuracy flat (−0.003). |
| **Adaptive second-pass rescue** (`ry_v3_adaptive`, `CV_SECOND_PASS`) | Low-confidence base + down-weighted TTA on dense images | **Overfit the held-out JSON**: 0.927 offline ensemble lab → 0.571 cloud. Kept opt-in only; Dockerfile reverted. |
| **OWLv2 zero-shot** (`CV_MODEL_FAMILY=owlv2`) | Open-vocabulary detector | Path exists, but it never beat YOLO and is not proven. Default stays YOLO. |
| **RF-DETR-B** (`CV_MODEL_FAMILY=rfdetr`, DINOv2 DETR) | Different backbone family against the content-shift ceiling | Cloud **0.666/0.921, blended 0.7298 < champion 0.7410**. Ties champion accuracy with a *narrower* gap (0.26 vs YOLO 0.35–0.44) but hits the same ~0.67 cloud ceiling and loses on speed. **Closed the architecture-family question: the ceiling is content, not backbone.** Scaffold kept for a possible RFDETRLarge revisit (low expected value). |

**General lesson:** offline/held-out gains that don't survive the *full-set*
proxy or a clean held-out gate are usually overfit to our hard distribution and
evaporate (or invert) on cloud. Gate on cloud blended score before believing any lift.

### RESOLVED 30 May — RF-DETR did NOT beat the champion (architecture-family question closed)

The architecture-family bet was placed against the content-shift ceiling.
RF-DETR (from Roboflow; DINOv2 ViT backbone; DETR set-prediction, NMS-free) was
the strongest "different family, unknown gap, information-positive" candidate —
its headline benchmark strength is *domain transfer*, which is exactly our failure
mode. Trained RFDETRBase at resolution 728, using the all-data recipe.

**Cloud result (`rfdetr-base-728-v1`, 30 May): 0.666 acc / 0.921 speed, 0/500
errors → blended 0.7298. LOST to the champion's 0.7410.** Accuracy *ties* the
champion (0.666 vs 0.671, within cloud σ≈0.053); blended is lower purely due to
**speed** (0.921 vs 0.950 — RF-DETR-B is heavier per image). Do **not** promote.

**The decisive finding — the ~0.67 cloud ceiling is architecture-independent:**

```
Model         local-hard (HTTP-pycoco)   cloud acc   gap
v8s tier1            0.905                  0.556     0.349
v11m                 0.909                  0.474     0.435
RF-DETR-B            0.921                  0.666     0.255   ← narrower gap, same cloud ceiling
YOLO11l champ        ~0.92                  0.671     ~0.25
```

- **The DINOv2 thesis was partially right:** RF-DETR's local→cloud gap (0.255) is
  genuinely *narrower* than those of the old v8s/v11m backbones (0.35–0.44) — it
  does generalize to the hidden distribution better than they did.
- **But it doesn't win:** a strong YOLO (11l) already reaches the same ~0.67 cloud
  ceiling, so RF-DETR only *ties* on accuracy and loses on the blended score via
  speed. **A fundamentally different, foundation-model-backbone detector hits the
  same wall — the ceiling is content/distribution shift, not backbone.
  Architecture-family swapping is now a closed question for CV.**
- **Training (for the record):** converged cleanly with no overfit; hard-held-out
  torchmetrics mAP50-95 peaked at 0.912 @ epoch 16 / EMA 0.928, plateaued ~ep12.
  We served `checkpoint_best_ema.pth` (rfdetr 1.7.x writes `_ema` + `_regular`,
  not `_total`; EMA 0.928 > regular 0.912). HTTP-pycoco hard held-out = 0.921 @
  conf 0.20.
- **Serve-resolution sweep on the 728-trained weights (30 May, local hard set).**
  Tested the YOLO "serve big" trick on RF-DETR — it does NOT transfer cleanly (ViT
  positional embeddings are tied to the training grid; YOLO is fully convolutional).
  Total mAP drops monotonically (728 → 0.917, 840 → 0.907, 952 → 0.866, 1008 →
  0.841) **but small AP peaks at 952 (0.723 → 0.795, +0.072)** before collapsing
  at 1008. Upscaling helps small objects (more pixels) while pushing medium/large
  objects out of the learned scale range and degrading interpolated positional
  embeddings — net loss. A serve-time 952 submission was predicted to regress to
  ~0.61 cloud; not worth submitting.
- **VERDICT (31 May): native-952 retrain (`rfdetr-base-952-v1`) — LOST, RF-DETR
  thread closed.** Trained natively at resolution 952 (batch=2 / accum=8, no OOM;
  best EMA mAP ~0.951, plateaued ~ep20). Locally it *did* accomplish what the
  serve-resolution sweep could not: test-sweep total mAP **0.946** with small AP
  **0.831** (+0.108 vs 728's 0.723) AND medium/large held (0.915/0.959) — native
  training escaped the serve-time medium/large collapse. **However, cloud acc =
  0.664, statistically identical to 728's 0.666** (blended 0.725 < champion
  0.7473). The +0.029 local gain produced ZERO cloud movement, so local mAP is
  flatly **non-predictive** of cloud accuracy for RF-DETR. We then tested the YOLO
  upscale trick on the native-952 weights: the serve-resolution sweep (952 / 1008 /
  1064) kept total mAP flat (~0.946–0.948, noise) while small AP climbed cleanly
  0.831→0.879 (a gentle 1.12× interpolation with no medium/large collapse this
  time — so the earlier 728-serve collapse prediction was too pessimistic).
  Submitted the best-small config at 1064 (`rfdetr-base-952-up1064`) →
  **cloud 0.623, WORSE** (blended 0.690): the DETR positional-embedding
  interpolation penalty that is masked locally shows up on cloud — the opposite
  of conv-based YOLO. **Net: all three RF-DETR submissions (728 / 952-native /
  1064-serve) cluster at 0.62–0.67 cloud against local mAPs of 0.917–0.948.
  RF-DETR is hard-capped at ~0.665 cloud here regardless of resolution; upscaling
  hurts it. Champion YOLO11l-1408 stays. Do NOT re-run.** Expected value is in AE
  (~0.15 headroom, 40% weight).
- **Optional, LOW priority:** RFDETRLarge *might* squeak past the champion if its
  cloud-accuracy gain outpaces its (certain) speed cost — but the blended math is
  a coin-flip and **AE (~0.15 headroom) + noise-robustness are far higher expected
  value.** Not recommended under the 2026-06-10 deadline. The scaffold
  (`train_rfdetr.py` with `VARIANT=large`, `CV_RFDETR_VARIANT=large`) is ready if
  revisited.
- **Serving an RF-DETR model requires restoring the RF-DETR Dockerfile and
  dependencies.** The committed `cv/Dockerfile` + `cv/requirements.txt` on the
  `ethanAE` branch are the **YOLO11l champion** configuration (reverted after the
  728 submission). The full RF-DETR serving config (ENV block, offline prefetch,
  rfdetr / supervision / transformers-5.x deps) is preserved in git history at
  commit `eb87b61`. To build a 952 RF-DETR image: run
  `git show eb87b61:cv/Dockerfile > cv/Dockerfile` and
  `git show eb87b61:cv/requirements.txt > cv/requirements.txt`, then bump the
  `CV_RFDETR_RESOLUTION` (and the prefetch `RFDETRBase(resolution=...)`) to 952.
  Build, test, submit, then run `git checkout cv/Dockerfile cv/requirements.txt`
  to restore the champion config.
- **Settled gotchas (from the 29 May smoke / 30 May run):**
  1. **Class indexing — confirmed:** Use `--category-offset 1` (Roboflow reserves
     class 0; this yields categories 1..18 plus a dummy id-0; rf-detr reports "19
     classes"; the per-class validation table shows correct label associations).
     Use `CV_CATEGORY_MAP` `{"1":0,...,"18":17}`. Re-confirm with one served
     image.
  2. **T4 OOM — fixed:** the root cause was the multi-scale ~1008 px upsample,
     NOT batch size. `train_rfdetr.py` now sets a **resolution-aware batch
     default** (4 at res ≤ 840, 2 above) with `GRAD_ACCUM` auto-scaled to hold an
     effective batch of 16, plus `GRAD_CHECKPOINT=1`, `MULTI_SCALE=0`. This means
     728 → batch 4 and 952 → batch 2 automatically; override `BATCH` / `GRAD_ACCUM`
     if needed. **Confirmed: the `BATCH=2 GRAD_ACCUM=8` (effective batch 16)
     default held on the 15 GB T4 at resolution 952 — the native-952 run trained
     at batch=2 with no OOM.**
  3. **Albumentations + faster-coco-eval were missing** (rfdetr 1.7.x does not
     auto-install albumentations, resulting in "Built 0 transforms" / no
     augmentation; faster-coco-eval is the torchmetrics mAP backend for the
     validation callback). Both are now pinned in `requirements-dev.txt`; install
     with `pip install -r requirements-dev.txt`.
  4. **Dependency conflict — resolved:** rfdetr 1.7.1 installs without disturbing
     torch (`2.10.0+cu128`, CUDA OK). No need to isolate the image.
- **Resolution coupling:** train and serve at the same resolution (728) — the YOLO
  upscale-at-inference lever does NOT transfer to DETR.
- **Workbench run order:**
  ```bash
  pip install -r requirements-dev.txt                   # rfdetr + albumentations + faster-coco-eval
  python training/cv/build_rfdetr_dataset.py            # → /home/jupyter/cv_rfdetr_dataset (prints CV_CATEGORY_MAP)
  EPOCHS=3 NAME=rfdetr-base-728-smoke python training/cv/train_rfdetr.py   # optional smoke
  python training/cv/train_rfdetr.py                    # full run (BATCH=4 GRAD_ACCUM=4 EPOCHS=30, early-stops ~ep16)
  cp /home/jupyter/cv_runs/rfdetr-base-728-v1/checkpoint_best_ema.pth cv/models/best.pth   # EMA; _total does not exist in 1.7.x
  # til build/test/submit cv with CV_MODEL_FAMILY=rfdetr, CV_RFDETR_RESOLUTION=728,
  #   CV_CONF=0.30, CV_CATEGORY_MAP=<printed map>; sweep conf via sweep_cv_http.py --model-family rfdetr
  # eval_cv_http.py hard held-out → clean gap vs YOLO11l lineage (the go/no-go signal)
  ```

### Next direction (Semis): noise-robustness against opponent perturbations

In Semis/Finals an opponent may add noise to our CV input (bounded by the
fairness gate: SSIM floor + RMSE L2 ≤ 50) before our model sees it. The champion
was trained and scored on clean images only — its accuracy under attack is
unknown. **Verify the exact mechanic in the Wiki spec before starting.** Plan,
cheapest to strongest:

1. **Input purification** behind `CV_PURIFY=1` (JPEG re-encode q=75, random
   resize-and-pad in [1200, 1280], light Gaussian blur σ≈0.5) — negligible cost.
2. **Adversarial fine-tune** of champion weights on a 50/50 clean+noised mix
   (~5k images via our `level10-detector-stress` attack or a PGD surrogate,
   5–10 epochs, ~3–4h on T4). Highest expected-value single experiment; gate
   against a clean held-out to protect clean accuracy.
3. **WBF ensemble** of champion + plusval-v1 (~2× latency; need ≥+0.025 cloud acc).
4. **Re-test TTA under noise** (the "TTA is dead" finding was on clean images;
   cheap to re-check under noise).

Skip (poor cost/benefit at our scale): TV minimization, denoising autoencoders,
feature squeezing, randomized smoothing. Caveat: `level10` is *our own* attack;
gains may not fully transfer to opponents' attacks.

### Key files

- Manager (the file we edit): [src/cv_manager.py](src/cv_manager.py) — robust
  decode → YOLO/RF-DETR/OWLv2 → xyxy→**LTWH `[l,t,w,h]`** adapter (keep this
  conversion; output must be pixel LTWH, never normalized center-XYWH).
- Server (do not edit): [src/cv_server.py](src/cv_server.py)
- Build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt) — weights
  are baked to `/workspace/models/cv/best.pt` from local `cv/models/best.pt`.
- Training/eval: [../training/cv/](../training/cv/) — `build_final_yolo_dataset.py`,
  `train_v5_plusval.sh`, `build_rfdetr_dataset.py`, `train_rfdetr.py`,
  `eval_cv_http.py`, `sweep_cv_http.py` (`--model-family rfdetr`), `gap_diagnose.py`.
- ⚠️ `test/test_cv.py` pins `score=1.0` (see calibration facts above) — keep
  `CV_CONF` higher than the COCO default.

---

## Detailed history (archive — reverse chronological)

> Everything below is the original per-session log: full sweep tables, per-tag
> submission rows, the early-submission timeline, and the deployment/format
> reference. It is redundant with the digest above; consult it for the exact
> numbers behind a claim. (The most recent sessions — RF-DETR scaffold, Semis
> noise plan, Qualifiers-closed final high, shipped-tag detail, and the
> calibration tables — are summarized in the digest and not duplicated here.)

## Phase C.1 — augmented training didn't transfer (16 May 19:00 SGT)

### Pipeline executed as planned

1. `training/cv/build_aug_dataset.py` produced a ~3× train set: originals + JPEG-recompressed copies (q=40–85) + 1024×1024 native-resolution tile crops (≥0.5 visible-area filter).
2. `training/cv/train_v4.sh` trained v8s at `imgsz=1024`, `scale=0.80`, `copy_paste=0.10`, 80 epochs. Final val mAP50-95 ≈ 0.95.
3. Submitted twice — first at the Dockerfile default (model trained at 1024, served at 768 → resolution mismatch), then with a matched config (`imgsz=1280`, `conf=0.001`, `iou=0.7`, `aug=0` baked into ENV).

### Numbers across distributions

| Image set | augc1-v4 | tier1 | Δ |
|---|---:|---:|---:|
| Hard held-out 500 (sweep, small-object dense) | **0.948 / 0.779 small** | 0.905 / 0.746 small | **+0.043** |
| Full local 1250 (`til test`, easier/sparser) | 0.739 / 0.192 small | ~0.85 / high small | **-0.11** |
| **Cloud 500 (hidden)** | **0.553 / 0.959** | 0.556 / 0.956 | **-0.003** |

### Interpretation

- **Training worked on its target.** +0.04 hard held-out is a real improvement, exactly what JPEG + tile-crop augmentation was supposed to deliver.
- **Cloud distribution is closer to full-local than to hard.** Augmentation taught the model to excel at dense-small-object scenes at the cost of sparse easy scenes; the cloud set has more of the latter, so wins and losses cancel.
- **Gap WIDENED**: tier-1 gap was 0.349 (0.905 → 0.556). augc1 gap is 0.395 (0.948 → 0.553). Augmenting on the hard distribution *specialized* the model further from cloud, not closer.
- **`til test` 0.739 with small AP 0.192 is partly a measurement artifact**: at `CV_CONF=0.001`, `test_cv.py` pins all scores to 1.0 before pycocotools, flooding low-confidence detections that drown the PR curve on sparse easy images. The sweep's pycocotools at the same env variables reports 0.948 because it uses real confidence scores. Cloud uses something closer to the sweep's behavior (cloud=0.553, not crashed), so the +0.04 hard-set gain did not appear there.

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

`training/cv/gap_diagnose.py` re-encodes the hard held-out 500 images in-memory
under each transform and re-scores the tier-1 model.

Premise: all five inside-the-box recipe levers (Pass A confusion analysis,
`CV_CONF` sweep, tiled inference, v8s-1024, v11m@1280) had been exhausted. The
local→cloud gap had *not* yet been tested against JPEG quality shift or
scale/resolution shift between the training set and the hidden eval.

### Results

Baseline this run: mAP **0.9049** / small AP 0.7463 (note: +0.01 mAP and +0.10
small AP higher than Pass A's earlier 0.8947/0.6434 — same Docker image, same env
vars in theory; flagged as a baseline anomaly below).

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

- The auto-verdict ("JPEG aug dead, resolution dead") was misleading — it thresholded total mAP at 0.05. The real signal: **small AP drops 10–19 points under every shift**, and Pass A established that small AP is where cloud accuracy is lost. Both axes are real contributors to the cloud gap; they just aren't visible in the total-mAP rollup.
- Total-mAP drops sum to at most −0.07; the cloud gap is 0.349. JPEG + resolution shifts account for **≤ 20%** of the cloud gap.
- The remaining ~80% is most likely **different scene content** (heavier small-object bias in the hidden eval), not a transform applied to similar scenes.
- Realistic cloud ceiling for Phase C augmentation alone: **0.58–0.63**, not 0.70. Reaching 0.70 would also require a backbone change (RT-DETR / YOLOv9) or a wholesale distribution-shift training recipe.

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

Runtime is ~35 min for all 7 transforms; `--transforms baseline,jpeg-q50,downsample-2x`
is a ~15-min smoke run.

### Baseline anomaly (non-blocking)

Pass A on 16 May 02:30 SGT recorded tier-1 hard held-out 0.8947 / small 0.6434.
Phase C.0 diagnostic at 05:30 SGT recorded 0.9049 / small 0.7463 under the same
env vars and image. That is +0.10 small AP "for free" with no model change.
Possible causes: `cv/models/best.pt` replaced between runs (v8s-1024 or v11m
weights briefly substituted); `CV_TILE_MODE` default changed with the
tiled-inference patch; pycocotools / PIL version difference. Worth running
`sha256sum cv/models/best.pt` against the original tier-1 checksum if revisiting.

## Pass A — hard held-out failure analysis (15–16 May)

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

Per-class loss sorted by `(1-AP) * box_count`: the top losers (cargo ship,
fighter jet, helicopter, commercial aircraft, warship) are medium-frequency
classes with AP 0.86–0.88, not rare classes whose AP collapses. **This is not a
class-imbalance story.**

Confusion at IoU ≥ 0.5, conf ≥ 0.20:

```text
ground-truth boxes: 3334
matched:            3333 (class-correct 3317, class-wrong 16, FN 1)
false-positives:    176
```

**16/3334 = 0.48% class confusion.** Aircraft subclass confusion is dead as a
hypothesis. The 176 false positives are almost all in the conf < 0.80 range (see
below).

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

### CV_CONF sweep — Lever 1 (free false-positive cleanup) is dead

Hypothesis: raising `CV_CONF` should drop most false positives at low cost in
true positives and lift mAP. **Falsified** — mAP drops monotonically:

```text
CV_CONF=0.20  mAP50-95=0.8947   ← current shipped
CV_CONF=0.40  mAP50-95=0.8929   -0.0018
CV_CONF=0.60  mAP50-95=0.8909   -0.0038
CV_CONF=0.70  mAP50-95=0.8886   -0.0061
CV_CONF=0.80  mAP50-95=0.8854   -0.0093
```

pycocotools mAP integrates the PR curve at each IoU threshold. Cutting
low-confidence detections amputates the high-recall tail. Even precision-2.8%
predictions contribute via the recall axis. The hidden cloud evaluator probably
consumes our `score` field and/or also runs an integrated PR curve.

### Worst-image eyeballing

`1992.jpg`, `3919.jpg`, `4853.jpg` (worst 3 by per-image loss): photo-composited
backgrounds (mountain valley, urban riverside, forest/lake) with cutout/3D-render
objects pasted at wildly varying scales — some aircraft 25–50 px wide on a
1920×1080 native image. **The small-AP gap is genuinely about pixel-scale
localization at high IoU**, not memorization. At `imgsz=896`, a 30 px native
object becomes ~14 px in the model input — below the resolution at which YOLOv8s
anchors and stride can localize tightly enough to clear IoU ≥ 0.75.

## Tiled inference A/B — Lever 2 is dead (16 May)

Implemented in `cv/src/cv_manager.py` behind `CV_TILE_MODE` (default `off`).
Modes `2x2`, `2x1`, `3x2` crop the 1920×1080 image into overlapping tiles
(`CV_TILE_OVERLAP=0.20` default), run the detector at `CV_TILE_IMGSZ=768` per
tile, optionally also on the full image at `CV_IMGSZ=896` (`CV_TILE_FULL_PASS=1`
default), drop boxes touching internal tile edges (`CV_TILE_EDGE_MARGIN=4`), and
merge with class-aware NMS at `CV_TILE_MERGE_IOU=0.50`.

Off-mode hard held-out is bit-identical to the prior tier-1 number (0.8947) —
sanity confirmed.

| Mode | Total mAP | Small | Medium | Large | Tile passes |
|---|---:|---:|---:|---:|---:|
| `off`  | 0.8947 | 0.6434 | 0.8506 | 0.9128 | 1 |
| `2x2`  | 0.8993 | 0.6139 | 0.8475 | 0.9185 | 5 |
| `2x1`  | 0.8771 | 0.5929 | 0.8329 | 0.8925 | 3 |
| `3x2`  | 0.9009 | 0.6255 | **0.8728** | 0.9180 | 7 |
| `3x2` em=0 ov=0.30 | similar pattern (no small-AP recovery) | — | — | — | 7 |

Interpretation:

- **Small AP regressed in every tiled mode.** The hypothesis (tiling rescues small-object recall) was wrong on this dataset. The `edge-margin=0` test showed the problem is not the 4 px filter dropping legitimate small detections; it is how the model handles partial objects within tile crops.
- **The 3x2 lift is real, but it is a medium-object lift.** Medium AP +0.022 (0.8506 → 0.8728) explains nearly the entire +0.006 total improvement. 3x2 tiles are 738×600, so a 60 px native object becomes ~80 px effective at `imgsz=768` — which falls squarely in the medium-object bucket.

Speed math (original estimate was conservative; cloud GPU is faster than
estimated):

```text
Current shipped : 0.75 * 0.556 + 0.25 * 0.956 = 0.656
3x2 best case   : 0.75 * 0.580 + 0.25 * 0.870 = 0.653  (+0.024 cloud accuracy assumed)
3x2 likely case : 0.75 * 0.560 + 0.25 * 0.870 = 0.638  (cloud accuracy flat)
```

**Decision**: do not ship tiled inference on tier-1 weights. Patch stays merged
but disabled by default. Retested all four tile modes against v8s-1024 weights
after the retrain finished — no candidate improved on tier-1. Lever 2 dead.

## v8s-1024 retrain (16 May, NOT submitted)

3.8 hr on T4 via `training/cv/train_v4.sh`:

- `model=yolov8s.pt`, `imgsz=1024` (vs tier-1's 768), `epochs=80`, `batch=10`
- `copy_paste=0.40` (vs tier-1's 0.10), everything else matched tier-1

Final Ultralytics val (500-image split): `mAP50-95 0.879` (vs tier-1's 0.920 on
the same split — a real val regression of −0.041).

Hard held-out across three inference modes:

| Inference mode | Total mAP | Small | Medium | Large |
|---|---:|---:|---:|---:|
| `imgsz=1024 aug=1` | 0.8217 | 0.5327 | 0.7591 | 0.8425 |
| `imgsz=1280 aug=0` | 0.8370 | 0.5168 | 0.7594 | 0.8550 |
| `imgsz=1024 + tile=3x2` | 0.8361 | 0.5058 | 0.7953 | 0.8507 |

vs **tier-1 baseline** (0.8947/0.6434/0.8506/0.9128): total mAP regressed
−0.058 to −0.073; **small AP regressed −0.110 to −0.138 — the opposite of the
hypothesis** (the retrain was supposed to lift small AP via higher resolution;
instead it crashed it).

Most likely cause: `copy_paste=0.40` was too aggressive on this dataset, which
is *already* a carefully composed copy-paste set. Aggressive training-time
copy-paste slaps random instance crops onto random training images, shifting the
training distribution away from eval. **Key lesson for any future v8s retrain:**

```text
copy_paste     dataset behavior
0.0 - 0.10     safe; tier1's recipe
0.20 - 0.30    untested; v3 used 0.30 (also regressed, confounded with v11m + matched-imgsz)
0.40           toxic; this run
```

Not submitted.

## v11m@1280 submission analysis

Submitted as `v11m-1280-noaug-v1` on 16 May 04:04. Sweep top results:

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

- **Speed beat projection.** Projected 0.85–0.92; cloud delivered 0.949. The cloud GPU is faster than the Workbench T4; no-TTA at 1280 doesn't bottleneck.
- **Accuracy gap is structurally wider than v8s.** v11m@1280 gap = 0.435 vs v11m@1024 gap = 0.491 vs v8s tier-1 gap = 0.349. The v11m gap is wider regardless of resolution.

`til test cv v11m-1280-noaug-v1` printed local full-set:

```text
mAP@.5:.05:.95: 0.812
small AP: 0.144   ← collapsed on easy-but-numerous full-set images
medium AP: 0.701
large AP: 0.878
```

This is a different signal from the hard held-out: the full set is dominated by
easy images where v11m@1280 over-fit (small AP collapsed to 0.144). The hard
held-out is small-object-heavy, so v11m@1280 looks great there. The full set
looks worse because the model is mis-calibrated on common easy cases. **The hard
split was the better proxy for cloud direction; the full set was the better proxy
for cloud absolute level.**

## v11m@1280 sweep (15 May, before v11m-1280-noaug-v1 submission)

`aug=1` sweep at `imgsz=1280` after the v3-pre submission:

```text
mAP=0.9141  conf=0.001 iou=0.70 imgsz=1280 aug=1   ← best
mAP=0.9140  conf=0.05  iou=0.70 imgsz=1280 aug=1
mAP=0.9140  conf=0.10  iou=0.70 imgsz=1280 aug=1
mAP=0.9136  conf=0.001 iou=0.50 imgsz=1280 aug=1
(all imgsz=1024 rows: 0.8673 max — confirmed dead)
```

Hard held-out 0.9141 at `imgsz=1280` (+0.047 vs 1024; +0.009 vs tier-1's
0.9049). Small AP recovered to 0.742 (vs tier-1's 0.746 — essentially matched).

Speed problem with `aug=1`: ~1.04 s/img → 520 s for 500 images → speed score
~0.71. Blended: `0.75×0.60 + 0.25×0.71 = 0.628` vs tier-1's `0.656`. TTA killed
the blended score, so we switched to the `aug=0` sweep (results in section
above), which became the `v11m-1280-noaug-v1` submission.

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
yolo11l-896-plusval-v1 22/05 23:45     0.640   0.954   0 / 500   YOLO11l on plusval data. Served at imgsz=896. Prior high.
yolo11l-1024-alldata-final-v1-img1280 22/05 22:36 0.671 0.950 0 / 500 FINAL BLENDED HIGH (0.7410). v1 weights (all-data 1024px training) served at imgsz=1280. +0.031 over plusval-v1.
yolo11l-1280-alldata-final-v2 23/05 20:56 0.654 0.950 0 / 500 v2 weights served at imgsz=1024 (Dockerfile lag). Regressed -0.017 acc vs v1-img1280 — bigger-trained model + matched-imgsz lost to smaller-trained + upscale, same v8s/v11m pattern.
yolo11-optimized-v3 24/05 20:43 0.672 0.940 0 / 500 v2 weights @ 1280 + TTA + CONF=0.20. Raw-acc +0.001 vs v1-img1280 but blended -0.002 (0.7390 vs 0.7410). TTA dead.
```

## Detailed timeline (early submissions)

### cv-yolo-v2-best (14 May 14:00) — hard split + 768 inference tuning

- Created non-leaky hard splits from `/home/jupyter/novice/cv/annotations.json`: 4000 train / 500 val / 500 test. The hard test split had 3334 boxes, denser than train and intentionally useful for stress eval.
- Trained `yolov8s.pt` 80 epochs at `imgsz=768`, `batch=12`, `device=0`, with stronger augmentation (`close_mosaic=10`, `mixup=0.10`, `degrees=5`, `scale=0.60`). 2.329 hr on T4.
- Ultralytics val on 500-image val split: P 0.984, R 0.961, mAP50 0.985, mAP50-95 0.920.
- Default-container hard held-out HTTP eval (`CV_CONF=0.25 CV_IOU=0.70 CV_IMGSZ=640`): 0.8043 mAP50-95.
- HTTP sweep on 100 hard-test images: `imgsz=768` was the main win. Best smoke setting `CV_CONF=0.25 CV_IOU=0.50 CV_IMGSZ=768` scored 0.8652 on the 100-image slice.
- Full hard held-out HTTP confirmation at those settings: 0.8589 mAP50-95, 0.9414 mAP50, small AP 0.5596, medium 0.7557, large 0.8911.
- Full local `til test cv cv-yolo-v2-best`: 0.8839 mAP@.5:.05:.95, 0 errors, 1250/1250 batches in 13:59.
- Official: **0.549 / 0.960**, 0/500 errors. +0.147 official accuracy vs `cv-yolo-ft-v1`.

Interpretation: non-leaky held-out eval was directionally useful even though the hidden gap is still large (0.8589 → 0.549). Biggest confirmed lever: matching the inference `imgsz=768` to the training resolution. Remaining weakness: hidden small-object and aircraft-subclass distribution shift, not output format.

### cv-yolo-ft-v1 (14 May 03:53) — trained 18-class YOLOv8s

- Converted `/home/jupyter/novice/cv/annotations.json` into Ultralytics YOLO dataset: 4500 train / 500 val images, 16620 train / 1881 val boxes.
- Trained `yolov8s.pt` 60 epochs at `imgsz=640`, `batch=16`. 1.297 hr on T4.
- Ultralytics val: P 0.964, R 0.936, mAP50 0.975, mAP50-95 0.905.
- `til test` full local novice CV set: mAP@.5:.05:.95 0.885, mAP50 0.951, 0 errors, 1250 batches in 08:30.
- Official: **0.402 / 0.963**, 0/500 errors.

Interpretation: this is not an LTWH/output-format failure. A bounding-box format bug would score near 0.0 or cause result-loading errors. The score improved from 0.044 → 0.402 with unchanged speed and no errors, proving the schema and `xyxy → LTWH` adapter work correctly. The local→official gap (0.885 → 0.402) is hidden distribution shift / harder images / small objects / label-domain differences, not a plumbing issue.

### yolo-baseline / yolo-til-map-v2 (14 May) — pretrained detector + robust fallback

Implemented serving plumbing:

- Wrapped image decode + model inference so bad images return `[]` instead of erroring. `cv_server.py` passes the evaluator `key` to the manager for debug logs.
- Added `PIL.ImageOps.exif_transpose(...).convert("RGB")` so grayscale/RGBA/EXIF oddities are normalized before inference.
- Added Ultralytics `yolov8n.pt` loaded once in `CVManager.__init__`.
- Converts xyxy → official LTWH `[l,t,w,h]`, clamps boxes to image bounds, drops zero-area boxes, and emits plain Python `float`/`int`.
- Added a default YOLO-index to TIL category-id mapping. Set `CV_CATEGORY_MAP` to a JSON dict (e.g. `{"4": 3}`) or a path to a JSON file when tuning mappings.

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

This is only a baseline — a COCO-pretrained YOLO cannot distinguish drone / fighter / cargo / light aircraft, missile / tank / van, or specific ship subclasses without fine-tuning.

### latest (12 May 03:52) — submission plumbing only

Sanity-check submission: service starts, endpoint accepts requests, JSON shape correct. No detection logic. Scored 0.000 as expected.

## Tier-1 sweep — picking the 0.9049 row

After v2-best (0.549 cloud), env-variable-only changes to find the tier-1 config
(same weights):

- `CV_AUGMENT=1` (Ultralytics flip + multi-scale TTA) — wins on small/dense scenes
- `CV_HALF=1` (FP16 inference) — speed-neutral
- `CV_IMGSZ=896` — best point on the resolution vs. accuracy plateau
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

Confirmed: `imgsz=896 + aug=1` is the winning regime; IoU plateau 0.45–0.60
(Δ<0.001); `conf=0.20` slightly beats 0.25. Local→official gap 0.9049 → 0.556
= 0.349.

## v11m@1024 (Tier 2, May 15) — full training history

- Trainer: `training/cv/train_v3.sh`. YOLOv11m at `imgsz=1024`, `batch=6`, AdamW with cosine LR, `copy_paste=0.30` + `mosaic=1.0` + `mixup=0.15`.
- Wallclock estimate revised: the original 4–5 h estimate was wrong. YOLOv11m@1024 runs ~6:30/epoch on T4 → 120 epochs ≈ 13 h. v8s@768 was 80 epochs in 2.3 h. The compound factor (m vs s + 768 vs 1024 + 80 vs 120 epochs) is ~5.6×.
- The first run died at epoch 58/120 from CUDA OOM caused by leftover Docker containers from an earlier sweep consuming ~10 GB VRAM. `best.pt` was saved (epoch 56, val mAP50-95 0.895).
- Resumed from `last.pt`, ran to completion at 120 epochs (~7 h after resume). Final Ultralytics val mAP50-95 **0.937**, mAP50 **0.994** on 500-image val split. Per-class low: `warship` 0.862; high: `car` 0.988.
- Sweep on fully-trained `best.pt` against the hard test split: max mAP50-95 0.8673 at `conf=0.001 iou=0.70 imgsz=1024 aug=1`. **0.038 below v8s tier-1's 0.9049.** Small-object AP: v11m 0.587 vs tier-1 0.746.
- Submitted `cv-yolo11m-v3-pre` (15 May 11:34): **0.376 / 0.955**, 0/500 errors. **REGRESSED −0.180 vs tier-1 0.556.**

Post-mortem:

1. **Resolution mismatch.** v8s tier-1 was trained at 768 / inferred at 896 (upscale); upscaling helped small objects (+0.04 in the v8s sweep). v11m was trained at 1024 / inferred at 1024 (matched). v11m@1280 (later sweep) lifted the hard held-out to 0.9141 but did not transfer to cloud.
2. **Small-object hit from high-IoU bins.** v11m mAP50 = 0.997 (essentially perfect detection); the loss is in high-IoU precision bins (≥0.75) — exactly what higher inference resolution helps.
3. **Bigger model + matched imgsz < smaller model + upscaled imgsz** on this dataset. Counter-intuitive but reproducible.
4. **Local→official gap is structurally wider** (0.491 vs tier-1's 0.349). The hidden eval is harsher on bounding-box precision than the hard held-out suggests.

## What we run

Schema-safe Ultralytics YOLO service in [src/cv_manager.py](src/cv_manager.py),
packaged by [Dockerfile](Dockerfile):

```python
def cv(image_bytes: bytes, key=None) -> list[dict]:
    # Decode robustly, run YOLO, convert xyxy -> official LTWH.
    # Any decode/model failure returns [] and logs key/byte size.
```

Docker expects the trained checkpoint at `/workspace/models/cv/best.pt`, copied
from local `cv/models/best.pt` during build. `CV_CATEGORY_MAP` is the identity
mapping 0..17 (the fine-tuned model was trained on the official TIL label order).
Shipped inference defaults baked into the image vary by tag (see the tier-1 sweep
section).

## `test_cv.py` pins `score: 1.0` — gotcha

`test/test_cv.py` (the local evaluator that powers `til test`) hardcodes
`score=1.0` on every box before running pycocotools. Standard COCO mAP
integrates the PR curve over confidence scores; pinning all scores to 1.0 turns
mAP into a yes/no precision metric where every false positive at score 1.0
directly hurts you.

This is why the optimal `CV_CONF` is **higher** than the COCO default 0.001:
lower confidence adds false positives as score-1.0 ties and tanks precision. We
cannot unilaterally change the evaluator (it is competition scaffolding); the best
we can do is sweep conf up in the 0.20–0.60 range and pick the F1 optimum on the
hard held-out. Whether the cloud evaluator does the same or consumes our `score`
field is not knowable without submitting, but the env-driven design lets us pivot
either way.

## What CV needs to hit 0.7 (left for later if revisited)

None of the following have high confidence under our timeline:

1. **RT-DETR-L or YOLOv9c transfer.** Different backbone family with unknown local→cloud gap. ~6 h GPU per train. Could land anywhere 0.45–0.65 — high variance, but information-positive.
2. **Conservative re-augmentation.** Drop tile crops, keep only JPEG augmentation. Recovers full-local performance. Realistic cloud 0.55–0.58. Doesn't reach 0.7.
3. **Wholesale distribution-shift recipe** (synthetic photo-composition, harder mosaic, train at native-resolution multi-scale). Half a day of dataloader work + 6–8 h training. Plausible but not credible at deadline.

Default: **stay parked at tier-1.** Submission slots are uncapped, so any of the
above can be tried in parallel without risking the leaderboard position — but each
should be re-evaluated against AE/NLP marginal-hour return before kicking off.

## Output-format checks to preserve

- YOLO training labels are normalized center-XYWH; that is the training format only.
- API output must stay pixel LTWH `[left, top, width, height]`.
- `cv_manager.py` converts Ultralytics xyxy → LTWH and clamps to image bounds. Keep this adapter unchanged unless a local `til test` run catches a regression.
- `cv_manager.py` emits a `score` field per detection (Ultralytics confidence). The official `cv/README.md` schema doesn't list `score`, but `test/test_cv.py` silently overwrites it with 1.0 regardless. Including the real confidence is free upside if the cloud evaluator consumes it, and a no-op otherwise.

## Reproducibility / pointers

- Manager source: [src/cv_manager.py](src/cv_manager.py)
- HTTP server (don't edit): [src/cv_server.py](src/cv_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Score/status rollup: [../RESULTS.md](../RESULTS.md)
- Training scripts: [../training/cv/](../training/cv/) — `prepare_yolo_dataset.py`, `build_aug_dataset.py`, `train_v3.sh`, `train_v4.sh`, `eval_cv_http.py`, `sweep_cv_http.py`, `analyze_cv_failures.py`, `gap_diagnose.py`
