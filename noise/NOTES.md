# Noise — notes & history

Last updated: 24 May 2026 — **Level 10 detector-stress shipped.**
`level10-detector-stress` submitted 24 May 16:53 SGT scored
**1.000 / 0.947** with 0/500 errors. Workbench `til test noise
level10-detector-stress` passed 500/500 fairness locally before submission:
L2 RMSE mean `28.4257`, L2 inside mean `27.2377`, SSIM inside mean
`0.7254`, SSIM inside min `0.4118`. Cloud speed improved from Level 9's
`0.934` to `0.947` despite spending more of the legal distortion budget.

Noise still has no direct Qualifier reward per the official spec, so this
score does not move the leaderboard. The point of shipping was to swap
the deployed container from a pure JPEG round-trip to an actual
adversarial perturbation pipeline before Finals.

Per-task working log for Noise (adversarial image noising). For the
authoritative input/output/scoring spec see [README.md](README.md) and the
official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#noise).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current code state (shipped as `level10-detector-stress`)

Live on `main` and in the deployed `level10-detector-stress` image as of
24 May:

- `src/noise_manager.py` — Level 10 detector-stress inference. It keeps
  the Level 9 AdvGAN generator, then saturates weak regions with legal
  high-frequency, multi-scale, edge-aware perturbations aimed at CNN
  detector feature maps. Total perturbation remains clamped to
  `epsilon=32/255`, output is re-encoded as JPEG quality 95, and the
  manager falls back to echoing the original input on any exception.
  `NOISE_MODE=advgan` restores pure Level 9 behavior without code changes.
- `src/advgan.py` — `Generator` (small conv autoencoder, 8→16→8→3
  channels, tanh output) plus `AdvGANTrainer`.
- `src/advgan_generator.pth` — ~14 KB trained weights, baked into the
  Docker image via `COPY src .`.
- `src/train_advgan.py` — training driver against a ResNet18 victim on
  the Imagenette dataset.
- `src/simulate_damage.py` — local visualizer for manual inspection.

Gitignored (not part of the image, kept locally for training):

- `src/imagenette2/` — 1.5 GB Imagenette training set.
- `src/damage_report.png` — analysis artifact from the visualizer.

## What's shipped to cloud

**`level10-detector-stress` — official 1.000 / 0.947
(24 May 16:53 SGT, 0 of 500 errors).** Workbench validator before cloud:
500/500 images pass, L2 RMSE mean `28.4257`, L2 inside mean `27.2377`,
SSIM inside mean `0.7254`, SSIM inside min `0.4118`. This is now the live
Semifinals/Finals CV-disruption container.

`level9` — official 1.000 / 0.934 (24 May 06:18 SGT, 0 of 500 errors).
Validator: SSIM inside mean `0.9839` (min `0.9414`), L2 inside mean
`6.6800`, 500/500 images pass the fairness gate. Superseded by
`level10-detector-stress`.

`latest` — official 1.000 / 0.970 (12 May 03:54 SGT, 0 of 500 errors).
Plain JPEG re-encode baseline; superseded by `level9` and then
`level10-detector-stress`, but kept as a fallback tag.

## Why the architecture changed

The Level 7.1/8 line iteratively crafted noise per query using PGD across
an ensemble of surrogate classifiers (ResNet18, MobileNetV3, SqueezeNet,
ViT-B/16). That's 20 PGD steps × multiple surrogates per request, which
costs full forward+backward passes on every input. Empirically this would
have walked into the Qualifier 30-minute speed wall the moment the test
set got larger than a few dozen images.

Level 9 (AdvGAN) trains a small generator network offline so that
inference is a single forward pass — sub-millisecond on the deployed
GPU. The trained generator should produce a near-optimal perturbation
pattern for each input without per-query optimization.

## Submission history

```text
Tag                       Submitted          Score   Speed   Errors    Notes
level10-detector-stress   24/05 16:53        1.000   0.947   0 / 500   Detector-stress mode. Workbench: 500/500 fair, L2 mean 28.4257, L2 inside mean 27.2377, SSIM inside mean 0.7254, min 0.4118.
level9                    24/05 06:18        1.000   0.934   0 / 500   AdvGAN generator, ε=32/255, JPEG q=95. SSIM inside mean 0.9839, L2 inside mean 6.6800, 500/500 fairness pass.
latest                    12/05 03:54        1.000   0.970   0 / 500   Clean JPEG re-encode baseline (superseded by `level9` and `level10-detector-stress`).
```

## Levers if Finals requires more disruption

`level10-detector-stress` is the current shipped tool. It spends much more
of the legal distortion budget than Level 9 while still passing fairness
(Workbench SSIM-inside min `0.4118` vs floor `0.3`), so do not tweak it
blindly before a match. If Finals shows the current perturbation is not
degrading opponents' CV enough, the next useful levers are:

- Re-train the Generator against a detector closer to the competition's
  target distribution (current training target is ResNet18 / Imagenette
  — see `train_advgan.py`). YOLO-class objectives would likely transfer
  better than ImageNet classification gradients.
- The current architecture is tiny (~14 KB weights, 4 conv layers).
  Headroom on capacity if the validator allows it.
- Epsilon stays at `32/255`; further strength should come from a
  target-aligned generator/objective rather than blindly lowering SSIM
  margin.

If `level10-detector-stress` is good enough for Finals as-is, leave it
alone.

## Semifinals disruption candidate: detector-stress mode

24 May PM candidate shipped as `level10-detector-stress`:
`src/noise_manager.py` now defaults to
`NOISE_MODE=detector_stress`. This keeps the Level 9 AdvGAN generator but
saturates its weak regions with a legal high-frequency, multi-scale,
edge-aware perturbation pattern aimed at CNN detector feature maps. It still
clamps total perturbation to the existing `epsilon = 32/255` and re-encodes
as JPEG quality 95. `NOISE_MODE=advgan` restores the shipped Level 9 behavior
without code changes.

Local Mac smoke on a synthetic structured JPEG:

```text
NOISE_MODE=advgan          RMSE ~= 4.43
NOISE_MODE=detector_stress RMSE ~= 29.33
Synthetic fairness check       pass (L2 inside 28.68, SSIM inside 0.761)
```

The official fairness config allows RMSE inside up to `50` and SSIM inside
down to `0.3`, so this candidate deliberately spends more of the legal
distortion budget than Level 9. Workbench `til test noise
level10-detector-stress` passed before cloud submission with 500/500 fair
images; the official cloud eval returned 0/500 errors, Score `1.000`, Speed
`0.947`.

## Reproducibility / pointers

- Manager source: [src/noise_manager.py](src/noise_manager.py)
- Generator architecture: [src/advgan.py](src/advgan.py)
- Trained weights: [src/advgan_generator.pth](src/advgan_generator.pth)
- Training driver: [src/train_advgan.py](src/train_advgan.py)
- Visualizer: [src/simulate_damage.py](src/simulate_damage.py)
- HTTP server (don't edit): [src/noise_server.py](src/noise_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Score/status rollup: [../RESULTS.md](../RESULTS.md)
