# Noise — notes & history

Last updated: 24 May 2026 — **Level 9 shipped.** `level9` submitted
24 May 06:18 SGT scored **1.000 / 0.934** with 0/500 errors. Validator
metrics on the cloud batch: SSIM inside mean `0.9839` (min `0.9414`,
max `0.9915`), L2 inside mean `6.6800`, 500/500 images pass the per-image
fairness gate. The AdvGAN generator (`83051b9`) cleared the SSIM/RMSE
validity check at the shipped `epsilon = 32/255`; no need to drop epsilon
or retrain for now. Speed dipped from the JPEG baseline's `0.970` to
`0.934` for the extra Generator forward pass + bilinear upsample.

Noise still has no direct Qualifier reward per the official spec, so this
score does not move the leaderboard. The point of shipping was to swap
the deployed container from a pure JPEG round-trip to an actual
adversarial perturbation pipeline before Finals.

Per-task working log for Noise (adversarial image noising). For the
authoritative input/output/scoring spec see [README.md](README.md) and the
official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#noise).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current code state (shipped as `level9`)

Live on `main` and in the deployed `level9` image as of 24 May:

- `src/noise_manager.py` — Level 9 AdvGAN inference. Single Generator
  forward pass per image: `Generator(image) -> raw_noise`, bilinear
  upsample to original resolution, clamp to `epsilon=32/255`, add to
  image, clamp to `[0, 1]`, re-encode as JPEG (quality 95). Falls back to
  echoing the original input on any exception.
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

**`level9` — official 1.000 / 0.934 (24 May 06:18 SGT, 0 of 500 errors).**
Validator: SSIM inside mean `0.9839` (min `0.9414`), L2 inside mean
`6.6800`, 500/500 images pass the fairness gate. This is now the live
container.

`latest` — official 1.000 / 0.970 (12 May 03:54 SGT, 0 of 500 errors).
Plain JPEG re-encode baseline; superseded by `level9` but kept as a
fallback tag.

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
Tag       Submitted          Score   Speed   Errors    Notes
level9    24/05 06:18        1.000   0.934   0 / 500   AdvGAN generator, ε=32/255, JPEG q=95. SSIM inside mean 0.9839, L2 inside mean 6.6800, 500/500 fairness pass.
latest    12/05 03:54        1.000   0.970   0 / 500   Clean JPEG re-encode baseline (superseded by `level9`).
```

## Levers if Finals requires more disruption

`level9` passes validity comfortably (min SSIM `0.9414`, well above any
typical floor), so there is room to push harder if Finals shows the
current perturbation isn't degrading opponents' CV enough:

- Re-train the Generator against a detector closer to the competition's
  target distribution (current training target is ResNet18 / Imagenette
  — see `train_advgan.py`). YOLO-class objectives would likely transfer
  better than ImageNet classification gradients.
- The current architecture is tiny (~14 KB weights, 4 conv layers).
  Headroom on capacity if the validator allows it.
- Epsilon stays at `32/255`; could push toward the validator edge if
  needed, but doing so without a re-trained, target-aligned generator is
  unlikely to help.

If `level9` is good enough for Finals as-is, leave it alone.

## Reproducibility / pointers

- Manager source: [src/noise_manager.py](src/noise_manager.py)
- Generator architecture: [src/advgan.py](src/advgan.py)
- Trained weights: [src/advgan_generator.pth](src/advgan_generator.pth)
- Training driver: [src/train_advgan.py](src/train_advgan.py)
- Visualizer: [src/simulate_damage.py](src/simulate_damage.py)
- HTTP server (don't edit): [src/noise_server.py](src/noise_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#noise](../SUMMARY.md)
