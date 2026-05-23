# Noise — notes & history

Last updated: 24 May 2026 — **Reactivated.** The codebase has moved from a
JPEG re-encode baseline to an active adversarial perturbation pipeline:
Level 8 (PGD/EoT/ViT ensemble) landed in `7117b9c` and Level 9 (AdvGAN
generator) replaced it in `83051b9`. **Nothing new has been submitted yet** —
last cloud row is still `latest` at `1.000 / 0.970` from 12 May. The Level 9
manager needs a `til build` / `til test` / `til submit` cycle before there is
any new cloud signal.

Noise still has no direct Qualifier reward per the official spec, so this
work is Finals-facing rather than leaderboard-facing. Worth shipping if and
only if a Level 9 submission still passes the SSIM/RMSE validity gate.

Per-task working log for Noise (adversarial image noising). For the
authoritative input/output/scoring spec see [README.md](README.md) and the
official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#noise).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current code state (unshipped)

Live on `main` as of 24 May:

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

**`latest` — official 1.000 / 0.970 (12 May 03:54 SGT, 0 of 500 errors).**

This row predates all of Level 8/9. The deployed container still runs the
old JPEG re-encode baseline. The Level 9 path on `main` has never been
through `til build` / `til test` / `til submit`.

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

## Before submitting

The shipped score is `1.000 / 0.970` against a JPEG round-trip. Level 9
*will* introduce perturbation that the cloud SSIM/RMSE validator may or
may not accept. Do these locally first:

1. `til build noise level9` on the Workbench.
2. `til test noise level9` — verify (a) no errors, (b) the SSIM/RMSE
   validity score is still acceptable, (c) speed stays in the green.
3. Only then `til submit noise level9`.

The current code base-rates `epsilon = 32/255`. If the validity gate
rejects, the obvious lever is to drop epsilon (e.g. 16/255 or 8/255)
before re-training the generator.

## Submission history

```text
Tag       Submitted          Score   Speed   Errors    Notes
latest    12/05 03:54        1.000   0.970   0 / 500   Clean JPEG re-encode baseline (still the deployed container)
```

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
