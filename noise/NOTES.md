# Noise — notes & history

Last updated: 13 May 2026

Per-task working log for Noise (adversarial image noising). For input/output
spec see [README.md](README.md). For submission history across all tasks see
[../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`latest` — official 1.000 / 0.970 (12 May 03:54 SGT, 0 of 500 errors).**

Already scoring `1.000`. Noise has **no observed qualifier weight** in the
handbook — it's required-but-unweighted, so the perfect score doesn't move
the leaderboard. Don't optimize further unless qualifier rules change or
finals weight it.

## What our model runs on

From [src/noise_manager.py](src/noise_manager.py):

- Decode incoming image bytes
- Re-encode as a clean JPEG
- Return base64-encoded bytes

No perturbation, no surrogate model, no adversarial signal. The baseline
simply round-trips the image through JPEG re-compression. Valid and "safe"
(passes the fairness/validity check), scores 1.000.

CPU-bound, fast (speed score 0.970).

## Submission history

```text
Tag       Submitted          Score   Speed   Errors    Notes
latest    12/05 03:54        1.000   0.970   0 / 500   Clean JPEG re-encode baseline
```

## Why we're not touching this

Per [../SUMMARY.md "Scoring weight"](../SUMMARY.md):

```text
AE   40%
ASR  20%
CV   20%
NLP  20%
Noise: required/useful, but no direct qualifier points observed
```

The 1.000 / 0.970 baseline already satisfies the "required" criterion. There
is no leaderboard point gain available here within the qualifier scoring
formula we understand.

Reserved future work, only if rules change or finals weight it:

1. **Cheap path**: bounded random noise within the perturbation budget. Easy
   to add, modest adversarial impact.
2. **Better path**: untargeted FGSM/PGD against a public surrogate classifier
   (ResNet/ViT), clipped to budget. Risk: JPEG re-encoding wipes high-freq
   adversarial signal so compensation may be needed.
3. **Best path**: ensemble attack across multiple surrogates for
   transferability. Highest cost.

All deferred indefinitely.

## State

Noise is **done**. Don't spend hours here until we have leaderboard evidence
the score matters.

## Reproducibility / pointers

- Manager source: [src/noise_manager.py](src/noise_manager.py)
- HTTP server (don't edit): [src/noise_server.py](src/noise_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#noise](../SUMMARY.md)
