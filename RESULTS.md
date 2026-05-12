# TIL-AI 2026 Submission Results

Team: `melanie-minions`
Last updated: 12 May 2026

## Latest submitted scores

```text
Task   Image                    Tag     Submitted             Errors        Score   Speed
NLP    melanie-minions-nlp      latest  12/05/2026 03:23:35   0 / 700       0.301   0.971
ASR    melanie-minions-asr      norm-v1 12/05/2026 16:23:35   0 / 400       0.877   0.864
CV     melanie-minions-cv       latest  12/05/2026 03:52:32   4 / 500       0.000   0.981
Noise  melanie-minions-noise    latest  12/05/2026 03:54:55   0 / 500       1.000   0.970
AE     melanie-minions-ae       latest  12/05/2026 04:20:34   0 / 30        0.051   0.856
```

## Qualifier weighted score estimate

Qualifier weights from the handbook:

```text
AE   40%
NLP  20%
ASR  20%
CV   20%
Noise has no direct qualifier weight observed
```

Using raw task scores only:

```text
0.40 * AE 0.051  = 0.0204
0.20 * NLP 0.301 = 0.0602
0.20 * ASR 0.877 = 0.1754
0.20 * CV 0.000  = 0.0000
--------------------------------
Estimated weighted qualifier score = 0.2560
```

Using the observed ~75% score / 25% speed blend:

```text
AE   contribution = 0.1009
NLP  contribution = 0.0937
ASR  contribution = 0.1748
CV   contribution = 0.0491
--------------------------------
Estimated blended qualifier score = 0.4184
```

## Notes

- All 5 tasks have been submitted successfully at least once.
- ASR norm-v1 (12 May 16:23) official `0.877 / 0.864` is the second submission. **A newer local build is ready to ship**:
  - `vad-off-v1` — local English WER **0.0554** (down from norm-v1's 0.0759). Adds `vad_filter=False`, hallucination guards (`no_speech_threshold=0.6`, `log_prob_threshold=-1.0`, `compression_ratio_threshold=2.4`, `temperature=0.0`), spoken-form ordinals (`23rd` → `twenty third`), coordinate-safe decimal regex, tighter silence guard for short noisy clips. **Recommended next submission.** Predicted official ≈ 0.91-0.92.
  - `vad-off-v2` — REGRESSED to local WER 0.0604 after reversing the slang-prompt order (intent was to survive Whisper's truncation but reversing over-primed the decoder). Reverted. Do not submit.
- The local `1 - MER` number is a **scoring artifact** of the local manifest being English-only (three other language buckets contribute 0 to the divide-by-4 mean). Track the bare `english error rate (WER)` line instead.
- Noise scored `1.000`, but appears required/useful rather than directly weighted for qualifiers.
- CV submitted but had `4 / 500` errors and score `0.000`; fix robustness/schema edge cases before improving model quality.
- AE submitted cleanly with score `0.051`; because AE is 40% of qualifiers, this is the highest-priority improvement target.

## Next priority

1. Submit ASR `vad-off-v2` (or whichever local-WER-≤-0.045 build lands first).
2. ASR LoRA fine-tune via [training/asr/](training/asr/) — target official 0.92-0.95. See [training/asr/ERROR_ANALYSIS.md](training/asr/ERROR_ANALYSIS.md).
3. Improve AE rule-based planner.
4. Improve NLP retrieval/chunking.
5. Fix CV errors before adding a detector.
