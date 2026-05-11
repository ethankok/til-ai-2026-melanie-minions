# TIL-AI 2026 Submission Results

Team: `melanie-minions`
Last updated: 12 May 2026

## Latest submitted scores

```text
Task   Image                    Tag     Submitted             Errors        Score   Speed
NLP    melanie-minions-nlp      latest  12/05/2026 03:23:35   0 / 700       0.301   0.971
ASR    melanie-minions-asr      latest  12/05/2026 03:42:19   0 / 400       0.000   0.993
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

Using the latest submitted task scores:

```text
0.40 * AE 0.051  = 0.0204
0.20 * NLP 0.301 = 0.0602
0.20 * ASR 0.000 = 0.0000
0.20 * CV 0.000  = 0.0000
--------------------------------
Estimated weighted qualifier score = 0.0806
```

## Notes

- All 5 tasks have been submitted successfully at least once.
- NLP is currently the only scored 20% task contributing non-zero points.
- Noise scored `1.000`, but appears required/useful rather than directly weighted for qualifiers.
- CV submitted but had `4 / 500` errors and score `0.000`; fix robustness/schema edge cases before improving model quality.
- ASR submitted cleanly but score is `0.000`; expected if it is still returning blank transcripts. **Status (post-submission): rebuilt locally — faster-whisper distil-large-v3 + slang prompt mined from NLP corpus; LoRA fine-tune pipeline in `training/asr/`. Not yet resubmitted (need to run training + export on Workbench first).**
- AE submitted cleanly with score `0.051`; because AE is 40% of qualifiers, this is the highest-priority improvement target.

## Next priority

1. Run ASR training pipeline on Workbench (`training/asr/`) and resubmit — target WER ≤ 0.10 on novice dev.
2. Improve AE rule-based planner.
3. Improve NLP retrieval/chunking.
4. Fix CV errors before adding a detector.
