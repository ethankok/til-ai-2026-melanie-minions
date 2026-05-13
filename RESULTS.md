# TIL-AI 2026 Submission Results

Team: `melanie-minions`
Last updated: 13 May 2026

## Latest submitted scores

```text
Task   Image                    Tag         Submitted             Errors        Score   Speed
NLP    melanie-minions-nlp      latest      12/05/2026 03:23:35   0 / 700       0.301   0.971
ASR    melanie-minions-asr      ft-lora32-v1 13/05/2026 11:22:30  0 / 400       0.957   0.849
CV     melanie-minions-cv       latest      12/05/2026 03:52:32   4 / 500       0.000   0.981
Noise  melanie-minions-noise    latest      12/05/2026 03:54:55   0 / 500       1.000   0.970
AE     melanie-minions-ae       latest      12/05/2026 04:20:34   0 / 30        0.051   0.856
```

## ASR submission history

```text
Tag           Submitted          Score   Speed   Local Eng-WER   Notes
v1            12/05 03:42        0.000   0.993   —               Empty-string baseline
norm-v1       12/05 16:23        0.877   0.864   0.0759          + digit verbalization + silence guard
vad-off-v1    12/05 20:00        0.938   0.859   0.0554          + VAD off + hallucination guards + ordinals + decimal-safe
ft-lora32-v1  13/05 11:22        0.957   0.849   0.0299*         + LoRA rank-32 decoder fine-tune (3 epochs, lr 1e-4)
```

*`ft-lora32-v1` local Eng-WER is **leaky** (trained on 90% of the 4110-clip test
set; the bare 0.0299 includes memorization). The held-out 10% val WER at step
1000 was **0.04662** — that's the cleaner proxy. Official WER 0.043 means the
generalization gap turned out *negative* (val 0.04662 → official 0.043), i.e.
distil-large-v3 + LoRA generalized **better** than leaky-val suggested.

## AE local validation history

```text
Variant     Date/time          Local novice score   Errors/action validity   Notes
baseline    12/05              0.051 official       0 / 30 official errors   Periodic-forward + periodic bomb baseline
planner-v1  13/05 10:31 +08    0.732 local Mac      0 invalid actions        Stateful belief map + objective/frontier BFS + LOS-safe tactical bombs
planner-v1  13/05 Workbench    0.697 local          til test completed       Built/tested with official Workbench Docker flow before submission
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
0.20 * ASR 0.957 = 0.1914
0.20 * CV 0.000  = 0.0000
--------------------------------
Estimated weighted qualifier score = 0.2720
```

If `planner-v1` official AE roughly matches the Workbench local score (`0.697`), the raw weighted qualifier estimate becomes `0.5305` before any NLP/CV upgrades.

Using the observed ~75% score / 25% speed blend:

```text
AE   contribution = 0.1009
NLP  contribution = 0.0937
ASR  contribution = 0.1858
CV   contribution = 0.0491
--------------------------------
Estimated blended qualifier score = 0.4295
```

## Notes

- All 5 tasks have been submitted successfully at least once.
- **ASR crossed the 0.95 accuracy target**: `ft-lora32-v1` officially scored `0.957 / 0.849` (errors `0 / 400`). That's **+0.019 absolute accuracy** over `vad-off-v1`, achieved by a single 3-epoch LoRA-rank-32 decoder fine-tune of `distil-whisper/distil-large-v3` on the full 4110-clip novice manifest. Speed dipped by `0.010` (CT2 file size noise; recoverable via int8_float16 re-export, see Next priority). Generalization gap turned out **negative**: held-out val WER `0.04662` → official WER `~0.043`, i.e. the official 400-clip distribution is slightly easier than the local held-out slice — a useful piece of leaderboard intuition for future runs.
- **`vad-off-v1` (the prior peak)**: inference-only fixes (`vad_filter=False`, hallucination guards, spoken-form ordinals, coordinate-safe decimal regex, tighter silence guard) took accuracy from `norm-v1`'s `0.877` to `0.938` with speed barely changed (-0.005). Local-official WER gap on that run was +0.007 absolute. Those fixes stayed in `ft-lora32-v1` and compounded with the LoRA gains.
- The local `1 - MER` number is a **scoring artifact** of the local manifest being English-only (three other language buckets contribute 0 to the divide-by-4 mean). Track the bare `english error rate (WER)` line instead.
- `vad-off-v2` REGRESSED to local WER 0.0604 after reversing the slang-prompt order (intent was to survive Whisper's truncation but reversing over-primed the decoder). Reverted; not submitted.
- Noise scored `1.000`, but appears required/useful rather than directly weighted for qualifiers.
- CV submitted but had `4 / 500` errors and score `0.000`; fix robustness/schema edge cases before improving model quality.
- AE baseline submitted cleanly with score `0.051`. `planner-v1` built and passed Workbench local testing at `0.697`; official submission/result pending as of this note.

## Next priority

1. **Record AE planner-v1 official result** once the Workbench submission finishes; keep baseline `0.051` until the official score appears.
2. **ASR re-export `int8_float16`** — same LoRA-merged checkpoint, no retrain. Target: speed `0.849 → 0.90+` with accuracy delta `≤ 0.005`. Tag `ft-lora32-int8f16`.
3. **NLP retrieval / chunking** upgrade — currently 0.301.
4. **CV** — fix the `4 / 500` errors first, then drop in a pretrained detector.
5. **ASR rank-64 escalation** is deferred — `ft-lora32-v1` already crosses 0.95, so the marginal lift isn't worth the extra training time unless int8 quantization unexpectedly regresses accuracy.
