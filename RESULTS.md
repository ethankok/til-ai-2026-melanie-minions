# TIL-AI 2026 Submission Results

Team: `melanie-minions`
Last updated: 13 May 2026

## Latest submitted scores

```text
Task   Image                    Tag         Submitted             Errors        Score   Speed
NLP    melanie-minions-nlp      latest      12/05/2026 03:23:35   0 / 700       0.301   0.971
ASR    melanie-minions-asr      vad-off-v1  12/05/2026 20:00:21   0 / 400       0.938   0.859
CV     melanie-minions-cv       latest      12/05/2026 03:52:32   4 / 500       0.000   0.981
Noise  melanie-minions-noise    latest      12/05/2026 03:54:55   0 / 500       1.000   0.970
AE     melanie-minions-ae       latest      12/05/2026 04:20:34   0 / 30        0.051   0.856
```

## ASR submission history

```text
Tag        Submitted             Score   Speed   Local Eng-WER   Notes
v1         12/05 03:42           0.000   0.993   —               Empty-string baseline
norm-v1    12/05 16:23           0.877   0.864   0.0759          + digit verbalization + silence guard
vad-off-v1 12/05 20:00           0.938   0.859   0.0554          + VAD off + hallucination guards + ordinals + decimal-safe
```

## AE local validation history

```text
Variant     Date/time          Local novice score   Errors/action validity   Notes
baseline    12/05              0.051 official       0 / 30 official errors   Periodic-forward + periodic bomb baseline
planner-v1  13/05 10:31 +08    0.732 local Mac      0 invalid actions        Stateful belief map + objective/frontier BFS + LOS-safe tactical bombs
planner-v1  13/05 Workbench    0.697 local          til test completed       Built/tested with official Workbench Docker flow before submission
planner-v2  13/05 pending      —                    9/9 unit tests           Bomb timer 4→3 (matches env), bounded escape check, enemy soft threat, frontier scoring by unseen yield
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
0.20 * ASR 0.938 = 0.1876
0.20 * CV 0.000  = 0.0000
--------------------------------
Estimated weighted qualifier score = 0.2682
```

If `planner-v1` official AE roughly matches the Workbench local score (`0.697`), the raw weighted qualifier estimate becomes `0.5267` before any ASR/NLP/CV upgrades.

Using the observed ~75% score / 25% speed blend:

```text
AE   contribution = 0.1009
NLP  contribution = 0.0937
ASR  contribution = 0.1837
CV   contribution = 0.0491
--------------------------------
Estimated blended qualifier score = 0.4274
```

## Notes

- All 5 tasks have been submitted successfully at least once.
- **ASR is now a strong contributor**: `vad-off-v1` officially scored `0.938 / 0.859` (errors `0 / 400`). That's a **+6.1 absolute point jump** in accuracy from `norm-v1`'s `0.877` with speed barely changed (-0.005). The inference-only fixes (`vad_filter=False`, hallucination guards, spoken-form ordinals, coordinate-safe decimal regex, tighter silence guard) generalized cleanly to the hidden set — local-official WER gap collapsed from +0.047 absolute on `norm-v1` to **+0.007 absolute** on `vad-off-v1`.
- The local `1 - MER` number is a **scoring artifact** of the local manifest being English-only (three other language buckets contribute 0 to the divide-by-4 mean). Track the bare `english error rate (WER)` line instead.
- `vad-off-v2` REGRESSED to local WER 0.0604 after reversing the slang-prompt order (intent was to survive Whisper's truncation but reversing over-primed the decoder). Reverted; not submitted.
- Noise scored `1.000`, but appears required/useful rather than directly weighted for qualifiers.
- CV submitted but had `4 / 500` errors and score `0.000`; fix robustness/schema edge cases before improving model quality.
- AE baseline submitted cleanly with score `0.051`. `planner-v1` built and passed Workbench local testing at `0.697`; official submission/result pending as of this note.

## Next priority

1. **Record AE planner-v1 official result** once the Workbench submission finishes; keep baseline `0.051` until the official score appears.
2. **ASR LoRA fine-tune** ([training/asr/README.md](training/asr/README.md) quick-start; in progress). To hit `0.95+` we need WER ≤ 0.05 on the hidden set. Vad-off-v1 is at official WER 0.062 — LoRA realistically gets us 20-40% relative more.
3. **ASR re-export `int8_float16`** after FT lands for the speed score (~0.86 → 0.90+).
4. **NLP retrieval / chunking** upgrade — currently 0.301.
5. **CV** — fix the `4 / 500` errors first, then drop in a pretrained detector.
