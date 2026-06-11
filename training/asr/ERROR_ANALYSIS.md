# ASR Error Analysis

Last updated: 13 May 2026

## Progression of submitted results

```text
v1            faster-whisper distil-large-v3 zero-shot       official 0.839 / 0.864
norm-v1       + digit verbalization + silence guard          official 0.877 / 0.864   local Eng-WER 0.0759
vad-off-v1    + VAD off + hallucination guards + ordinals    official 0.938 / 0.859   local Eng-WER 0.0554
vad-off-v2    + slang prompt reversed                        REGRESSED local Eng-WER 0.0604, not submitted
ft-lora32-v1  + LoRA rank-32 decoder fine-tune (3 epochs)    official 0.957 / 0.849   local Eng-WER 0.0299 (leaky), val 0.04662
```

Local English WER trajectory (full 4110-clip test set): **0.113 → 0.076 → 0.055 → 0.030**.

**Local-official gap behavior changed on `ft-lora32-v1`** because the LoRA run
memorized ~90% of the local test set (the manifest has no held-out train/test
split — Option B). The cleaner generalization proxy is the 10% held-out val set
used for early-stopping inside training: val WER `0.04662` at step 1000.

| Tag | Local test WER | Local val WER (held-out) | Official WER | Gap (best proxy → official) |
|---|---:|---:|---:|---:|
| `norm-v1` | 0.0759 | — | 0.123 | **+0.047** (local→official) |
| `vad-off-v1` | 0.0554 | — | 0.062 | **+0.007** (local→official) |
| `ft-lora32-v1` | 0.0299 (leaky) | 0.04662 | ~0.043 | **−0.0036** (val→official) |

The `ft-lora32-v1` gap turned out **negative**: the official 400-clip set is
slightly easier than the held-out 409-clip val slice. Useful piece of
leaderboard intuition for future runs — held-out val WER is a *slight
over-estimate* of official WER on this dataset.

## Slang prompt ordering experiment (vad-off-v2)

Hypothesis: faster-whisper truncates `initial_prompt` to the last ~223 tokens; with ~200 proper nouns the prompt overflows, so reversing the list to put high-frequency terms at the END should make them survive truncation and help.

Result: **wrong direction**. Reversing the prompt regressed local English WER from 0.0554 → 0.0604 (≈+9% relative) with no other changes between v1 and v2. Wall-clock similar (44:29 → 46:22).

Likely cause: putting the most common in-world proper nouns (`cyanite`, `sarento`, `phyrexis`, `mewan`, ...) immediately before decode-start over-primes the decoder and causes false-positive hallucinations of those tokens on unrelated audio. v1's ordering left high-frequency terms in the truncated head and rarer terms at the end — a weaker, less biased prior.

Action taken: reverted `extract_slang.py` to write highest-frequency first. The `vad-off-v1` image (built before the reversal) is the recommended submission.

## Scoring artifact — confirmed

Local `test_asr.py` output:

```text
english error rate (WER): 0.0759       <-- the only meaningful number
chinese error rate (CER): 0.0000       <-- empty bucket
malay error rate (WER):   0.0000       <-- empty bucket
tamil error rate (WER):   0.0000       <-- empty bucket
1 - MER: 0.9810
```

The local manifest has only `english`-labeled samples; `jiwer.wer([], [])` returns 0 for the other three buckets; the scorer divides the sum by 4 unconditionally. So `MER = English_WER / 4` and `1 - MER ≈ 0.98` even when real WER is ~7%. The official 400-clip set must distribute samples across all four buckets, because official 0.877 ≈ `1 - English_WER` directly (suggesting English_WER ≈ 0.12 on the hidden set).

**Implication**: tune against the `english WER` line, not `1 - MER`. Local-official gap is roughly +0.05 absolute WER from local to official.

## Workbench facts (confirmed 12 May 2026)

- GPU: **Tesla T4** (16 GB). fp16 not bf16.
- Manifest: `/home/jupyter/novice/asr/asr.jsonl` has **4110 entries**. Same manifest is what `test_asr.py` evaluates against — there is no separate `train.jsonl`. Fine-tuning on it leaks into local eval; we trust only the official submission as a leaderboard signal. (Strategy: **Option B** — train on all, hold out 10% only for stability.)
- Schema per entry: `{"key", "audio": "sample_N.wav", "transcript", "language": "english"}`.
- Slang prompt was populated in the deployed `norm-v1` image (1 line, ~80 tokens).
- No accessible noise corpus on Workbench — augmentation falls back to SpecAugment + speed perturb only.
- 309 GB free disk.

## Observed error patterns

### 1. VAD truncation on long clips (DOMINANT issue — fixed in vad-off-v1)

The 30-worst-WER dump showed ~25 of 30 clips were 25-38s long with the same pattern: predictions missing leading words, mid-content chunks, or trailing content. Examples (from the norm-v1 dump):

| File              | Dur  | WER  | What was missing                                            |
|-------------------|------|------|-------------------------------------------------------------|
| sample_3776.wav   | 35s  | 0.60 | First sentence dropped                                      |
| sample_176.wav    | 25s  | 0.56 | Prediction starts mid-sentence                              |
| sample_596.wav    | 38s  | 0.40 | Last ~30% missing                                            |
| sample_2397.wav   | 35s  | 0.30 | 30+ words missing in the middle                              |

Root cause: faster-whisper's `BatchedInferencePipeline` requires VAD, and the Silero VAD with `min_silence_duration_ms=500` was clipping speech across utterance boundaries on long clips.

**Fix in vad-off-v1** ([asr/src/asr_manager_fasterwhisper.py](../../asr/src/asr_manager_fasterwhisper.py)):

- Dropped `BatchedInferencePipeline` (it only batches encoder segments within a single audio anyway, no real throughput gain for our per-clip loop).
- Switched to plain `model.transcribe(..., vad_filter=False)`.
- The audio-level `_is_probably_silence()` guard still catches empty-clip hallucinations.

### 2. Numeric formatting (handled in norm-v1, extended in vad-off-v1)

The scorer lowercases + strips punctuation but does not convert digits to words. Whisper outputs digits; references spell them out.

Examples:

```text
REF : seventy-two hours       PRED: 72 hours
REF : zero six hundred        PRED: 0600
REF : seven niner             PRED: 7-9-er
REF : the twenty-third        PRED: the 23rd
```

Fixes in `_digits_to_words()`:

- Plain integers, decimals, comma-formatted numbers, four-digit times, `niner` callsigns.
- **vad-off-v1 extensions**:
  - Spoken ordinals: `"23rd"` → `"twenty third"` (not `"twenty threerd"`), `"15th"` → `"fifteenth"`.
  - Coordinate-safe decimals: `"1.1.7"` no longer mangled into `"one point one.seven"`; dotted coordinates collapse to `"one one seven"` cleanly.

### 3. Silent / near-silent clips hallucinated text (handled in norm-v1, tightened in vad-off-v1)

```text
sample_924.wav  (0.6s)  REF:       PRED: Thank you.
sample_2373.wav (0.6s)  REF:       PRED: I
```

The original `rms < 2e-4 and peak < 2e-3` guard was too conservative for short noisy bursts. vad-off-v1 adds:

```python
if audio.size < self.TARGET_SR * 1.5 and rms < 1e-2:
    return True   # short + quiet = noise burst, not speech
```

Plus Whisper-side hallucination guards on the decode call: `no_speech_threshold=0.6`, `log_prob_threshold=-1.0`, `compression_ratio_threshold=2.4`, `temperature=0.0`.

### 4. Proper-noun substitutions (partially fixed by slang prompt + reversal)

Almost every long-clip WER point comes from substitutions of in-world vocabulary:

| Reference     | Norm-v1 PRED  |
|---------------|---------------|
| Sarento       | Sorrento      |
| Cyanite       | cyanide       |
| Phyrexis      | Pyrex's       |
| New Mewan     | "new Mee-one" |
| Kestrelian    | Castilian     |
| Acolyte       | accolite      |
| Belford Straits | Belford Streets |

`extract_slang.py` mines exactly these from the NLP corpus. The current defaults are `--top-k 200 --min-count 2`. Top-frequency tokens include `cyanite phyrexis sarento mewan kestrelian floodwall ashcastle fullwalker ...`.

**Slang prompt ordering** (post-experiment): the original ordering (highest-frequency first) works better than the reversed ordering, even though the prompt overflows the ~223-token retention budget. See the experiment write-up at the top of this file.

## Current inference settings (ft-lora32-v1)

Inference path is unchanged from `vad-off-v1`; only the model weights are
different (LoRA merged into base). Slang-prompt ordering is highest-frequency
first (the reversal experiment regressed; see above).

```python
self.model.transcribe(
    audio,
    language="en",
    task="transcribe",
    beam_size=1,
    vad_filter=False,                       # changed in vad-off-v1
    condition_on_previous_text=False,
    initial_prompt=self.initial_prompt,     # 200-token slang prompt, high-freq first
    without_timestamps=True,                # small speed win
    temperature=0.0,                        # no temp fallback retries
    compression_ratio_threshold=2.4,        # repetition guard
    log_prob_threshold=-1.0,                # low-confidence guard
    no_speech_threshold=0.6,                # silence hallucination guard
)
```

Audio-level silence pre-check ([_is_probably_silence](../../asr/src/asr_manager_fasterwhisper.py)) runs before the model call; output post-processing runs `_digits_to_words()` on the joined segment text.

## Path to score > 0.95 and speed > 0.9

```text
Official target:           0.95+ accuracy AND 0.90+ speed
Current best (submitted):  0.957 / 0.849   (ft-lora32-v1)
Distance to target:        accuracy DONE (+0.007 margin); speed still −0.051
```

Priority order now:

1. **DONE — LoRA rank-32 fine-tune** ([training/asr/README.md](README.md)). 3-epoch run pushed local val WER 0.0554 → 0.0466 and landed official 0.957. Crossed the accuracy target. `load_best_model_at_end=True` rolled back to the step-1000 checkpoint (best val WER).
2. **NEXT — Re-export `--quantization int8_float16`** of the same LoRA-merged checkpoint. Expected: speed 0.849 → 0.90+, accuracy delta ≤ 0.005 → blended ~0.952 / 0.90+.
3. **Beam=2** only if step 2 has speed margin (≥0.92) and accuracy still has headroom below 0.95.
4. **Rank-64 escalation** (5 epochs, lr 5e-5, alpha 128) is deferred — would likely buy another 0.005–0.010 absolute on local val, but `ft-lora32-v1` already crossed target and another 7-hour training run isn't worth it unless step 2 regresses accuracy.
5. **Larger model / ensemble** (e.g., add full whisper-large-v3 in parallel and ROVER-vote) only if 1-4 still fall short of 0.95 — which is unlikely now.

## Workbench test commands

```bash
cd /home/jupyter/til
git pull origin main
export TIL_FOLDER=/home/jupyter/til   # one-time, or set in .bash_profile

# Re-extract slang only when corpus or top-k changes.
python training/asr/extract_slang.py \
    --nlp-dir /home/jupyter/novice/nlp \
    --out asr/models/slang_prompt.txt
tail -c 300 asr/models/slang_prompt.txt   # should end with high-freq in-world nouns

til build asr <tag>
til test asr <tag>
```

Submit only if local `english WER` improves over the last shipped tag:

```bash
til submit asr <tag>
```

## Fine-tuning path

See [README.md](README.md) for the end-to-end commands. With Option B locked in, train on all 4110 entries with a deterministic 10% held-out solely for early stopping; treat the official submission as the only real validator.

```bash
python training/asr/extract_slang.py \
    --nlp-dir /home/jupyter/novice/nlp \
    --out asr/models/slang_prompt.txt

python training/asr/prepare_data.py \
    --data-dir /home/jupyter/novice/asr \
    --slang-file asr/models/slang_prompt.txt \
    --out-dir training/asr/data \
    --slang-multiplier 2

python training/asr/train_distil_whisper.py \
    --data-dir training/asr/data \
    --output-dir training/asr/runs/distil-en-lora32-v1 \
    --epochs 3 --per-device-batch-size 16 \
    --lora-rank 32 --lora-alpha 64 --lr 1e-4

python training/asr/export_ct2.py \
    --adapter-dir training/asr/runs/distil-en-lora32-v1/best \
    --slang-file asr/models/slang_prompt.txt \
    --output-dir asr/models \
    --quantization float16

til build asr ft-lora32-v1
til test asr ft-lora32-v1
til submit asr ft-lora32-v1
```

If the moderate run plateaus, escalate to rank 64 / 5 epochs / lr 5e-5.

Speed re-export after FT:

```bash
python training/asr/export_ct2.py \
    --adapter-dir training/asr/runs/distil-en-lora32-v1/best \
    --slang-file asr/models/slang_prompt.txt \
    --output-dir asr/models \
    --quantization int8_float16

til build asr ft-lora32-int8f16
til test asr ft-lora32-int8f16
```
