# ASR Error Analysis

Last updated: 12 May 2026

## Current best official submission

```text
Image: melanie-minions-asr:norm-v1
Submitted: 12/05/2026 16:23:35
Errors: 0 / 400
Score: 0.877
Speed: 0.864
```

Local pre-submission test on the Workbench scored higher:

```text
1 - MER: 0.9718490510343567
english WER: 0.1126
1028 local clips in 47m50s
```

Official hidden score is lower than local, so prioritize fixes that generalize rather than overfitting the local manifest.

## Observed error patterns

### 1. Numeric formatting causes avoidable WER

The scorer lowercases and removes punctuation, but it does not convert digits into spoken words. Faster-Whisper often recognizes the meaning correctly but returns digits, which are counted as word errors.

Examples:

```text
REF : seventy-two hours
PRED: 72 hours

REF : zero six hundred
PRED: 0600 / 0,600

REF : seven niner
PRED: 7-9-er
```

Fix added in `asr/src/asr_manager.py`:

- `_digits_to_words()` verbalizes numerals before returning predictions.
- Handles plain integers, decimals, comma-formatted numbers, four-digit times, and `niner` callsigns.

Commit:

```text
bc34715 feat(asr): normalize numeric transcripts
```

### 2. Silent / near-silent clips hallucinate text

Whisper sometimes emits short hallucinations on silence.

Examples:

```text
REF :
PRED: Thank you.

REF :
PRED: I
```

Fix added in `asr/src/asr_manager.py`:

- `_is_probably_silence()` returns blank output for very low RMS + low peak audio.
- Goal: reduce hallucinated non-empty transcripts on empty references.

Risk: threshold that is too aggressive can blank very quiet real speech. Current threshold is conservative:

```python
rms < 2e-4 and peak < 2e-3
```

### 3. Long clips can be truncated

Some bad local examples looked like the model captured the start but missed later content. Likely causes:

- VAD cuts speech too aggressively.
- Long/noisy clips exceed the model's comfortable context.
- Greedy decoding (`beam_size=1`) chooses a locally plausible but incomplete transcript.

Current inference settings:

```python
beam_size=1
vad_filter=True
vad_parameters={"min_silence_duration_ms": 500}
condition_on_previous_text=False
language="en"
```

Next A/B tests:

1. `vad_filter=False`
2. weaker VAD, e.g. larger `min_silence_duration_ms`
3. `beam_size=2` or `3` only if speed remains above target

### 4. Domain slang / rare names still need data adaptation

The prompt helps, but hidden official data may include vocabulary not covered by the current slang prompt.

Current prompt source:

```bash
python training/asr/extract_slang.py \
  --nlp-dir /home/jupyter/novice/nlp \
  --out asr/models/slang_prompt.txt \
  --top-k 80
```

Potential improvement:

```bash
--top-k 120
```

Then rebuild and test.

## Path to score > 0.95 and speed > 0.9

Current official:

```text
Score: 0.839  -> error approx 0.161
Speed: 0.864
```

Target:

```text
Score: 0.950  -> error <= 0.050
Speed: 0.900
```

Required improvement:

```text
~69% relative error reduction
~4.2% speed improvement
```

Priority order:

1. Run numeric/silence post-processing patch and compare against local baseline.
2. A/B VAD off vs current VAD.
3. If still below target, fine-tune Distil-Whisper with LoRA.
4. For speed, test CT2 `int8_float16` export only after accuracy improves.

## Workbench test commands

```bash
cd /home/jupyter/til
git pull origin main
export TIL_FOLDER=/home/jupyter/til

til build asr norm-v1
til test asr norm-v1
```

Compare against previous local baseline:

```text
1 - MER: 0.9718490510343567
```

Submit only if local score is same or better, or if the changes clearly target hidden official failures:

```bash
til submit asr norm-v1
```

## Fine-tuning path if inference fixes are insufficient

```bash
python training/asr/extract_slang.py \
  --nlp-dir /home/jupyter/novice/nlp \
  --out asr/models/slang_prompt.txt \
  --top-k 120

python training/asr/prepare_data.py \
  --data-dir /home/jupyter/novice/asr \
  --slang-file asr/models/slang_prompt.txt \
  --out-dir training/asr/data \
  --slang-multiplier 3

python training/asr/train_distil_whisper.py \
  --data-dir training/asr/data \
  --output-dir training/asr/runs/distil-en-lora64-v1 \
  --epochs 5 \
  --per-device-batch-size 16 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --lr 5e-5 \
  --eval-steps 250 \
  --save-steps 250

python training/asr/export_ct2.py \
  --adapter-dir training/asr/runs/distil-en-lora64-v1/best \
  --slang-file asr/models/slang_prompt.txt \
  --output-dir asr/models \
  --quantization float16

til build asr ft-lora64-v1
til test asr ft-lora64-v1
```

If OOM:

```bash
--per-device-batch-size 8 --grad-accum 2
```

Speed A/B after accuracy improves:

```bash
python training/asr/export_ct2.py \
  --adapter-dir training/asr/runs/distil-en-lora64-v1/best \
  --slang-file asr/models/slang_prompt.txt \
  --output-dir asr/models \
  --quantization int8_float16

til build asr ft-lora64-int8f16
til test asr ft-lora64-int8f16
```
