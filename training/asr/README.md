# ASR training pipeline (Workbench-only)

Three practical ASR paths live here:

* **Parakeet unified (NeMo)** — current staged A/B. Downloads
  `nvidia/parakeet-unified-en-0.6b` and bakes the `.nemo` into the ASR image.
* **Parakeet-TDT (NeMo)** — current shipped cloud high. Fine-tunes
  `nvidia/parakeet-tdt-0.6b-v2` and exports the `.nemo` directly when needed.
* **distil-whisper LoRA** — legacy path, kept as fallback while Parakeet
  experiments run. Fine-tunes `distil-whisper/distil-large-v3` with PEFT
  LoRA, then exports to CTranslate2.

Nothing in this folder ships in the eval Docker image — it's all training-side
tooling.

**Current shipped artifact**: `nemo-zs` — Parakeet-TDT-0.6B-v2 zero-shot,
official **0.956 / 0.946** (14 May 20:33 SGT). `nemo-zs-v2` re-exported with
`cuda-python>=12.3` enabled CUDA-graph fast path locally (37:28 → 34:42
wall clock) but cloud speed is identical at 0.946 — the bottleneck is no
longer the TDT decoder, it's HTTP / audio I/O / Python overhead. On 22 May,
`parakeet-unified-zs` was staged as the next low-risk accuracy A/B before any
fine-tune.

## Parakeet unified quick start: zero-shot A/B (Workbench)

This is the current recommended ASR action. It swaps only the NeMo checkpoint,
from `nvidia/parakeet-tdt-0.6b-v2` to `nvidia/parakeet-unified-en-0.6b`.

```bash
cd /home/jupyter/til
export TIL_FOLDER=/home/jupyter/til

# Host side only needs this to fetch the .nemo. The Docker image installs NeMo.
python -m pip install -U huggingface_hub

python training/asr/extract_slang.py \
    --nlp-dir /home/jupyter/novice/nlp \
    --out asr/models/slang_prompt.txt

python training/asr/download_models_nemo.py \
    --model nvidia/parakeet-unified-en-0.6b \
    --out asr/models

til build asr parakeet-unified-zs
til test asr parakeet-unified-zs
til submit asr parakeet-unified-zs   # only if local Eng-WER beats 0.0429, or for one protected cloud probe
```

The committed `asr/Dockerfile` now points at this model via
`ASR_NEMO_MODEL=parakeet-unified-en-0.6b.nemo`, so `til build asr
parakeet-unified-zs` is enough after the weights are downloaded. To rebuild
the old TDT-v2 path, download `nvidia/parakeet-tdt-0.6b-v2` and override
`ASR_NEMO_MODEL=parakeet-tdt-0.6b-v2.nemo`.

## Parakeet quick start: fine-tune on T4 (Workbench)

The defaults in `train_parakeet.py` are tuned for T4 (16 GB) at fp16 and still
point at the proven TDT-v2 checkpoint. Revisit this only if the unified
zero-shot A/B is flat but ASR remains worth more time.

```bash
cd /home/jupyter/til
export TIL_FOLDER=/home/jupyter/til

# 0. Install training deps once.
pip install -r asr/requirements-nemo.txt

# 1. Refresh the slang prompt (idempotent).
python training/asr/extract_slang.py \
    --nlp-dir /home/jupyter/novice/nlp \
    --out asr/models/slang_prompt.txt

# 2. Make sure the base .nemo is on disk (no-op if it already is).
python training/asr/download_models_nemo.py \
    --model nvidia/parakeet-tdt-0.6b-v2 \
    --out asr/models

# 3. Build NeMo manifests with slang oversampling.
python training/asr/prepare_data_nemo.py \
    --data-dir /home/jupyter/novice/asr \
    --slang-file asr/models/slang_prompt.txt \
    --out-dir training/asr/data_nemo \
    --slang-multiplier 2

# 4. Fine-tune. Encoder frozen, decoder + joint trainable. ~3-4 hr on T4.
#    Defaults are T4-safe (batch 4, grad-accum 4, max-duration 30s) — the
#    RNNT loss is O(B*T*U*V) so larger batches OOM. Effective batch is 16.
python training/asr/train_parakeet.py \
    --data-dir training/asr/data_nemo \
    --base-model asr/models/parakeet-tdt-0.6b-v2.nemo \
    --output-dir training/asr/runs/parakeet-ft-v1 \
    --epochs 5 --lr 5e-5

# 5. Stage the fine-tuned .nemo where the docker build expects it.
python training/asr/export_parakeet.py \
    --trained-nemo training/asr/runs/parakeet-ft-v1/best.nemo \
    --output-dir asr/models \
    --slang-file asr/models/slang_prompt.txt

# 6. Build, test, submit.
docker build -f asr/Dockerfile.nemo -t melanie-minions-asr:parakeet-ft-v1 ./asr
til test asr parakeet-ft-v1
til submit asr parakeet-ft-v1   # only if local Eng-WER beats nemo-zs's 0.0429
```

If the val_wer plateaus before epoch 5, ModelCheckpoint + EarlyStopping
inside `train_parakeet.py` will roll back to the best step automatically.
If you see val_wer still falling at epoch 5, rerun with `--epochs 8` and a
slightly tighter LR (`--lr 3e-5`).

To unfreeze the encoder for one low-LR pass:
```bash
python training/asr/train_parakeet.py \
    --no-freeze-encoder \
    --lr 1e-5 --epochs 2 \
    [other args]
```
This burns more T4 wall clock but can reclaim a small accuracy delta if
decoder-only FT plateaus above the cloud-eval gate.

## Whisper LoRA quick start (legacy fallback)

The whisper pipeline is preserved as a fallback in case Parakeet
experiments regress badly enough that we need to roll back. Skip this
section unless you're rolling back.

## Design at a glance (whisper)

- **Base model**: `distil-whisper/distil-large-v3` — English-only distillation of large-v3. Same encoder quality, ~6× faster decoder. Right call for the Novice (English-only) track; avoids paying for a multilingual decoder we don't need.
- **Fine-tune**: LoRA (rank 32, alpha 64) on decoder attention projections only. Encoder frozen (distil-v3 inherits a strong large-v3 encoder; budget should go to the decoder, which distillation shrank).
- **Slang prior**: tokens rare in standard English but frequent in the NLP corpus are mined into a single-line prompt baked next to the model weights. Passed as `initial_prompt=` to bias decoding; same set is used to oversample slang-containing audio clips during fine-tune. **Ordering**: highest-frequency first. Reversing the list (so high-freq lands at the prompt end, immediately before decode-start) was tested in `vad-off-v2` and regressed local WER from 0.055 → 0.060 — likely because heavy decoder priming on common in-world nouns caused false-positive hallucinations. See [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).
- **Augmentation** (noisy-audio robustness): SpecAugment, optional online noise mixing at SNR 5–20 dB (requires `--noise-dir`), optional ×0.9/×1.1 speed perturbation.
- **Inference**: faster-whisper plain `model.transcribe()` with CT2 float16 (CPU fallback int8), greedy beam, `vad_filter=False` (VAD was eating speech on long clips), `condition_on_previous_text=False`, `language="en"` forced, `without_timestamps=True`, hallucination guards (`no_speech_threshold=0.6`, `log_prob_threshold=-1.0`, `compression_ratio_threshold=2.4`, `temperature=0.0`). Audio-level RMS/peak silence guard with a sub-1.5s aggressive threshold handles the empty-clip hallucination case. Output post-processed by `_digits_to_words()` (ints, decimals, comma-thousands, military times, four-digit codes, niner callsigns, spoken ordinals, coordinate-safe decimals). Server passes the whole HTTP batch to `manager.asr_batch(list[bytes])`.

See [../../asr/src/asr_manager.py](../../asr/src/asr_manager.py) for the inference flags and [../../asr/src/asr_server.py](../../asr/src/asr_server.py) for the batched server path.

## Prereqs

```bash
pip install -r requirements-dev.txt
python -c "import nltk; nltk.download('words')"   # if wordfreq unavailable
```

GPU is required for training. Bf16 used on Ampere+; fp16 fallback on T4.

## Smoke-test the serving path without training

`til build asr` will fail with `COPY failed: file not found` until `asr/models/`
contains a faster-whisper / CT2 model directory. To bring up the container
end-to-end with **no fine-tune yet** (just to validate serving):

```bash
# One-time: produce a zero-shot CT2 export of the un-fine-tuned base.
pip install ctranslate2
ct2-transformers-converter \
    --model distil-whisper/distil-large-v3 \
    --output_dir asr/models \
    --quantization float16 \
    --copy_files tokenizer.json preprocessor_config.json generation_config.json
echo "" > asr/models/slang_prompt.txt   # empty prompt is fine

til build asr && til test asr
```

This proves the manager, server, and Docker plumbing work before spending GPU
time on fine-tuning.

## Quick start: LoRA on Tesla T4 (Workbench)

The defaults in `train_distil_whisper.py` are tuned for T4 (16 GB). On
Workbench, after `git pull origin main`:

```bash
cd /home/jupyter/til
export TIL_FOLDER=/home/jupyter/til

# 0. Install training deps once (skip if already done).
pip install -r requirements-dev.txt

# 1. Skip if slang_prompt.txt is already current.
python training/asr/extract_slang.py \
    --nlp-dir /home/jupyter/novice/nlp \
    --out asr/models/slang_prompt.txt

# 2. Build the HF dataset (90/10 stratified split + slang oversampling).
python training/asr/prepare_data.py \
    --data-dir /home/jupyter/novice/asr \
    --slang-file asr/models/slang_prompt.txt \
    --out-dir training/asr/data

# 3. LoRA fine-tune. ~2-3 hrs on T4 for 3 epochs.
python training/asr/train_distil_whisper.py \
    --data-dir training/asr/data \
    --output-dir training/asr/runs/distil-en-lora32-v1 \
    --epochs 3 \
    --lora-rank 32 --lora-alpha 64 --lr 1e-4

# 4. Merge LoRA into base + convert to CT2 + copy slang prompt next door.
python training/asr/export_ct2.py \
    --adapter-dir training/asr/runs/distil-en-lora32-v1/best \
    --slang-file asr/models/slang_prompt.txt \
    --output-dir asr/models \
    --quantization float16

# 5. Build, local-test, submit.
til build asr ft-lora32-v1
til test asr ft-lora32-v1     # ft-lora32-v1 shipped at local 0.0299 (leaky), official 0.957
til submit asr ft-lora32-v1
```

If the moderate run plateaus or you see val WER still falling at epoch 3,
relaunch with the heavier settings:

```bash
--epochs 5 --lora-rank 64 --lora-alpha 128 --lr 5e-5
```

### Known gotchas on T4 (all already patched as of 13 May 2026)

The seven errors below all fired in sequence across the first two Workbench
runs and are now fixed in the committed scripts. Listed for posterity so a
future contributor recognizes them if they reappear. The first four hit on the
12 May initial run; the last three hit on the 13 May resumed run after the
Workbench env got bumped to `transformers 5.8.0`.

Initial four (12 May):

- **`RuntimeError: element 0 of tensors does not require grad`** — PEFT +
  gradient checkpointing on a frozen-encoder setup. Fixed by calling
  `model.gradient_checkpointing_enable()` + `model.enable_input_require_grads()`
  BEFORE `get_peft_model(...)`.
- **`TypeError: ... unexpected keyword argument 'evaluation_strategy'`** —
  `transformers >= 4.46` renamed it to `eval_strategy`. Fixed in the kwargs to
  `Seq2SeqTrainingArguments`.
- **`TypeError: Seq2SeqTrainer.__init__() got an unexpected keyword argument 'tokenizer'`** —
  same transformers rename family: `tokenizer=` → `processing_class=`. Fixed.
- **`RuntimeError: Could not load libtorchcodec ... libavutil.so.* not found`** —
  HF `datasets` Audio decoding defaults to torchcodec, which needs FFmpeg
  system libraries that aren't in the Workbench base image. Fixed by casting
  the audio column to `Audio(decode=False)` and loading via soundfile +
  librosa in the collator (same path the inference container uses). No
  system FFmpeg install required.

Three more from the 13 May resumed run (Workbench env was bumped to
`transformers 5.8.0` overnight):

- **`TypeError: WhisperDecoder ... got multiple values for keyword argument 'input_ids'`**
  on step 0 — transformers 5.x's Whisper forward signature is stricter and PEFT
  was double-passing `input_ids`. **Workaround**: pin transformers to 4.x on
  the user site:
  `pip install --user 'transformers>=4.46,<5.0'`
  (resolves to 4.57.x as of writing). Verify with
  `python -c "import transformers; print(transformers.__version__)"`.
- **`TypeError: WhisperForConditionalGeneration.forward() got an unexpected keyword argument 'input_ids'`**
  on step 0 — `PeftModelForSeq2SeqLM.forward` (selected by
  `task_type=TaskType.SEQ_2_SEQ_LM`) explicitly passes `input_ids=None` down
  to the base model, which transformers 4.57's Whisper rejects (audio path
  uses `input_features`). **Fix**: drop `task_type` from `LoraConfig` — plain
  `LoraModel` passes kwargs through unchanged. Patched in
  `train_distil_whisper.py`.
- **`RuntimeError: output directory asr/models already exists, use --force to override`** —
  `ct2-transformers-converter` checks for *directory existence*, not just
  emptiness, so the wipe loop in `export_ct2.py` wasn't enough on re-export.
  **Fix**: pass `--force` to the converter. Patched in `export_ct2.py`.
- **`WARN: slang file asr/models/slang_prompt.txt not found`** on re-export —
  silent (only a WARN, build succeeds), but ships a container WITHOUT the
  slang prompt, which silently degrades decoding on in-world vocabulary
  (cyanite, sarento, phyrexis, …). Happens when `--slang-file` points inside
  `--output-dir`: the wipe loop deletes the slang file before the copy-back
  step runs. **Fix**: `export_ct2.py` now reads the slang file into memory
  BEFORE wiping, then writes it back after the convert step. If you see this
  WARN, **abort the build/test/submit cycle** and rebuild — submitting
  without the slang prompt regresses official WER.

Still possible during a longer run:

- **OOM at batch 8**: drop to `--per-device-batch-size 4 --grad-accum 4`. The
  effective batch stays the same; per-step compute halves.
- **Per-step wall-clock**: T4 takes ~5-7s per optimizer step at batch 8 with
  grad-accum 2. Total ~1.5-2 hrs for 1164 steps (3 epochs × 6196 train clips ÷
  effective batch 16).

## End-to-end

```bash
# 1. Mine in-world slang from the NLP corpus. Defaults: --top-k 200 --min-count 2.
python training/asr/extract_slang.py \
    --nlp-dir /home/jupyter/novice/nlp \
    --out asr/models/slang_prompt.txt

# 2. Build the dataset with slang oversampling.
python training/asr/prepare_data.py \
    --data-dir /home/jupyter/novice/asr \
    --slang-file asr/models/slang_prompt.txt \
    --out-dir training/asr/data \
    --slang-multiplier 2

# 3. LoRA fine-tune. (Adjust --noise-dir if a noise corpus is available.)
python training/asr/train_distil_whisper.py \
    --data-dir training/asr/data \
    --output-dir training/asr/runs/distil-en-v1 \
    --epochs 3 \
    --per-device-batch-size 16

# 4. Merge LoRA + convert to CT2 + drop slang prompt next to the weights.
python training/asr/export_ct2.py \
    --adapter-dir training/asr/runs/distil-en-v1/best \
    --slang-file asr/models/slang_prompt.txt \
    --output-dir asr/models \
    --quantization float16

# 5. Optional: push artifacts to GCS so other Workbench instances can re-pull.
gsutil -m cp -r asr/models gs://<your-team-bucket>/asr-models

# 6. Build and submit the container.
til build asr
til test asr
til submit asr
```

## Runtime config

`ASRManager` auto-picks device and precision:

- CUDA visible to ctranslate2 → `(cuda, float16)` (expected eval path).
- Otherwise → `(cpu, int8)`. Container stays up, accuracy intact, speed score will collapse. The chosen pair is printed at startup so a mystery-slow eval is immediately diagnosable.

Override with env vars in `asr/Dockerfile` if needed:

```dockerfile
ENV ASR_DEVICE=cuda
ENV ASR_COMPUTE_TYPE=float16          # or int8_float16 for more speed
ENV ASR_MODELS_DIR=/workspace/models/asr
ENV ASR_SLANG_PROMPT_PATH=/workspace/models/asr/slang_prompt.txt
```

## Notes

- `asr.jsonl` (4110 entries) is the **only** ASR manifest on Workbench, at
  `/home/jupyter/novice/asr/asr.jsonl`. The same file is what `test_asr.py`
  evaluates against — there is no separate `train.jsonl`. Fine-tuning on it
  leaks into local eval, so post-FT local WER will be inflated by
  memorization. **Treat the official submission as the only real validator.**
  (Strategy: **Option B** — train on all 4110, hold out 10% only for early
  stopping / overfitting detection.)
- WER on the held-out 10% val split is what `train_distil_whisper.py` reports;
  it uses the same `jiwer` transforms as the official scorer
  ([test/test_asr.py:24-33](../../test/test_asr.py)).
- The official scorer divides per-language error rates by 4 unconditionally.
  Local manifest has only `english`-labeled samples, so the three empty
  buckets return WER=0 from `jiwer` and the local `1 - MER` is inflated 4×
  vs. real WER. **Track the bare `english error rate (WER)` line**, not
  `1 - MER`. See [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) for details.
- Don't commit `asr/models/` or `training/asr/runs/` — they're in
  `.gitignore`.

## Iteration knobs (current state: `nemo-zs-v2` @ 0.956 / 0.946)

Priority order, top is the next action:

1. **Parakeet decoder-only fine-tune** — `train_parakeet.py` with the default
   args (encoder frozen, lr 5e-5, 5 epochs, batch 8 × grad-accum 2). Val WER
   target: ≤ 0.035 (from current zero-shot 0.0429). Wall-clock: ~3-4 hr on T4.
   Expected official accuracy bump: 0.956 → 0.965-0.975.
2. **Slang word-boost via NeMo `boosting_words`** — bump `nemo_toolkit[asr]`
   to 2.1+ (or higher) and re-test. Currently `_configure_biasing` in
   `asr_manager_nemo.py` falls back to "no biasing" because NeMo 2.0.0 doesn't
   expose the boost API. Expected impact: +0.003-0.010 absolute on slang-heavy
   clips. Cost: low (one config bump + re-test).
3. **Encoder unfreeze for one low-LR pass** — only after step 1 plateaus.
   `python training/asr/train_parakeet.py --no-freeze-encoder --lr 1e-5
   --epochs 2`. Adds ~2-3 hr T4. Expected: +0.002-0.005 absolute beyond
   decoder-only FT.

Deferred (don't pursue unless steps 1-3 still leave us below 0.97):

- **Larger Parakeet variant** — `parakeet-tdt-1.1b-v2` exists but is ~2× the
  decoder cost. Speed score would drop below 0.92, likely a blended
  regression for an accuracy gain we can probably get cheaper.
- **Whisper LoRA rank-64 escalation** — only relevant if we roll back to the
  fallback Whisper image.

Speed knobs (mostly mined out for the current container shape):

- `cuda-python` is already pinned in `requirements-nemo.txt` for the CUDA-graph
  fast path. Local wall clock improved 7%; cloud speed unchanged from `nemo-zs`
  to `nemo-zs-v2` because audio decode + HTTP / Python overhead dominate at
  cloud-side latency.
- Possible further speed lever: drop `librosa` resample (audio is already 16
  kHz wav per the manifest spec). Small magnitude, risk of regression on
  edge-case sample rates.

## Whisper iteration knobs (legacy / fallback)

Same content as before, kept for the fallback path:

1. **Re-export `--quantization int8_float16`** — same merged checkpoint, no retrain. Expected speed `0.849 → 0.90+`, accuracy delta `≤ 0.005`. Tag the resulting image `ft-lora32-int8f16`. This is the **next submission**.
2. **`beam_size=2`** (not 5) at [../../asr/src/asr_manager.py](../../asr/src/asr_manager.py) — only worth trying if step 1's int8 image has speed margin ≥ 0.92 AND accuracy is borderline. Typically buys 0.002–0.005 WER.

Deferred (don't pursue unless step 1 unexpectedly regresses below 0.95):

- **`--lora-rank 32 → 64`** (+ `--lora-alpha 64 → 128`, `--lr 1e-4 → 5e-5`, `--epochs 3 → 5`) — would have been Step 1b had `ft-lora32-v1` not hit the target. Val WER was still gently falling at epoch 3 (0.04982 → 0.04662), so there's likely 0.005–0.010 more to extract, but not enough to justify another 7-hour run at this stage of the qualifier.
- **Encoder unfreeze for one low-LR pass** — middle ground between our full-freeze and full FT. Per notebook 03's `freeze_feature_encoder()` pattern.
- **Ensemble distil-large-v3 + whisper-large-v3 with ROVER vote** — ~2× inference cost, only worth it if the accuracy ceiling becomes the bottleneck.

Other knobs that exist but aren't current levers:

- `--per-device-batch-size`, `--grad-accum` — default 8 × 2 = effective 16, T4-safe. Drop to 4 × 4 if OOM, raise on bigger GPUs.
- `--noise-dir <path>` — wire in noise corpus if/when one is available on Workbench (none found 12 May).
