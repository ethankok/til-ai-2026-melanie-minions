# ASR training pipeline (Workbench-only)

Fine-tunes `distil-whisper/distil-large-v3` on the novice ASR dataset with a
slang-aware data prep step, then exports to CTranslate2 for the inference
container in [../../asr/](../../asr/).

Nothing in this folder ships in the eval Docker image — it's all training-side
tooling.

## Design at a glance

- **Base model**: `distil-whisper/distil-large-v3` — English-only distillation of large-v3. Same encoder quality, ~6× faster decoder. Right call for the Novice (English-only) track; avoids paying for a multilingual decoder we don't need.
- **Fine-tune**: LoRA (rank 32, alpha 64) on decoder attention projections only. Encoder frozen (distil-v3 inherits a strong large-v3 encoder; budget should go to the decoder, which distillation shrank).
- **Slang prior**: tokens rare in standard English but frequent in the NLP corpus are mined into a single-line prompt baked next to the model weights. Passed as `initial_prompt=` to bias decoding; same set is used to oversample slang-containing audio clips during fine-tune.
- **Augmentation** (noisy-audio robustness): SpecAugment, optional online noise mixing at SNR 5–20 dB, optional ×0.9/×1.1 speed perturbation.
- **Inference**: faster-whisper with CT2 float16 (CPU fallback int8), greedy beam, VAD trim, `condition_on_previous_text=False`, `language="en"` forced. Server passes the whole HTTP batch to `manager.asr_batch(list[bytes])` for true GPU batching.

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

## End-to-end

```bash
# 1. Mine in-world slang from the NLP corpus.
python training/asr/extract_slang.py \
    --nlp-dir /home/jupyter/novice/nlp \
    --out asr/models/slang_prompt.txt \
    --top-k 80

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

- `asr.jsonl` is the official novice ASR manifest at
  `/home/jupyter/novice/asr/asr.jsonl`. `prepare_data.py` reads `audio` /
  `path` and `transcript` fields. **Confirm on Workbench whether a separate
  training manifest exists** — `asr.jsonl` may be eval-only.
- WER on the held-out 10% val split is what `train_distil_whisper.py` reports;
  it uses the same `jiwer` transforms as the official scorer
  ([test/test_asr.py:24-33](../../test/test_asr.py)).
- The official scorer averages error rate across four language buckets
  (en/zh/ms/ta) even on Novice. Verify that empty non-English buckets
  contribute 0 to the sum (rather than raising in `jiwer`) before relying on
  the score target.
- Don't commit `asr/models/` or `training/asr/runs/` — they're in
  `.gitignore`.

## Iteration knobs (after first successful submission)

- `--per-device-batch-size`, `--grad-accum` — fit the GPU.
- `--lora-rank` 32 → 64 if val WER underfits.
- `--noise-dir <path>` — wire in MUSAN-style noise if the augment doesn't run by default.
- Try `beam_size=5` in `asr_manager.py` if speed budget allows after a fast pass.
- Unfreeze encoder for one low-LR pass at the end if WER plateau persists.
