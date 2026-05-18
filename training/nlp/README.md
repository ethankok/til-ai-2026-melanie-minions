# NLP — training fine-tuned QA models

Current status: **do not train more NLP blindly for the qualifier.** The
shipping truth after the 18 May NLP push is:

- `v9-doc-ensemble-rescue` is the trusted blended submission: local `0.711`,
  cloud `0.683 / 0.866`, `0 / 700`.
- `v14-llm-rag` remains best raw accuracy: official `0.734 / 0.286`.
- `v15-lora-qwen3-8b` trained cleanly but is blocked at serving/packaging:
  direct vLLM LoRA crashes on T4/cloud, and merged-AWQ quantization is not
  working on the Workbench T4 with current tooling.
- Current experiment is `v19-hybrid-router`: v9 RoBERTa for easy questions,
  Qwen2.5-7B-AWQ only for heuristic-hard questions, on the v14 NGC base.

This README is historical context plus reproduction notes. If NLP is reopened,
the credible routes are: validate the v19 router, quantize the merged Qwen3
LoRA on bigger hardware, or retrain/merge on Qwen2.5-7B so it can run on the
NGC base image that already survived cloud.

Historical training paths:

| Script | Model class | Output dir | Manager priority |
|---|---|---|---|
| [`finetune_qa.py`](finetune_qa.py) | extractive (RoBERTa/DeBERTa/ModernBERT SQuAD2) | `nlp/models/*-finetuned-squad2/` | RoBERTa v9 fine-tune is preferred; later DeBERTa/ModernBERT artefacts failed gate |
| [`finetune_genqa.py`](finetune_genqa.py) | generative (Flan-T5) | `nlp/models/flan-t5-finetuned/` | explicit `NLP_QA_MODE=generative` only |
| [`train_answer_ranker.py`](train_answer_ranker.py) | lightweight candidate ranker | `nlp/models/answer_ranker.json` | optional v12 reranker |
| [`finetune_lora.py`](finetune_lora.py) | QLoRA generative LLM adapter | `nlp/models/lora/` | trained successfully; not currently shippable on Qwen3/vLLM T4 stack |
| [`merge_lora_and_quantize.py`](merge_lora_and_quantize.py) | merge LoRA + AWQ quantize | `nlp/models/qwen3-8b-merged-bf16/`, `nlp/models/llm-merged/` | merge works; AWQ quant is blocked on T4 |

## v16 DeBERTa-v3 extractive retry

`v13b-deberta` already tried `deepset/deberta-v3-large-squad2` with the v8b
chunked-context recipe and failed the local gate (`0.667` vs v9's `0.711`)
while running much slower. Do not repeat that exact run. The only DeBERTa retry
worth trying is a lower-learning-rate, one-epoch gate to avoid the overfit
pattern from v13b (`eval_loss` was best at epoch 1 and worsened afterward).

Historical note: during the v16 gate the Dockerfile defaulted back to
`NLP_ANSWERER=extractive`, skipped the LLM download, bundled local
`deberta-finetuned-squad2/` when present, and set `NLP_QA_MAX_SEQ_LEN=256`.
Current main has moved on to the v19 hybrid-router experiment.

Workbench commands:

```bash
cd ~/til
git pull origin main

python training/nlp/finetune_qa.py \
  --base-model deepset/deberta-v3-large-squad2 \
  --use-answer-chunk \
  --epochs 1 \
  --lr 1e-5 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --gradient-checkpointing \
  --output nlp/models/deberta-finetuned-squad2

til build nlp v16-deberta-v3
til test nlp v16-deberta-v3
```

Result: local `NLP RAG QA Accuracy` was `0.692`, with the QA loop taking
`8:27`. This improves on the old `v13b-deberta` run (`0.667`) but still misses
the v9 local gate (`0.711`) and remains much slower than v9 (`~3:48`). Do not
submit; keep `v9-doc-ensemble` as the blended-score submission.

## v17 ModernBERT stock vs fine-tuned A/B

Purpose: test whether training the extractive reader is still a real lever.
Use the same ModernBERT QA checkpoint first without Clairos training, then
fine-tune it on the retained local span examples and compare.

Implementation details:

- Stock QA checkpoint: `kiddothe2b/ModernBERT-base-squad2` downloaded into
  `/workspace/models/modernbert-base-squad2`.
- Fine-tuned output: `nlp/models/modernbert-finetuned-squad2`.
- Manager priority is now:
  `NLP_QA_MODEL_DIR override > modernbert-finetuned > modernbert-base-squad2 >
  deberta-finetuned > roberta-finetuned > roberta-base`.
- Use `NLP_QA_MODEL_DIR=/workspace/models/modernbert-base-squad2` for the
  stock run if a fine-tuned ModernBERT directory is already present.

Stock, no Clairos training:

```bash
cd ~/til
git pull origin main

til build nlp v17-modernbert-stock
til test nlp v17-modernbert-stock
```

Fine-tuned:

```bash
python training/nlp/finetune_qa.py \
  --base-model kiddothe2b/ModernBERT-base-squad2 \
  --use-answer-chunk \
  --epochs 1 \
  --lr 1e-5 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --output nlp/models/modernbert-finetuned-squad2

til build nlp v17-modernbert-ft
til test nlp v17-modernbert-ft
```

Gate: if stock is already near v9 (`0.711`) and fine-tuning improves it, this
is a credible extractive replacement. If stock and fine-tuned both sit below
v9, the evidence points against more QA-head training on this corpus.

Result:

```text
v17-modernbert-stock   local 0.459   QA loop 6:17
v17-modernbert-ft      local 0.624   QA loop 5:28
```

Training improved ModernBERT strongly (+0.165 absolute), so the training lever
is real, but the ModernBERT backbone still failed the v9 local gate (`0.711`)
and was slower than v9. Do not submit either ModernBERT image.

Cloud/startup lesson: `v16-deberta-v3` timed out in cloud and
`v17-modernbert-stock` failed Vertex startup. Both used the current
`vllm/vllm-openai` Docker base, so this is packaging/base evidence rather than
an extractive-reader finding. The old successful v9 image used the NGC PyTorch
base.

Do not use a sibling `git worktree` to rebuild v9 via `til build`; on Workbench
that still built the current-main Dockerfile and retagged the ModernBERT image
as `v9-doc-ensemble-rescue`. To rescue v9, check out or restore the v9-era NLP
files directly in canonical `~/til` before calling `til build`.

Validated rescue: after killing the stale container on port 5004 and rebuilding
from canonical `~/til`, `til test nlp v9-doc-ensemble-rescue` returned local
`0.711` with a 4:13 QA loop. This matches the real v9 baseline. Cloud score is
`0.683 / 0.866` with `0 / 700` errors; this rescue tag is the trusted NLP
submission candidate.

Current decision after the v18 session: do not spend more time on reranker
swaps or on post-hoc analysis of failed local runs. `nlp_results.json` is worth
mining only when a model clears the local gate or is close enough to explain a
small regression. `v18-qwen-reranker` was not close (`0.547`, 14:58 QA loop,
then cloud `700 / 700` errors), so keep v9 rescue and only reopen Qwen work on
the higher-upside answerer-serving problem.

## v18 Qwen reranker-only gate (18 May)

Do not try the full Qwen3-4B answerer stack first: `v14c-qwen3-4b` already
failed local gate at `0.659`. The cheap ablation was to keep the v9 RoBERTa
answerer and replace only the cross-encoder reranker:

```bash
cd ~/til
git pull origin main

docker build \
  --build-arg NLP_RERANKER_REPO=tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
  --build-arg NLP_RERANKER_LOCAL_NAME=qwen3-reranker-0.6b-seq-cls \
  -t melanie-minions-nlp:v18-qwen-reranker \
  nlp
til test nlp v18-qwen-reranker
```

Expected boot log should include:

```text
[nlp_manager] reranker model: /workspace/models/qwen3-reranker-0.6b-seq-cls
[nlp_manager] QA model: ext-roberta-finetuned
```

Result:

```text
v18-qwen-reranker local 0.547
QA loop: 14:58
cloud: 0.000 / 0.417, 700 / 700 errors
```

This failed both gates by a wide margin. Do not submit. The Dockerfile default
has been reverted to the BGE reranker so normal `til build nlp ...` recreates
the trusted v9-style retrieval stack unless the reranker env/build args are
explicitly overridden.

Interpretation: Qwen as an answerer is still promising (`Qwen3-8B-AWQ` local
0.755; `v14-llm-rag` cloud 0.734), but Qwen as a drop-in reranker was a bad fit
for this pipeline and too slow. The remaining credible Qwen work is packaging
and quantization, not reranking.

## v19 hybrid router gate (18 May)

This tests the GPT Pro recommendation: keep v9 for easy questions and route
only hard/L2-looking questions to the cloud-proven v14 Qwen2.5 answerer.

```bash
cd ~/til
git pull origin main

til build nlp v19-hybrid-router
til test nlp v19-hybrid-router
```

Expected boot logs:

```text
[llm_answerer] loading vLLM from /workspace/models/llm ...
[nlp_manager] QA model: ext-roberta-finetuned
[nlp_manager] hybrid routed N/M questions to Qwen (threshold=3.0)
```

Gate: submit only if local beats `0.711` and runtime is not wildly above v9.
If startup OOMs, try one smaller build by editing Docker/env defaults:

```text
NLP_LLM_GPU_MEM_FRACTION=0.55
NLP_LLM_MAX_MODEL_LEN=2048
```

If it runs but routes too many questions and slows down, raise
`NLP_HYBRID_QWEN_THRESHOLD` above `3.0`.

## v15 QLoRA / AWQ lessons (18 May)

The Qwen3-8B LoRA training command that completed was:

```bash
python training/nlp/finetune_lora.py \
  --base Qwen/Qwen3-8B \
  --data /home/jupyter/novice/nlp/nlp.jsonl \
  --docs /home/jupyter/novice/nlp/documents \
  --out nlp/models/lora \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8
```

Training facts:

- `batch-size 2` OOMed on T4; `batch-size 1 --grad-accum 8` fits and keeps
  the same effective batch size as `2 x 4`, just slower.
- A free `nvidia-smi` before launch only means the GPU is idle. During training,
  the 8B base, LoRA activations, sequence length, and shifted logits fill most
  of the 14-15 GiB T4 memory.
- The real T4 runtime was about 8 hours for 2 epochs / 200 steps at 2048 tokens.
  It was not a 30-minute run.
- If Workbench idle-shutdown settings cannot be edited, use `tmux` for the
  training process and keep a Jupyter notebook kernel active with a tiny
  heartbeat cell. GPU activity alone may not count as Workbench UI activity.

Environment lesson:

- Do **not** install `llmcompressor` into the main Workbench environment. It
  pins `torch<=2.10.0` and can downgrade a CUDA 13 / torch 2.12 stack, leaving
  `torchvision` compiled against the wrong torch and causing
  `RuntimeError: operator torchvision::nms does not exist`, which then surfaces
  through Transformers as `Could not import module 'PreTrainedModel'`.
- Do **not** run `til test` while `~/quant-venv` is active. The quant venv is
  only for merge/quantization and lacks normal test deps such as
  `python-dotenv`.
- If base-env `til test nlp ...` fails while loading ModernBERT with
  `operator torchvision::nms does not exist`, remove the broken optional
  `torchvision` package from the host env and rerun the evaluator:

```bash
deactivate 2>/dev/null || true
python -m pip uninstall -y torchvision
python -m pip uninstall -y torchvision  # repeat once in case both user/site copies exist
python - <<'PY'
from transformers import AutoModelForSequenceClassification
AutoModelForSequenceClassification.from_pretrained(
    "./test/models/nlp_eval_512",
    local_files_only=True,
)
print("ModernBERT evaluator imports OK")
PY
```

- Use an isolated `~/quant-venv` for quantization experiments:

```bash
python3 -m venv ~/quant-venv
source ~/quant-venv/bin/activate
pip install --upgrade pip
pip install torch==2.10.0 llmcompressor transformers accelerate peft safetensors datasets
python - <<'PY'
from llmcompressor import oneshot
try:
    from llmcompressor.modifiers.gptq import GPTQModifier
except ImportError:
    from llmcompressor.modifiers.quantization import GPTQModifier
print("OK")
PY
```

Merge/quant status:

- BF16 merge into `nlp/models/qwen3-8b-merged-bf16/` works.
- `autoawq` is not a reliable path now; the import path used by old examples is
  broken/deprecated.
- `llm-compressor` 0.10 on T4 failed both at DecoderLayer granularity (OOM) and
  at `sequential_targets=["Linear"]` / `max_seq_length=1024` with a Qwen3-GQA
  symbolic-trace `NoneType` failure after 3/254 calibration groups.
- The current script defaults to GPTQ W4A16 with `max_seq_length=256`, which
  avoids AWQ smoothing and is the T4-safe retry path. Keep GPTQ on the default
  block-level sequential pipeline; forcing per-Linear sequencing hits the Qwen3
  symbolic-trace `NoneType` failure at `o_proj`.
- On T4, GPTQ still OOMed at `model.layers.29.mlp.down_proj` during
  `torch.cholesky_inverse(H)` even with 32 samples and 256 tokens. The script
  now defaults to `--gptq-ignore-down-proj-from-layer 29`, leaving only
  `model.layers.29-35.mlp.down_proj` unquantized while quantizing the rest.
  Set `--gptq-ignore-down-proj-from-layer -1` only on larger GPUs.
- This produced `nlp/models/llm-merged/` successfully on 18 May. Build copied a
  6.61 GB context and the NLP container became healthy; local scoring is blocked
  only by the host ModernBERT evaluator env until the `torchvision` repair above
  is applied.
- After the host evaluator repair, `v15-merged-qwen3-8b` scored 0.659 locally.
  That matches the known v14c 4B / corrupted-runtime bucket, not the v14d 8B
  base score of 0.755. Treat it as failed unless container logs prove the
  correct merged GPTQ artifact was loaded and a higher-quality quantization is
  tested. Runtime now prints `model_type`, `quant_method`, and `MERGED_FROM`
  from `/workspace/models/llm` at boot for this verification.
- Any Docker build that falls through without `nlp/models/llm-merged/` is just
  testing the un-tuned base.

The container's [nlp_manager.py](../../nlp/src/nlp_manager.py) now defaults to
extractive QA even if an old Flan-T5 directory is present, because `v8a-genqa`
regressed on cloud. Set `NLP_QA_MODE=generative` only when deliberately
reproducing that experiment. Container log prints `QA model: ...` on first
request.

Build warning: the manager will prefer `modernbert-finetuned-squad2`, then
`modernbert-base-squad2`, then `deberta-finetuned-squad2` if those directories
are bundled. A safety rebuild intended to preserve v9 must either remove those
directories from `nlp/models/` or set `NLP_QA_MODEL_DIR` explicitly.

## Local corpus inspection

When a local novice corpus snapshot is available under gitignored `data/`, use:

```bash
python training/nlp/analyze_answer_templates.py \
  --data data/novice-nlp-light-20260515/novice/nlp/nlp.jsonl \
  --docs data/novice-nlp-light-20260515/novice/nlp/documents
```

This is stdlib-only and does not train on the data. It summarizes question templates, answer types, and whether each gold answer appears literally in its source docs. The 15 May snapshot showed `481/883` answers are not literal source substrings, including `225/592` L1 cases. Later canonicalization and candidate-ranker attempts did not transfer through the 0.9 AE threshold, so this remains diagnostic only.

## Replay answer canonicalization

After a Workbench `til test`, package or copy the saved predictions and replay the manager's post-processing without loading the Torch models:

```bash
python training/nlp/replay_canonicalizer.py \
  --results data/nlp-v11-failure-pack/nlp_results.json \
  --ground data/nlp-v11-failure-pack/nlp.jsonl \
  --docs data/novice-nlp-light-20260515/novice/nlp/documents \
  --changed-out data/nlp-v11-failure-pack/v11_canonicalizer_changed.jsonl
```

This script imports [nlp_manager.py](../../nlp/src/nlp_manager.py) with lightweight stubs for model-only dependencies, so it tests the actual `_canonicalize_answer` implementation. The first v11 replay against the v9 failure pack moved the exact/substr proxy `451 -> 461` and diff `395 -> 385`, with retrieval unchanged at `37` misses and no proxy regressions.

## v12 — answer-candidate reranking

The v12 path keeps the winning v9 retrieval and extractive RoBERTa answerer,
then generates multiple short answer candidates:

- top RoBERTa spans instead of only the single best span
- arithmetic/date candidates from existing rules
- full-document canonicalizer candidates
- literal candidates mined from the top-3 returned docs: dates, money, codes,
  percentages, proper nouns, and short relation phrases

The manager chooses with a conservative heuristic by default. If
`nlp/models/answer_ranker.json` exists, Docker bakes it into
`/workspace/models/answer_ranker.json` and the manager uses it as a learned
candidate ranker with a small heuristic prior.

Train the ranker on Workbench:

```bash
cd /home/jupyter/til

python training/nlp/train_answer_ranker.py \
  --data /home/jupyter/novice/nlp/nlp.jsonl \
  --docs /home/jupyter/novice/nlp/documents \
  --out nlp/models/answer_ranker.json

til build nlp v12-candidate-ranker
til test nlp v12-candidate-ranker
python nlp/error_report.py /home/jupyter/melanie-minions/nlp_results.json \
                          /home/jupyter/novice/nlp/nlp.jsonl
til submit nlp v12-candidate-ranker
```

Final outcome: v12 scored 0.663 locally and 0.642/0.829 officially, then v13a
confirmed both learned and heuristic candidate selection were below v9. Do not
submit or retrain this path.

## v7 — extractive fine-tune (shipped at 0.517/0.880)

`finetune_qa.py` takes the local `/home/jupyter/<track>/nlp/nlp.jsonl` ground-truth (883 question / answer / source_docs tuples for novice), pulls the corresponding documents from `/home/jupyter/<track>/nlp/documents/`, builds SQuAD-style training examples, and fine-tunes a SQuAD2-pretrained encoder on them.

The fine-tuned weights are saved to `nlp/models/roberta-finetuned-squad2/`. The NLP Dockerfile picks them up automatically at build time; the manager prefers fine-tuned weights over the stock SQuAD2 weights it bundles via `download_models.py`.

## Why fine-tune

The cloud held-out questions follow the exact same style as the local `nlp.jsonl` set (same authoring team, same fictional world). Stock `deepset/roberta-base-squad2` knows SQuAD-style English QA but doesn't know:
- Clairos proper nouns (the world's organisations, places, IDs)
- The canonical answer format (e.g. dates as `76-07-19`, prices as `4.5 million Credits`)
- The question wording style ("What penalty was assessed against …", "What does X measure?")

Fine-tuning on the local 883 pairs teaches all three. Based on competition leaderboards, this is the lever that gets NLP scores from `~0.5` to `0.9+`.

## Run on Workbench

```bash
cd ~/til

# Default: fine-tune deepset/roberta-large-squad2, 3 epochs, batch 8, lr 3e-5.
# Expect ~30-60 min on a single GPU depending on which Workbench instance type.
python training/nlp/finetune_qa.py
```

Common knobs:

```bash
# Quicker iteration with the base model (~125M params, ~10x faster training).
python training/nlp/finetune_qa.py --base-model deepset/roberta-base-squad2 --epochs 3

# More epochs (watch eval_loss to avoid overfit; load_best_model_at_end is on)
python training/nlp/finetune_qa.py --epochs 5

# Different learning rate
python training/nlp/finetune_qa.py --lr 2e-5
```

Outputs:

- `nlp/models/roberta-finetuned-squad2/` — the saved model. Picked up by the Docker build.
- `training/nlp/runs/<timestamp>/` — Trainer logs and intermediate checkpoints (gitignored).

## Build and submit after training

```bash
til build nlp v7-finetuned
til test nlp v7-finetuned
# Container log should print: "[nlp_manager] QA model: finetuned (...)"

python nlp/error_report.py /home/jupyter/melanie-minions/nlp_results.json \
                          /home/jupyter/novice/nlp/nlp.jsonl

til submit nlp v7-finetuned
```

Historical expectation before v7/v8b: local equiv_rate **0.80–0.90** and cloud **0.60–0.80**. Actual best path was v8b/v9 RoBERTa chunked-context plus document ensemble; the later DeBERTa retune failed local gate.

## How `_build_squad_examples` chooses contexts

For each `(question, answer, source_docs)` row in `nlp.jsonl`:

1. Iterates through `source_docs` in order and reads each from `documents/`.
2. Calls `_find_span(answer, doc)`, which tries (in order):
   1. **Exact + case-insensitive find** on each of: original answer, trailing-punctuation-stripped, leading-article-stripped, possessive-stripped variants.
   2. **Flexible-whitespace regex** allowing `\s+` between answer tokens and optional trailing punctuation.
   3. **(optional) rapidfuzz sliding-window** — if `rapidfuzz` is installed, scans windows of length `len(answer) × {0.9, 1.0, 1.1, 1.3}` and accepts the best match if `ratio >= 88`.
3. The first source_doc that yields a span becomes the training context.

Empirical retention:

| Strategy ladder | Train pairs retained (out of 883) |
|---|---:|
| v1: exact + case-insensitive only | 353 (40%) |
| v2: + variants + flexible-whitespace | ~500 (57%) expected |
| v2 + rapidfuzz installed | ~600-650 (68-74%) expected |

To enable the fuzzy fallback:

```bash
pip install --user rapidfuzz
```

The skipped examples are those where the reference answer doesn't appear in source_docs in any near-verbatim form (rare; truly paraphrased answers).

## What the script does not do

- It does **not** run any answer-equivalence (AE) eval during training. Trainer eval_loss is the proxy; it's correlated with downstream AE score but not identical. After training, the actual NLP score comes from `til test` running the AE model.
- It does **not** try to handle L4/L5 unanswerable cases. Novice track only has L1/L2 (every question has source_docs and a real answer).
- It does **not** modify retrieval. Retrieval is at 95.5% hit rate already; this script only improves the QA step.

## v8b — chunked-context extractive training

Mirrors inference exactly: trains on the 3-sentence chunk that contains the answer (instead of the whole source doc). The model sees the same context distribution at train and inference time, which should compress the `retrieval_hit_diff` bucket.

```bash
python training/nlp/finetune_qa.py --use-answer-chunk
# Same output dir; overwrites nlp/models/roberta-finetuned-squad2/
```

Trade-offs vs whole-doc training (v7-v1):
- (+) Better calibration between train and inference contexts.
- (+) Shorter training contexts → faster training, larger effective batch.
- (−) Possibly less context for L2 multi-fact questions if the answer span crosses chunk boundaries (only ~1% of cases, given 1-sentence overlap).

Gate to submit: local equiv_rate ≥ 0.74 (clear beat over v7-v1's 0.709).

## v8a — generative QA fine-tune

Different script, different model class. Trains [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) (or `-large`) to generate the answer text given the prompt `question: ... context: ...`. Includes **all 883 examples** — no span-match requirement, so the 452 paraphrased answers that extractive can't see are now in the training set.

```bash
# Default: flan-t5-base (250M), 3 epochs, batch 8
python training/nlp/finetune_genqa.py

# Chunked context (recommended — matches inference)
python training/nlp/finetune_genqa.py --use-chunk-context

# Bigger model if memory allows
python training/nlp/finetune_genqa.py --base-model google/flan-t5-large --batch-size 4
```

Outputs to `nlp/models/flan-t5-finetuned/`. The manager will detect it via `config.is_encoder_decoder` and route inference through `.generate()` with `num_beams=4`.

Trade-offs vs extractive:
- (+) Full coverage of paraphrased answers; covers the `retrieval_hit_diff` bucket extractive can't reach.
- (+) Can compose multi-fact L2 answers from chunked context.
- (−) AE 0.9 ModernBERT threshold rewards near-verbatim source spans; paraphrases sometimes fail equivalence even when semantically correct.
- (−) Per-question latency higher (autoregressive decoding with `num_beams=4`); may dip the 25% speed score.

Risk: if the speed dip is severe (cloud speed drops from 0.880 toward 0.70), blended score might *not* improve even with better accuracy. Worth testing locally before submitting.

## What lives where

- This README: how to run the scripts.
- [`finetune_qa.py`](finetune_qa.py): extractive (v7, v8b).
- [`finetune_genqa.py`](finetune_genqa.py): generative (v8a).
- [`../../nlp/src/nlp_manager.py`](../../nlp/src/nlp_manager.py): preference ladder is `QA_GEN_FINETUNED_DIR > QA_EXT_FINETUNED_DIR > QA_BASE_DIR > hub`. Inference path (`_extract_answer_span` vs `_generate_answer`) is chosen at load time from `config.is_encoder_decoder`.
- [`../../nlp/Dockerfile`](../../nlp/Dockerfile): COPY block bundles whichever fine-tuned dir is present in `nlp/models/`. Both can coexist; manager picks the generative one if both exist.
- [`../../nlp/NOTES.md`](../../nlp/NOTES.md): strategic context for v7/v8 submissions and confirmed regressors.
