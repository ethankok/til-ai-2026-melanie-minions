# NLP — training fine-tuned QA models

Current status: **do not train more NLP blindly for the qualifier.** The
shipping truth after the 18 May Qwen3 push is:

- `v9-doc-ensemble` remains best blended: official `0.683 / 0.886`
  (blended 0.734).
- `v14-llm-rag` remains best raw accuracy: official `0.734 / 0.286`.
- `v15-lora-qwen3-8b` trained cleanly but is blocked at serving/packaging:
  direct vLLM LoRA crashes on T4/cloud, and merged-AWQ quantization is not
  working on the Workbench T4 with current tooling.

This README is historical context plus reproduction notes. If NLP is reopened,
the credible routes are: quantize the merged Qwen3 LoRA on bigger hardware, or
retrain LoRA on Qwen2.5-7B so it can run on the NGC base image that already
survived cloud.

Historical training paths:

| Script | Model class | Output dir | Manager priority |
|---|---|---|---|
| [`finetune_qa.py`](finetune_qa.py) | extractive (RoBERTa/DeBERTa SQuAD2) | `nlp/models/roberta-finetuned-squad2/` or `nlp/models/deberta-finetuned-squad2/` | DeBERTa is preferred if present, otherwise RoBERTa |
| [`finetune_genqa.py`](finetune_genqa.py) | generative (Flan-T5) | `nlp/models/flan-t5-finetuned/` | explicit `NLP_QA_MODE=generative` only |
| [`train_answer_ranker.py`](train_answer_ranker.py) | lightweight candidate ranker | `nlp/models/answer_ranker.json` | optional v12 reranker |
| [`finetune_lora.py`](finetune_lora.py) | QLoRA generative LLM adapter | `nlp/models/lora/` | trained successfully; not currently shippable on Qwen3/vLLM T4 stack |
| [`merge_lora_and_quantize.py`](merge_lora_and_quantize.py) | merge LoRA + AWQ quantize | `nlp/models/qwen3-8b-merged-bf16/`, `nlp/models/llm-merged/` | merge works; AWQ quant is blocked on T4 |

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
- Use an isolated `~/quant-venv` for quantization experiments:

```bash
python3 -m venv ~/quant-venv
source ~/quant-venv/bin/activate
pip install --upgrade pip
pip install torch==2.10.0 llmcompressor transformers accelerate peft safetensors datasets
python -c "from llmcompressor import oneshot; from llmcompressor.modifiers.awq import AWQModifier; print('OK')"
```

AWQ status:

- BF16 merge into `nlp/models/qwen3-8b-merged-bf16/` works.
- `autoawq` is not a reliable path now; the import path used by old examples is
  broken/deprecated.
- `llm-compressor` 0.10 on T4 failed both at DecoderLayer granularity (OOM) and
  at `sequential_targets=["Linear"]` / `max_seq_length=1024` with a Qwen3-GQA
  symbolic-trace `NoneType` failure after 3/254 calibration groups.
- So `nlp/models/llm-merged/` is not a usable artifact yet, and any Docker build
  that falls through without that directory is just testing the un-tuned base.

The container's [nlp_manager.py](../../nlp/src/nlp_manager.py) now defaults to
extractive QA even if an old Flan-T5 directory is present, because `v8a-genqa`
regressed on cloud. Set `NLP_QA_MODE=generative` only when deliberately
reproducing that experiment. Container log prints `QA model: ...` on first
request.

Build warning: the manager will prefer `deberta-finetuned-squad2` if that
directory is bundled. Since v13b failed, a safety rebuild intended to preserve
v9 must avoid bundling DeBERTa weights and must avoid candidate-ranker defaults.

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
