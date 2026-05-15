# NLP — training fine-tuned QA models

Two training paths, picked by which one we want to ship:

| Script | Model class | Output dir | Manager priority |
|---|---|---|---|
| [`finetune_qa.py`](finetune_qa.py) | extractive (RoBERTa-SQuAD2) | `nlp/models/roberta-finetuned-squad2/` | 2 (after generative) |
| [`finetune_genqa.py`](finetune_genqa.py) | generative (Flan-T5) | `nlp/models/flan-t5-finetuned/` | 1 (preferred when present) |

The container's [nlp_manager.py](../../nlp/src/nlp_manager.py) auto-detects which is bundled (via `config.is_encoder_decoder`) and routes inference to either span extraction or `.generate()` accordingly. Container log prints `QA model: gen-finetuned (generative) ...` or `ext-finetuned (extractive) ...` on first request.

## Local corpus inspection

When a local novice corpus snapshot is available under gitignored `data/`, use:

```bash
python training/nlp/analyze_answer_templates.py \
  --data data/novice-nlp-light-20260515/novice/nlp/nlp.jsonl \
  --docs data/novice-nlp-light-20260515/novice/nlp/documents
```

This is stdlib-only and does not train on the data. It summarizes question templates, answer types, and whether each gold answer appears literally in its source docs. The 15 May snapshot showed `481/883` answers are not literal source substrings, including `225/592` L1 cases, so the next lever after `v9-doc-ensemble` is answer syntax / canonicalization rather than another retrieval-only push.

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

Expected local equiv_rate after fine-tune: **0.80–0.90** (up from v5c's 0.674). Expected cloud: **0.60–0.80** (cloud distribution shift typically eats some of the local gain; we've seen ~0.20 gap on this competition).

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
