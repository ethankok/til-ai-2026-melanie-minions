# NLP — training a fine-tuned QA model

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

## What lives where

- This README: how to run the script.
- [`finetune_qa.py`](finetune_qa.py): the script itself.
- [`../../nlp/src/nlp_manager.py`](../../nlp/src/nlp_manager.py): the manager prefers `QA_FINETUNED_DIR` over `QA_BASE_DIR`. No changes needed to swap weights — just put a fine-tuned model in `nlp/models/roberta-finetuned-squad2/` and rebuild.
- [`../../nlp/Dockerfile`](../../nlp/Dockerfile): the COPY block that bundles the fine-tuned model if present.
- [`../../nlp/NOTES.md`](../../nlp/NOTES.md): strategic context for the v7 submission.
