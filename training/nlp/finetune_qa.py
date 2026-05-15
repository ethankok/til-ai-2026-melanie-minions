"""Fine-tune a SQuAD-style QA model on the local NLP corpus.

Reads /home/jupyter/<track>/nlp/nlp.jsonl (the 883 ground-truth Q/A/source
tuples) plus /home/jupyter/<track>/nlp/documents/ (the corpus), builds
SQuAD-format training examples (question + context + answer-span), and
fine-tunes a SQuAD2-pretrained encoder. Output is saved under
nlp/models/roberta-finetuned-squad2/ for the container to pick up.

Strategy:
- The local 883 Q/A pairs follow the exact style and vocabulary of the
  cloud held-out set. Fine-tuning a SQuAD-pretrained encoder on this
  data teaches it Clairos proper nouns, the question wording style, and
  the canonical answer formatting (dates, prices, names). Empirically
  this is how 0.95+ NLP scores are reached on this competition.
- We use the first source_doc as the context for each question. If the
  answer string doesn't appear verbatim in that doc, we skip the example
  (rather than fake an answer span). Typical retention rate: ~70-80%.

Usage on Workbench:

    cd ~/til
    python training/nlp/finetune_qa.py \\
        --base-model deepset/roberta-large-squad2 \\
        --epochs 3 \\
        --batch-size 8

Outputs:
    nlp/models/roberta-finetuned-squad2/{config,model.safetensors,tokenizer*}
    training/nlp/runs/<timestamp>/             trainer logs + checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from datetime import datetime
from pathlib import Path

# Lazy imports of heavy deps so --help still works without them installed.

# rapidfuzz is optional; gives the strongest paraphrase recovery when available.
try:
    from rapidfuzz import fuzz as _rfuzz  # type: ignore
    _RAPIDFUZZ = True
except ImportError:
    _rfuzz = None  # type: ignore
    _RAPIDFUZZ = False


_LEADING_ARTICLES = ("the ", "The ", "a ", "A ", "an ", "An ")
_TRAILING_TRIM = string.punctuation + " \t\n"


def _strip_articles(s: str) -> str:
    for art in _LEADING_ARTICLES:
        if s.startswith(art):
            return s[len(art):]
    return s


def _answer_variants(answer: str) -> list[str]:
    """Permutations of an answer that are still semantically the same.

    Used in order; first match wins. Avoids false positives by only
    trimming punctuation/articles, never reordering words or rewriting.
    """
    seen: set[str] = set()
    out: list[str] = []
    for v in (
        answer,
        answer.strip(_TRAILING_TRIM),
        _strip_articles(answer),
        _strip_articles(answer.strip(_TRAILING_TRIM)),
        # Possessive 's stripped — "Velez's" -> "Velez"
        re.sub(r"'s\b", "", answer).strip(_TRAILING_TRIM),
    ):
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _find_span(answer: str, context: str) -> tuple[int, str] | None:
    """Locate `answer` (or a near variant) in `context`. Returns (start, exact_text).

    Strategy ladder (first hit wins):
      1. Exact + case-insensitive on each `_answer_variants(answer)`.
      2. Flexible-whitespace / optional-trailing-punctuation regex.
      3. (if rapidfuzz installed) sliding-window partial_ratio >= 88.
    """
    # 1. Variants × case-{sensitive,insensitive} find.
    ctx_lower = context.lower()
    for v in _answer_variants(answer):
        if not v:
            continue
        pos = context.find(v)
        if pos >= 0:
            return pos, v
        pos = ctx_lower.find(v.lower())
        if pos >= 0:
            return pos, context[pos : pos + len(v)]

    # 2. Flexible whitespace + optional trailing punctuation regex.
    a_norm = answer.strip(_TRAILING_TRIM)
    if a_norm:
        pattern = re.escape(a_norm)
        pattern = re.sub(r"\\\s+", r"\\s+", pattern)
        pattern = pattern + r"[.,;:!?'\")\]]*"
        m = re.search(pattern, context, re.IGNORECASE)
        if m:
            return m.start(), m.group(0)

    # 3. Optional fuzzy match via rapidfuzz.
    if _RAPIDFUZZ and len(answer) >= 3:
        base = answer.lower()
        n = len(answer)
        best_score = 0.0
        best_pos = -1
        best_len = 0
        # Try a few window lengths; stride keeps it ~O(L) per length.
        step = max(1, n // 4)
        for span_len in (n, max(1, int(n * 0.9)), int(n * 1.1), int(n * 1.3)):
            if span_len <= 0 or span_len > len(context):
                continue
            for i in range(0, len(context) - span_len + 1, step):
                window = context[i : i + span_len]
                score = _rfuzz.ratio(window.lower(), base)
                if score > best_score:
                    best_score = score
                    best_pos = i
                    best_len = span_len
        if best_score >= 88.0 and best_pos >= 0:
            return best_pos, context[best_pos : best_pos + best_len]

    return None


def _build_squad_examples(track: str) -> list[dict]:
    data_dir = Path(f"/home/jupyter/{track}/nlp")
    jsonl_path = data_dir / "nlp.jsonl"
    docs_dir = data_dir / "documents"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"missing ground truth: {jsonl_path}")
    if not docs_dir.exists():
        raise FileNotFoundError(f"missing docs dir: {docs_dir}")

    docs = {p.stem: p.read_text() for p in docs_dir.glob("*.txt")}
    print(f"loaded {len(docs)} documents from {docs_dir}", file=sys.stderr)

    examples: list[dict] = []
    skipped = {"empty_answer": 0, "no_source_doc": 0, "doc_not_found": 0,
               "answer_not_in_doc": 0}
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            srcs = row.get("source_docs") or []
            if not a:
                skipped["empty_answer"] += 1
                continue
            if not srcs:
                skipped["no_source_doc"] += 1
                continue
            # Try each source_doc; first hit wins.
            found_span = None
            chosen_doc = None
            for doc_id in srcs:
                if doc_id not in docs:
                    continue
                hit = _find_span(a, docs[doc_id])
                if hit is not None:
                    found_span = hit
                    chosen_doc = doc_id
                    break
            if found_span is None:
                if all(d not in docs for d in srcs):
                    skipped["doc_not_found"] += 1
                else:
                    skipped["answer_not_in_doc"] += 1
                continue
            pos, a_exact = found_span
            examples.append({
                "question": q,
                "context": docs[chosen_doc],
                "answers": {"text": [a_exact], "answer_start": [pos]},
            })

    print(
        f"built {len(examples)} examples; skipped {skipped} "
        f"(rapidfuzz={'on' if _RAPIDFUZZ else 'off'})",
        file=sys.stderr,
    )
    return examples


def _prepare_features(examples, tokenizer, max_seq_len: int, doc_stride: int):
    questions = [ex["question"] for ex in examples]
    contexts = [ex["context"] for ex in examples]
    answers = [ex["answers"] for ex in examples]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_seq_len,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions: list[int] = []
    end_positions: list[int] = []

    for feat_idx, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][feat_idx]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_idx = sample_mapping[feat_idx]
        ans = answers[sample_idx]
        start_char = ans["answer_start"][0]
        end_char = start_char + len(ans["text"][0])

        seq_ids = tokenized.sequence_ids(feat_idx)
        ctx_start = 0
        while ctx_start < len(seq_ids) and seq_ids[ctx_start] != 1:
            ctx_start += 1
        ctx_end = len(seq_ids) - 1
        while ctx_end >= 0 and seq_ids[ctx_end] != 1:
            ctx_end -= 1

        # Answer fully outside this feature's context window -> mark as no-answer (CLS).
        if (
            ctx_end < ctx_start
            or offsets[ctx_start][0] > end_char
            or offsets[ctx_end][1] < start_char
        ):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        s = ctx_start
        while s <= ctx_end and offsets[s][0] <= start_char:
            s += 1
        start_positions.append(s - 1)

        e = ctx_end
        while e >= ctx_start and offsets[e][1] >= end_char:
            e -= 1
        end_positions.append(e + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-model", default="deepset/roberta-large-squad2",
                   help="HF repo id of the SQuAD-pretrained checkpoint to fine-tune")
    p.add_argument("--track", default=os.getenv("TEAM_TRACK", "novice"),
                   choices=["novice", "advanced"])
    p.add_argument("--output", default="nlp/models/roberta-finetuned-squad2",
                   help="where to save the final model (relative to repo root)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-seq-len", type=int, default=384)
    p.add_argument("--doc-stride", type=int, default=128)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp16", action="store_true",
                   help="disable fp16 training (default: enabled on cuda)")
    args = p.parse_args()

    import random
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    examples = _build_squad_examples(args.track)
    if not examples:
        print("no training examples built — aborting", file=sys.stderr)
        return 1

    # Deterministic shuffle, then split.
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    val_n = max(int(len(examples) * args.val_fraction), 20)
    train_examples = examples[:-val_n]
    val_examples = examples[-val_n:]
    print(f"split: {len(train_examples)} train / {len(val_examples)} val",
          file=sys.stderr)

    print(f"loading base model: {args.base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)

    train_feats = _prepare_features(
        train_examples, tokenizer, args.max_seq_len, args.doc_stride
    )
    val_feats = _prepare_features(
        val_examples, tokenizer, args.max_seq_len, args.doc_stride
    )
    print(
        f"features: {len(train_feats['input_ids'])} train / "
        f"{len(val_feats['input_ids'])} val (overflow expansion included)",
        file=sys.stderr,
    )

    train_ds = Dataset.from_dict(train_feats)
    val_ds = Dataset.from_dict(val_feats)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(f"training/nlp/runs/{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    fp16 = (not args.no_fp16) and torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"saved fine-tuned model -> {out_dir.resolve()}", file=sys.stderr)
    print(f"trainer logs/checkpoints -> {run_dir.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
