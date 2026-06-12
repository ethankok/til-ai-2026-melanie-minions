"""Fine-tune a seq2seq (generative) QA model on the local NLP corpus.

Unlike `finetune_qa.py` (extractive: predicts a span in the context),
this script trains a seq2seq model to *generate* the answer text given
the question + context. Because we don't need the answer to appear
verbatim in the source doc, we can include all 883 (question, answer,
source_docs) tuples — coverage that extractive can't match (we lose
~530 to "answer_not_in_doc" skips in the extractive prep).

Trade-offs vs extractive:
- (+) Full coverage of paraphrased answers; covers the `retrieval_hit_diff`
      bucket that extractive can't reach.
- (+) Can compose multi-fact L2 answers from chunked context.
- (-) The AE 0.9 ModernBERT threshold rewards near-verbatim source spans;
      generated paraphrases sometimes fail equivalence even when correct.
- (-) Per-question latency higher (autoregressive decoding); may impact
      the 25% speed score.

Default base: `google/flan-t5-base` (~250M params, ~1 GB on disk). Switch
to `flan-t5-large` (780M, ~3 GB) for max capacity if speed budget allows.

Usage on Workbench:

    cd ~/til
    python training/nlp/finetune_genqa.py
    # or with bigger model:
    python training/nlp/finetune_genqa.py --base-model google/flan-t5-large --batch-size 4

Outputs:
    nlp/models/flan-t5-finetuned/{config,model.safetensors,tokenizer*}
    training/nlp/runs/<timestamp>/                                trainer logs

The NLP manager (nlp/src/nlp_manager.py) auto-detects whether the
fine-tuned model is QA (encoder, has SQuAD heads) or seq2seq (has a
decoder) and routes inference accordingly. See QA_GENERATIVE_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Reuse the extractive trainer's variant/regex/fuzzy span-finding; here we
# just emit (question, context, answer) triples, no answer_start needed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_qa import (  # noqa: E402
    _doc_chunks_with_offsets,
    _chunk_containing,
    _find_span,
)


def _build_genqa_examples(track: str, use_chunk_context: bool) -> list[dict]:
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
    skipped = {"empty_answer": 0, "no_source_doc": 0, "doc_not_found": 0}

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
            # Pick the first source_doc that exists; the answer need not
            # appear in it verbatim — generative training accepts the mismatch.
            chosen_doc = None
            for doc_id in srcs:
                if doc_id in docs:
                    chosen_doc = doc_id
                    break
            if chosen_doc is None:
                skipped["doc_not_found"] += 1
                continue

            ctx = docs[chosen_doc]
            if use_chunk_context:
                # Narrow to the chunk containing the answer (mirrors inference)
                # if findable; otherwise keep the whole doc.
                hit = _find_span(a, ctx)
                if hit is not None:
                    pos, a_exact = hit
                    chunk_hit = _chunk_containing(ctx, pos, pos + len(a_exact))
                    if chunk_hit is not None:
                        ctx = chunk_hit[0]

            examples.append({"question": q, "context": ctx, "answer": a})

    print(
        f"built {len(examples)} examples; skipped {skipped} "
        f"(chunk_context={'on' if use_chunk_context else 'off'})",
        file=sys.stderr,
    )
    return examples


def _format_input(q: str, ctx: str) -> str:
    # Flan-T5 expects a flat text prompt; question-before-context matches
    # the HF SQuAD pretraining convention.
    return f"question: {q.strip()} context: {ctx.strip()}"


def _tokenize_for_seq2seq(examples, tokenizer, max_input_len: int, max_target_len: int):
    inputs = [_format_input(ex["question"], ex["context"]) for ex in examples]
    targets = [ex["answer"] for ex in examples]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_len,
        truncation=True,
        padding="max_length",
    )
    # Mask pad tokens in labels so loss doesn't reward predicting padding.
    pad_id = tokenizer.pad_token_id
    label_ids = [
        [(tok if tok != pad_id else -100) for tok in seq]
        for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-model", default="google/flan-t5-base",
                   help="HF repo id of the seq2seq checkpoint to fine-tune. "
                        "flan-t5-small (77M), flan-t5-base (250M), "
                        "flan-t5-large (780M) all work.")
    p.add_argument("--track", default=os.getenv("TEAM_TRACK", "novice"),
                   choices=["novice", "advanced"])
    p.add_argument("--output", default="nlp/models/flan-t5-finetuned",
                   help="where to save the final model (relative to repo root)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-input-len", type=int, default=512)
    p.add_argument("--max-target-len", type=int, default=64,
                   help="must be >= the eval's 64-token answer cap")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="bf16",
                   choices=["fp32", "fp16", "bf16"],
                   help="training precision. Default bf16 — T5 has known fp16 "
                        "overflow in attention that produces NaN gradients from "
                        "step 1 (seen 15/05 with flan-t5-base on T4). bf16 is "
                        "stable on T5 and runs on any Ampere+ or T4 GPU. Use "
                        "fp32 if bf16 is somehow unavailable.")
    p.add_argument("--use-chunk-context", action="store_true",
                   help="narrow context to the inference-style chunk containing "
                        "the answer (when findable). Mirrors inference; usually "
                        "improves answer fidelity.")
    args = p.parse_args()

    import random
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    examples = _build_genqa_examples(args.track, use_chunk_context=args.use_chunk_context)
    if not examples:
        print("no training examples built — aborting", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    val_n = max(int(len(examples) * args.val_fraction), 20)
    train_examples = examples[:-val_n]
    val_examples = examples[-val_n:]
    print(f"split: {len(train_examples)} train / {len(val_examples)} val",
          file=sys.stderr)

    print(f"loading base model: {args.base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    train_feats = _tokenize_for_seq2seq(
        train_examples, tokenizer, args.max_input_len, args.max_target_len
    )
    val_feats = _tokenize_for_seq2seq(
        val_examples, tokenizer, args.max_input_len, args.max_target_len
    )
    print(
        f"features: {len(train_feats['input_ids'])} train / "
        f"{len(val_feats['input_ids'])} val",
        file=sys.stderr,
    )

    train_ds = Dataset.from_dict(train_feats)
    val_ds = Dataset.from_dict(val_feats)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(f"training/nlp/runs/{timestamp}-genqa")
    run_dir.mkdir(parents=True, exist_ok=True)

    on_cuda = torch.cuda.is_available()
    use_fp16 = args.precision == "fp16" and on_cuda
    use_bf16 = args.precision == "bf16" and on_cuda
    if use_bf16 and not torch.cuda.is_bf16_supported():
        print("WARN: bf16 requested but GPU reports no bf16 support; "
              "falling back to fp32 for T5 stability.", file=sys.stderr)
        use_bf16 = False
    print(
        f"precision: {'fp16' if use_fp16 else 'bf16' if use_bf16 else 'fp32'} "
        f"(on_cuda={on_cuda})",
        file=sys.stderr,
    )
    targs = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        predict_with_generate=False,  # eval_loss is enough; skip generation in eval
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # T5 + fp16 can silently produce NaN gradients (loss=0.0, grad_norm=nan
    # from step 1) and HF will happily save the corrupted weights. Refuse to
    # save in that state.
    has_nan = any(
        not torch.isfinite(p).all().item()
        for p in model.parameters()
    )
    if has_nan:
        print(
            "ERROR: trained model contains NaN/Inf weights — refusing to save. "
            "Likely cause: precision mismatch (was the run in fp16?). "
            "Retry with --precision bf16 (default) or --precision fp32.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"saved fine-tuned generative model -> {out_dir.resolve()}",
          file=sys.stderr)
    print(f"trainer logs/checkpoints -> {run_dir.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
