"""QLoRA fine-tune of Qwen3-8B for the NLP RAG answerer (v15-lora).

Same architecture lever that took v7-finetuned-v1 → v8b-chunked-context
(+0.162 cloud accuracy from RoBERTa-large), applied to the generative LLM
used by v14d. The expectation is the LLM has more capacity to absorb the
Clairos vocabulary, the answer-form conventions (PCE dates, codenames,
unit phrasing), and the verbatim-quoting behavior that the cloud
ModernBERT-AE @ 0.9 evaluator rewards.

Why QLoRA, not full fine-tune:
- Qwen3-8B BF16 weights = ~16 GB; full fine-tune doesn't fit on Workbench T4
  (16 GB).
- QLoRA loads the base in 4-bit nf4 (bitsandbytes, ~4.5 GB) and trains
  small fp16 adapter matrices. Total training-time VRAM ≈ 10-12 GB on a T4
  with gradient_checkpointing + bs=2 + grad_accum=4.
- The trained adapter is ~50-150 MB. vLLM 0.9 loads LoRA adapters on top of
  AWQ-quantised bases at inference time via LoRARequest, so we never need
  to merge-then-requantize.

Why train on chunked contexts (not full docs):
- v8b proved the inference-distribution training principle on this exact
  corpus: the answerer is run against 3 reranked chunks at inference, so
  training on 3 chunks per question is the right loss surface. The
  exact-source-doc fallback (full doc) was tested by v7-v1 and was the
  baseline; chunked-context was +0.162 cloud accuracy.
- This script reuses `_doc_chunks_with_offsets` from `finetune_qa.py` to
  get the same chunking as inference.

Usage on Workbench:

    cd ~/til
    python training/nlp/finetune_lora.py \\
        --base Qwen/Qwen3-8B \\
        --epochs 2 \\
        --out nlp/models/lora

Outputs:
    nlp/models/lora/{adapter_config.json,adapter_model.safetensors,...}
    training/nlp/runs/lora-<timestamp>/                 # trainer logs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Lazy imports: heavy deps loaded only inside main() so --help is responsive.

# Match nlp_manager.py's chunking exactly. Keep these in sync.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CHUNK_SENTENCES = 3
CHUNK_OVERLAP = 1
MAX_CONTEXT_CHUNKS = 3
MAX_SEQ_LEN = 2048

# Must match llm_answerer._DEFAULT_SYSTEM_PROMPT exactly — drift here
# silently destroys the fine-tune (train/inference prompt mismatch).
SYSTEM_PROMPT = (
    "You are an extractive question-answering assistant for the world of "
    "Clairos.\n"
    "Rules:\n"
    "1. Answer ONLY from the provided context. If the context does not "
    "contain the answer, return an empty string.\n"
    "2. Quote the answer using the EXACT wording, dates, numbers, and units "
    "from the context. Do not paraphrase. Do not add explanation. Do not "
    "wrap the answer in quotes.\n"
    "3. Keep the answer as short as possible — typically 1 to 8 words, "
    "never more than a single short sentence.\n"
    "4. For dates use the source format (e.g. 76-07-19, Q4 78 PCE). For "
    "money use the source units (e.g. 4.5 million Phi Credits). For "
    "codenames return the uppercase token only (e.g. SEASTITCH).\n"
    "5. Output only the answer, nothing else."
)


# ---------------------------------------------------------------- chunking


def _split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else ([text.strip()] if text.strip() else [])


def _chunk_doc(text: str) -> list[str]:
    sents = _split_sentences(text)
    if not sents:
        return []
    if len(sents) <= CHUNK_SENTENCES:
        return [" ".join(sents)]
    step = max(1, CHUNK_SENTENCES - CHUNK_OVERLAP)
    out: list[str] = []
    for i in range(0, len(sents), step):
        window = sents[i : i + CHUNK_SENTENCES]
        if not window:
            break
        out.append(" ".join(window))
        if i + CHUNK_SENTENCES >= len(sents):
            break
    return out


def _pick_training_chunks(
    answer: str, source_docs: list[str], docs_dir: Path
) -> list[str]:
    """For one training example, build the 3-chunk context the answerer
    will see at inference time.

    Strategy:
      1. For each source_doc, chunk it like inference does.
      2. Prefer chunks that contain the gold answer as a substring (case
         insensitive). These are the "answer chunks" — the v8b/v9 sweet
         spot.
      3. If no chunk contains the answer (true for the ~55% of paraphrased
         answers), take the first chunk from each source doc as a
         best-effort proxy.
      4. Cap to MAX_CONTEXT_CHUNKS (3) total.

    The inference manager retrieves+reranks across the whole corpus, but
    that requires running BGE+reranker on 296 documents for every one of
    883 training examples — a half-hour of setup. The source_docs field is
    a high-precision oracle for which docs contain the answer, so we save
    that overhead and let inference handle retrieval at test time.
    """
    answer_lc = (answer or "").lower().strip()
    answer_chunks: list[str] = []
    fallback_chunks: list[str] = []

    for did in source_docs or []:
        doc_path = docs_dir / f"{did}.txt"
        if not doc_path.exists():
            continue
        text = doc_path.read_text(errors="ignore")
        chunks = _chunk_doc(text)
        if not chunks:
            continue

        added_from_this_doc = 0
        if answer_lc:
            for c in chunks:
                if answer_lc in c.lower():
                    answer_chunks.append(c)
                    added_from_this_doc += 1
                    if added_from_this_doc >= 2:
                        break
        if added_from_this_doc == 0:
            fallback_chunks.append(chunks[0])

    chosen: list[str] = []
    for c in answer_chunks:
        if c not in chosen:
            chosen.append(c)
        if len(chosen) >= MAX_CONTEXT_CHUNKS:
            return chosen
    for c in fallback_chunks:
        if c not in chosen:
            chosen.append(c)
        if len(chosen) >= MAX_CONTEXT_CHUNKS:
            break
    return chosen


# ---------------------------------------------------------------- data build


def _load_few_shots(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    blob = json.loads(path.read_text())
    return list(blob.get("examples", []))


def _build_messages(
    question: str,
    chunks: list[str],
    answer: str,
    few_shots: list[dict[str, str]],
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shots:
        msgs.append(
            {
                "role": "user",
                "content": f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}",
            }
        )
        msgs.append({"role": "assistant", "content": ex["answer"]})
    ctx = "\n\n".join(chunks) if chunks else "(no context)"
    msgs.append(
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}
    )
    msgs.append({"role": "assistant", "content": answer})
    return msgs


def _load_examples(
    data_path: Path, docs_dir: Path, few_shots: list[dict[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped_no_chunks = 0
    skipped_no_answer = 0
    skipped_no_doc = 0
    for line in data_path.read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        # Skip unanswerable/false-premise cases. The Novice track shouldn't
        # have these but the schema allows them and they'd train the model
        # to emit empty strings.
        if not ex.get("answerable", True):
            continue
        answer = (ex.get("answer") or "").strip()
        if not answer:
            skipped_no_answer += 1
            continue
        source_docs = ex.get("source_docs") or []
        if not source_docs:
            skipped_no_doc += 1
            continue
        chunks = _pick_training_chunks(answer, source_docs, docs_dir)
        if not chunks:
            skipped_no_chunks += 1
            continue
        question = ex.get("question") or ""
        rows.append(
            {
                "messages": _build_messages(question, chunks, answer, few_shots),
                "question": question,
                "answer": answer,
            }
        )
    print(
        f"[finetune_lora] loaded {len(rows)} training examples "
        f"(skipped: no-doc={skipped_no_doc}, no-answer={skipped_no_answer}, "
        f"no-chunks={skipped_no_chunks})",
        flush=True,
    )
    return rows


# ---------------------------------------------------------------- training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.getenv("LORA_BASE", "Qwen/Qwen3-8B"))
    parser.add_argument(
        "--data",
        default=os.getenv("LORA_DATA", "/home/jupyter/novice/nlp/nlp.jsonl"),
    )
    parser.add_argument(
        "--docs",
        default=os.getenv("LORA_DOCS", "/home/jupyter/novice/nlp/documents"),
    )
    parser.add_argument(
        "--few-shots",
        default=str(Path(__file__).resolve().parents[2] / "nlp/src/few_shots.json"),
    )
    parser.add_argument("--out", default="nlp/models/lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Held-out fraction for early stopping on eval_loss.",
    )
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    data_path = Path(args.data)
    docs_dir = Path(args.docs)
    few_shots_path = Path(args.few_shots)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        sys.exit(f"data file not found: {data_path}")
    if not docs_dir.is_dir():
        sys.exit(f"docs dir not found: {docs_dir}")

    print(f"[finetune_lora] base model: {args.base}", flush=True)
    print(f"[finetune_lora] data: {data_path}", flush=True)
    print(f"[finetune_lora] docs: {docs_dir}", flush=True)

    few_shots = _load_few_shots(few_shots_path)
    print(f"[finetune_lora] few-shots: {len(few_shots)} from {few_shots_path}", flush=True)

    rows = _load_examples(data_path, docs_dir, few_shots)
    if not rows:
        sys.exit("no training examples built; aborting")

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    print(
        f"[finetune_lora] split: train={len(split['train'])} eval={len(split['test'])}",
        flush=True,
    )

    # 4-bit nf4 load; bf16 compute dtype works on T4 (sm_75) via emulation.
    # Double quant saves another ~0.4 bits/param.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False  # required when gradient_checkpointing=True
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention projections only; adding MLP projections raises capacity
        # but doubles VRAM cost.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Strip the extra `question` / `answer` debug columns so the trainer
    # doesn't try to interpret them as text fields.
    split["train"] = split["train"].remove_columns(
        [c for c in ("question", "answer") if c in split["train"].column_names]
    )
    split["test"] = split["test"].remove_columns(
        [c for c in ("question", "answer") if c in split["test"].column_names]
    )

    run_name = f"lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    runs_dir = Path(__file__).resolve().parent / "runs" / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(runs_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        packing=False,
        # With a `messages`-format dataset, trl auto-applies the chat template
        # and masks loss to the assistant turn.
        completion_only_loss=True,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=sft_config,
        processing_class=tokenizer,
    )

    print("[finetune_lora] starting training...", flush=True)
    trainer.train()

    print(f"[finetune_lora] saving best adapter to {out_dir}", flush=True)
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Drop a small marker file for the manager to confirm the adapter was
    # produced by THIS base model — guards against silently mismatched bases.
    (out_dir / "BASE_MODEL").write_text(args.base.strip() + "\n")
    print(f"[finetune_lora] done. Adapter dir: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
