"""Train the v13a NLP answer-candidate ranker using the ACTUAL ModernBERT
answer-equivalence model as the label source (not exact/substr proxy).

v12's regressor was the exact/substr proxy: it labelled "37" as positive when
the gold answer was "37 days" (because one is a substring of the other), but
the cloud scorer rates them ≠ at the 0.9 threshold. v13a fixes that by
running the official AE model from test/models/nlp_eval_512 over every
(question, gold, candidate) triple during training and labelling on prob >= 0.9.

Run on Workbench after the v9/v12 model weights are available:

  python training/nlp/train_answer_ranker.py \
      --data /home/jupyter/novice/nlp/nlp.jsonl \
      --docs /home/jupyter/novice/nlp/documents \
      --out nlp/models/answer_ranker.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.-]", "", text)
    return text


def _is_positive_proxy(pred: str, gold: str) -> bool:
    """Fallback exact/substr label — used only as a sanity signal alongside AE."""
    pred_norm = _norm(pred)
    gold_norm = _norm(gold)
    if not pred_norm or not gold_norm:
        return pred_norm == gold_norm
    return pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm


def _ae_label_batch(
    triples: list[tuple[str, str, str]],
    ae_model,
    ae_tokenizer,
    threshold: float,
    device,
    batch_size: int = 64,
) -> list[float]:
    """Score (question, gold, candidate) triples with the AE model.

    Returns 1.0 where AE prob >= threshold (cloud full credit), else 0.0.
    Empty-gold/empty-candidate are handled specially (matches test_nlp.py).
    """
    import torch
    import torch.nn.functional as F

    labels: list[float] = [0.0] * len(triples)
    eval_idx: list[int] = []
    eval_texts: list[str] = []

    for i, (q, gold, cand) in enumerate(triples):
        if not gold and not cand:
            labels[i] = 1.0
            continue
        if not gold or not cand:
            labels[i] = 0.0
            continue
        # Match the AE evaluator's input formatting exactly.
        from string import printable
        _printable = "".join(c for c in cand if c in printable)
        tokens = ae_tokenizer.tokenize(_printable, max_length=64, truncation=True)
        recon_cand = ae_tokenizer.convert_tokens_to_string(tokens)
        eval_idx.append(i)
        eval_texts.append(f"Question: {q} Reference: {gold} Candidate: {recon_cand}")

    with torch.no_grad():
        for start in range(0, len(eval_texts), batch_size):
            batch = eval_texts[start : start + batch_size]
            enc = ae_tokenizer(
                batch, max_length=128, padding="longest", truncation=True,
                return_tensors="pt",
            ).to(device)
            logits = ae_model(**enc).logits
            probs = F.softmax(logits, dim=-1)[:, 1].tolist()
            for j, prob in enumerate(probs):
                labels[eval_idx[start + j]] = 1.0 if prob >= threshold else 0.0

    return labels


def _load_docs(docs_dir: Path) -> list[dict[str, str]]:
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        docs.append({"id": path.stem, "document": path.read_text(errors="replace")})
    if not docs:
        raise FileNotFoundError(f"no .txt docs found under {docs_dir}")
    return docs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _train_logistic(
    feature_rows: list[dict[str, float]],
    labels: list[float],
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[float, dict[str, float]]:
    names = sorted({name for row in feature_rows for name in row})
    x = np.asarray(
        [[float(row.get(name, 0.0)) for name in names] for row in feature_rows],
        dtype=np.float32,
    )
    y = np.asarray(labels, dtype=np.float32)
    if x.size == 0:
        raise ValueError("no training candidates generated")

    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    sample_w = np.where(y > 0.5, neg / pos, 1.0).astype(np.float32)

    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = float(np.log(pos / neg))

    for _ in range(epochs):
        logits = x @ weights + bias
        probs = _sigmoid(logits)
        err = (probs - y) * sample_w
        denom = max(float(sample_w.sum()), 1.0)
        grad_w = (x.T @ err) / denom + l2 * weights
        grad_b = float(err.sum() / denom)
        weights -= lr * grad_w
        bias -= lr * grad_b

    return bias, {name: float(value) for name, value in zip(names, weights)}


def _score(row: dict[str, float], bias: float, weights: dict[str, float]) -> float:
    return bias + sum(weights.get(k, 0.0) * float(v) for k, v in row.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="path to nlp.jsonl")
    parser.add_argument("--docs", type=Path, required=True, help="path to documents/")
    parser.add_argument("--out", type=Path, default=Path("nlp/models/answer_ranker.json"))
    parser.add_argument("--limit", type=int, default=0, help="debug: first N rows only")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--ae-model-path", type=Path, default=Path("./test/models/nlp_eval_512"),
        help="Path to ModernBERT AE model (same one test_nlp.py uses)",
    )
    parser.add_argument(
        "--ae-threshold", type=float, default=0.9,
        help="AE prob threshold for positive label (matches cloud eval, 0.9)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Question-level holdout fraction for honest validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Avoid accidentally training on an older answer_ranker.json.
    os.environ.setdefault("NLP_ANSWER_RANK_MODE", "off")

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "nlp" / "src"))
    from nlp_manager import NLPManager  # noqa: PLC0415

    print(f"Loading AE evaluator from {args.ae_model_path} ...", flush=True)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_tokenizer = AutoTokenizer.from_pretrained(str(args.ae_model_path))
    ae_model = AutoModelForSequenceClassification.from_pretrained(
        str(args.ae_model_path)
    ).to(ae_device).eval()
    print(
        f"AE model loaded on {ae_device}: "
        f"{sum(p.numel() for p in ae_model.parameters()):,} params",
        flush=True,
    )

    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]

    # Question-level train/val split — never mix candidates from one question.
    rng = random.Random(args.seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    n_val = max(int(round(len(idxs) * args.val_fraction)), 20)
    val_set = set(idxs[:n_val])
    print(f"Split: {len(idxs) - n_val} train / {n_val} val questions", flush=True)

    manager = NLPManager()
    print(f"Loading corpus from {args.docs} ...", flush=True)
    manager.load_corpus(_load_docs(args.docs))

    print("Generating candidates ...", flush=True)
    per_question: dict[int, list] = {}
    pending_triples: list[tuple[int, int, str, str, str]] = []  # (idx, cand_i, q, gold, cand)
    for idx, row in enumerate(rows):
        question = row.get("question", "")
        gold = row.get("answer", "") or ""
        retrieved, doc_candidates = manager._retrieve(question, 30)  # noqa: SLF001
        reranked = manager._rerank(question, retrieved)  # noqa: SLF001
        documents = manager._top_doc_ids(  # noqa: SLF001
            reranked, fallback=retrieved, doc_fallback=doc_candidates
        )
        candidates = manager._answer_candidates(question, reranked, documents)  # noqa: SLF001
        if not candidates:
            continue
        per_question[idx] = []
        for c_i, cand in enumerate(candidates):
            feats = manager._candidate_features(question, cand, documents)  # noqa: SLF001
            per_question[idx].append({"features": feats, "text": cand.text, "label": 0.0})
            pending_triples.append((idx, c_i, question, gold, cand.text))

    print(
        f"Labelling {len(pending_triples)} candidates with ModernBERT AE "
        f"(threshold {args.ae_threshold}) ...",
        flush=True,
    )
    ae_labels = _ae_label_batch(
        [(t[2], t[3], t[4]) for t in pending_triples],
        ae_model, ae_tokenizer, args.ae_threshold, ae_device,
    )
    for (idx, c_i, _, _, _), lbl in zip(pending_triples, ae_labels):
        per_question[idx][c_i]["label"] = lbl

    def _flatten(question_idxs: set[int]) -> tuple[list[dict], list[float], dict]:
        feats, lbls, grouped = [], [], {}
        for idx, cands in per_question.items():
            if idx not in question_idxs:
                continue
            grouped[idx] = cands
            for c in cands:
                feats.append(c["features"])
                lbls.append(c["label"])
        return feats, lbls, grouped

    train_idxs = set(per_question.keys()) - val_set
    val_idxs = set(per_question.keys()) & val_set
    train_feats, train_labels, train_grouped = _flatten(train_idxs)
    val_feats, val_labels, val_grouped = _flatten(val_idxs)

    positives_train = int(sum(train_labels))
    positives_val = int(sum(val_labels))
    print(
        f"Train: {len(train_labels)} candidates / {len(train_grouped)} questions, "
        f"AE-positive={positives_train}",
        flush=True,
    )
    print(
        f"Val:   {len(val_labels)} candidates / {len(val_grouped)} questions, "
        f"AE-positive={positives_val}",
        flush=True,
    )
    if positives_train == 0:
        raise RuntimeError("no AE-positive candidates in train; inspect candidate generation")

    bias, weights = _train_logistic(
        train_feats, train_labels, epochs=args.epochs, lr=args.lr, l2=args.l2
    )

    def _eval(grouped: dict) -> tuple[float, float]:
        oracle = 0
        top1 = 0
        for cands in grouped.values():
            oracle += int(any(c["label"] > 0.5 for c in cands))
            best = max(cands, key=lambda c: _score(c["features"], bias, weights))
            top1 += int(best["label"] > 0.5)
        n = max(len(grouped), 1)
        return oracle / n, top1 / n

    train_oracle, train_top1 = _eval(train_grouped)
    val_oracle, val_top1 = _eval(val_grouped)

    print(f"TRAIN oracle: {train_oracle:.3f}   top-1: {train_top1:.3f}")
    print(f"VAL   oracle: {val_oracle:.3f}   top-1: {val_top1:.3f}")
    print("(top-1 is the right number to compare; oracle is the ceiling if "
          "the ranker is perfect)")

    payload = {
        "bias": bias,
        "weights": weights,
        "metadata": {
            "train_rows": len(rows),
            "train_questions": len(train_grouped),
            "val_questions": len(val_grouped),
            "train_candidates": len(train_labels),
            "val_candidates": len(val_labels),
            "train_positive": positives_train,
            "val_positive": positives_val,
            "train_oracle": train_oracle,
            "train_top1": train_top1,
            "val_oracle": val_oracle,
            "val_top1": val_top1,
            "label": f"modernbert_ae_threshold_{args.ae_threshold}",
            "ae_model_path": str(args.ae_model_path),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
