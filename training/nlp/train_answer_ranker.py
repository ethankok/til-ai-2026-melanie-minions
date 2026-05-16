"""Train the v12 lightweight NLP answer-candidate ranker.

This script runs the deployed manager against the local novice corpus, asks it
to generate its answer candidates, labels candidates that match the gold answer
under the exact/substr proxy, and fits a tiny logistic ranker over the same
features used at inference time.

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


def _is_positive(pred: str, gold: str) -> bool:
    pred_norm = _norm(pred)
    gold_norm = _norm(gold)
    if not pred_norm or not gold_norm:
        return pred_norm == gold_norm
    return pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm


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
    args = parser.parse_args()

    # Avoid accidentally training on an older answer_ranker.json.
    os.environ.setdefault("NLP_ANSWER_RANK_MODE", "off")

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "nlp" / "src"))
    from nlp_manager import NLPManager  # noqa: PLC0415

    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]

    manager = NLPManager()
    print(f"Loading corpus from {args.docs} ...", flush=True)
    manager.load_corpus(_load_docs(args.docs))

    feature_rows: list[dict[str, float]] = []
    labels: list[float] = []
    grouped: dict[int, list[tuple[dict[str, float], float, str]]] = defaultdict(list)

    for idx, row in enumerate(rows):
        question = row.get("question", "")
        retrieved, doc_candidates = manager._retrieve(question, 30)  # noqa: SLF001
        reranked = manager._rerank(question, retrieved)  # noqa: SLF001
        documents = manager._top_doc_ids(  # noqa: SLF001
            reranked, fallback=retrieved, doc_fallback=doc_candidates
        )
        candidates = manager._answer_candidates(question, reranked, documents)  # noqa: SLF001
        if not candidates:
            continue
        for cand in candidates:
            feats = manager._candidate_features(question, cand, documents)  # noqa: SLF001
            label = 1.0 if _is_positive(cand.text, row.get("answer", "")) else 0.0
            feature_rows.append(feats)
            labels.append(label)
            grouped[idx].append((feats, label, cand.text))

    positives = int(sum(labels))
    print(
        f"Generated {len(labels)} candidates from {len(grouped)} questions; "
        f"positive exact/substr proxy={positives}",
        flush=True,
    )
    if positives == 0:
        raise RuntimeError("no positive candidates; inspect candidate generation first")

    bias, weights = _train_logistic(
        feature_rows, labels, epochs=args.epochs, lr=args.lr, l2=args.l2
    )

    top1_hits = 0
    oracle_hits = 0
    for group in grouped.values():
        oracle_hits += int(any(label > 0.5 for _, label, _ in group))
        best = max(group, key=lambda item: _score(item[0], bias, weights))
        top1_hits += int(best[1] > 0.5)

    print(f"Candidate oracle proxy: {oracle_hits / max(len(grouped), 1):.3f}")
    print(f"Ranker top-1 proxy:    {top1_hits / max(len(grouped), 1):.3f}")

    payload = {
        "bias": bias,
        "weights": weights,
        "metadata": {
            "train_rows": len(rows),
            "candidate_rows": len(labels),
            "positive_rows": positives,
            "oracle_proxy": oracle_hits / max(len(grouped), 1),
            "top1_proxy": top1_hits / max(len(grouped), 1),
            "label": "exact_or_substr_proxy",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
