"""Replay NLP answer canonicalization against saved local predictions.

This is a lightweight guardrail for `nlp/src/nlp_manager.py` changes that only
post-process answers. It loads a saved `nlp_results.json`, the matching
`nlp.jsonl`, and the local documents directory, then runs the manager's
`_canonicalize_answer` method without loading Torch models.

Usage:
  python training/nlp/replay_canonicalizer.py \
    --results data/nlp-v11-failure-pack/nlp_results.json \
    --ground data/nlp-v11-failure-pack/nlp.jsonl \
    --docs data/novice-nlp-light-20260515/novice/nlp/documents
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path


def _install_import_stubs() -> None:
    """Let us import nlp_manager on machines without the model stack."""
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.no_grad = lambda: (lambda fn: fn)
        torch.Tensor = object
        torch.device = lambda *_args, **_kwargs: None
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.empty = lambda *_args, **_kwargs: None
        sys.modules["torch"] = torch
        sys.modules["torch.nn"] = types.ModuleType("torch.nn")
        sys.modules["torch.nn.functional"] = types.ModuleType("torch.nn.functional")

    if "rank_bm25" not in sys.modules:
        rank_bm25 = types.ModuleType("rank_bm25")
        rank_bm25.BM25Okapi = object
        sys.modules["rank_bm25"] = rank_bm25

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        for name in (
            "AutoConfig",
            "AutoModelForQuestionAnswering",
            "AutoModelForSeq2SeqLM",
            "AutoModelForSequenceClassification",
            "AutoTokenizer",
        ):
            setattr(transformers, name, object)
        sys.modules["transformers"] = transformers


def _norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _bucket(answer: str, pred_docs: list[str], gt: dict) -> str:
    hit = bool(set(pred_docs[:3]) & set(gt.get("source_docs") or []))
    if not hit:
        return "retrieval_miss"
    if not answer:
        return "retrieval_hit_empty"
    gold_norm = _norm(gt.get("answer") or "")
    pred_norm = _norm(answer)
    if pred_norm == gold_norm:
        return "retrieval_hit_exact"
    if gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
        return "retrieval_hit_substr"
    return "retrieval_hit_diff"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ground", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--changed-out", type=Path)
    args = parser.parse_args()

    _install_import_stubs()
    sys.path.insert(0, str(Path("nlp/src").resolve()))
    from nlp_manager import NLPManager  # noqa: PLC0415

    preds = json.loads(args.results.read_text())
    gts = [json.loads(line) for line in args.ground.read_text().splitlines() if line]
    doc_files = sorted(args.docs.glob("*.txt"))

    manager = object.__new__(NLPManager)
    manager.doc_ids = [p.stem for p in doc_files]
    manager.documents = [p.read_text(errors="replace") for p in doc_files]
    manager.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(manager.doc_ids)}

    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    changed = []

    for pred, gt in zip(preds, gts):
        old_answer = (pred.get("answer") or "").strip()
        pred_docs = (pred.get("documents") or [])[:3]
        new_answer = manager._canonicalize_answer(  # noqa: SLF001
            gt.get("question", ""), old_answer, pred_docs
        )
        old_bucket = _bucket(old_answer, pred_docs, gt)
        new_bucket = _bucket(new_answer, pred_docs, gt)
        before[old_bucket] += 1
        after[new_bucket] += 1
        if old_answer != new_answer or old_bucket != new_bucket:
            changed.append(
                {
                    "key": gt.get("key"),
                    "before": old_bucket,
                    "after": new_bucket,
                    "question": gt.get("question"),
                    "gold_answer": gt.get("answer"),
                    "old_answer": old_answer,
                    "new_answer": new_answer,
                    "pred_docs": pred_docs,
                }
            )

    print("Before:", dict(before))
    print("After: ", dict(after))
    print(f"Changed answers: {len(changed)}")
    for row in changed[:20]:
        print()
        print(f"key={row['key']} {row['before']} -> {row['after']}")
        print(f"Q: {row['question']}")
        print(f"G: {row['gold_answer']}")
        print(f"P: {row['old_answer']}")
        print(f"N: {row['new_answer']}")

    if args.changed_out:
        with args.changed_out.open("w") as f:
            for row in changed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
