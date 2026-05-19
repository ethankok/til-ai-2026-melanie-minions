"""Replay opt-in NLP composition rules against a saved failure pack.

This is intentionally stdlib-only: it stubs the heavy model imports, imports the
real nlp_manager.py, and calls _canonicalize_answer with
NLP_COMPOSITION_MODE=conservative. Use it before any Workbench build to check
whether a rule patch moves exact/substr proxy buckets without obvious damage.

Example:

  python training/nlp/replay_composition_rules.py \
    --analysis data/nlp-v11-failure-pack/nlp_failure_analysis.jsonl \
    --docs data/novice/nlp/documents \
    --changed-out data/nlp-v11-failure-pack/composition_changed.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from collections import Counter
from pathlib import Path


def _install_import_stubs() -> None:
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.device = lambda *_args, **_kwargs: None
    torch.no_grad = lambda fn=None: (fn if fn is not None else (lambda f: f))
    torch.empty = lambda *_args, **_kwargs: []
    torch.cat = lambda seq, **_kwargs: seq
    torch.topk = lambda *_args, **_kwargs: None

    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    nn.functional = functional
    torch.nn = nn

    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object

    transformers = types.ModuleType("transformers")
    for name in (
        "AutoConfig",
        "AutoModelForQuestionAnswering",
        "AutoModelForSeq2SeqLM",
        "AutoModelForSequenceClassification",
        "AutoTokenizer",
    ):
        setattr(transformers, name, object)

    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", nn)
    sys.modules.setdefault("torch.nn.functional", functional)
    sys.modules.setdefault("rank_bm25", rank_bm25)
    sys.modules.setdefault("transformers", transformers)


def _norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.-]", "", text)
    return text


def _bucket(pred: str, gold: str, docs_hit: bool) -> str:
    if not docs_hit:
        return "retrieval_miss"
    pred_norm = _norm(pred)
    gold_norm = _norm(gold)
    if pred_norm == gold_norm:
        return "retrieval_hit_exact"
    if pred_norm and gold_norm and (pred_norm in gold_norm or gold_norm in pred_norm):
        return "retrieval_hit_substr"
    return "retrieval_hit_diff"


def _load_docs(docs_dir: Path) -> tuple[list[str], list[str]]:
    paths = sorted(docs_dir.glob("*.txt"))
    return [p.stem for p in paths], [p.read_text(errors="replace") for p in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--changed-out", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("NLP_COMPOSITION_MODE", "conservative")
    _install_import_stubs()
    sys.path.insert(0, str(Path("nlp/src").resolve()))
    from nlp_manager import NLPManager  # pylint: disable=import-error

    doc_ids, docs = _load_docs(args.docs)
    manager = NLPManager.__new__(NLPManager)
    manager.documents = docs
    manager.doc_ids = doc_ids
    manager.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    before = Counter()
    after = Counter()
    changed = []
    with args.analysis.open() as fh:
        for line in fh:
            row = json.loads(line)
            docs_hit = any(doc in set(row["pred_docs"]) for doc in row["gold_docs"])
            old_bucket = _bucket(row["pred_answer"], row["gold_answer"], docs_hit)
            new_answer = manager._canonicalize_answer(  # noqa: SLF001
                row["question"], row["pred_answer"], row["pred_docs"]
            )
            new_bucket = _bucket(new_answer, row["gold_answer"], docs_hit)
            before[old_bucket] += 1
            after[new_bucket] += 1
            if new_answer != row["pred_answer"] or new_bucket != old_bucket:
                changed.append(
                    {
                        "key": row["key"],
                        "before": old_bucket,
                        "after": new_bucket,
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "old_answer": row["pred_answer"],
                        "new_answer": new_answer,
                        "pred_docs": row["pred_docs"],
                    }
                )

    print("before:")
    for key, value in before.most_common():
        print(f"  {key:24s} {value}")
    print("after:")
    for key, value in after.most_common():
        print(f"  {key:24s} {value}")
    print(f"changed: {len(changed)}")

    if args.changed_out:
        args.changed_out.parent.mkdir(parents=True, exist_ok=True)
        with args.changed_out.open("w") as fh:
            for row in changed:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {args.changed_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
