"""Bucket local NLP results to characterise where score is being lost.

Reads the predictions saved by `test/test_nlp.py` (default
/home/jupyter/<team>/nlp_results.json) and the ground-truth jsonl
(default /home/jupyter/<track>/nlp/nlp.jsonl), then prints:

  - Bucket counts:
      retrieval_miss              -> 0.0 on the cloud
      retrieval_hit_exact         -> almost certainly full credit
      retrieval_hit_substr        -> probably full credit (subset/superset)
      retrieval_hit_diff          -> uncertain; AE model decides
      retrieval_hit_empty         -> 0.4 (retrieval-only) on the cloud
  - Per-difficulty hit rate
  - An estimated score using exact/substr as the "full credit" proxy

The estimate is a lower bound: the AE model at 0.9 threshold can mark
extra "diff" cases as equivalent that exact/substr misses. Compare the
estimate to the actual `equiv_rate` from `til test` to see whether your
answer-equivalence headroom is in the "diff" bucket (worth improving QA)
or already nearly exhausted by exact/substr (push retrieval instead).

Usage:
    python nlp/error_report.py                                   # uses TEAM_NAME/TEAM_TRACK env
    python nlp/error_report.py path/to/results.json
    python nlp/error_report.py path/to/results.json path/to/nlp.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _norm(s: str) -> str:
    """Lightweight normalization for approximate equality."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _bucket(pred: dict, gt: dict) -> str:
    pred_docs = (pred.get("documents") or [])[:3]
    pred_ans = (pred.get("answer") or "").strip()
    gt_docs = set(gt.get("source_docs") or [])
    gt_ans_norm = _norm(gt.get("answer") or "")
    pred_ans_norm = _norm(pred_ans)

    retrieval_hit = bool(set(pred_docs) & gt_docs)
    if not retrieval_hit:
        return "retrieval_miss"
    if not pred_ans:
        return "retrieval_hit_empty"
    if pred_ans_norm == gt_ans_norm:
        return "retrieval_hit_exact"
    if gt_ans_norm and (gt_ans_norm in pred_ans_norm or pred_ans_norm in gt_ans_norm):
        return "retrieval_hit_substr"
    return "retrieval_hit_diff"


def main(results_path: Path, jsonl_path: Path) -> int:
    preds = json.loads(results_path.read_text())
    with open(jsonl_path) as f:
        gts = [json.loads(line) for line in f if line.strip()]

    if len(preds) != len(gts):
        print(
            f"warning: length mismatch — {len(preds)} predictions vs {len(gts)} "
            f"ground-truth rows; pairing by order",
            file=sys.stderr,
        )
    n = min(len(preds), len(gts))
    if n == 0:
        print("no predictions to score", file=sys.stderr)
        return 1

    buckets: Counter[str] = Counter()
    by_difficulty: dict[str, Counter[str]] = defaultdict(Counter)
    miss_with_answer = 0
    miss_without_answer = 0
    answer_lengths: list[int] = []

    for pred, gt in zip(preds[:n], gts[:n]):
        b = _bucket(pred, gt)
        buckets[b] += 1
        by_difficulty[gt.get("difficulty", "?")][b] += 1
        ans = (pred.get("answer") or "").strip()
        answer_lengths.append(len(ans))
        if b == "retrieval_miss":
            if ans:
                miss_with_answer += 1
            else:
                miss_without_answer += 1

    hit = sum(v for k, v in buckets.items() if k.startswith("retrieval_hit"))
    miss = buckets["retrieval_miss"]
    full_proxy = buckets["retrieval_hit_exact"] + buckets["retrieval_hit_substr"]
    partial = buckets["retrieval_hit_diff"] + buckets["retrieval_hit_empty"]

    print("=" * 70)
    print(f"NLP error report — n={n}")
    print(f"  results : {results_path}")
    print(f"  ground  : {jsonl_path}")
    print("=" * 70)
    print()
    print("Bucket distribution:")
    order = [
        ("retrieval_miss",        "no retrieval -> 0.0"),
        ("retrieval_hit_exact",   "norm-equal answer -> ~1.0"),
        ("retrieval_hit_substr",  "answer subset/superset -> ~1.0"),
        ("retrieval_hit_diff",    "different answer -> AE-dependent"),
        ("retrieval_hit_empty",   "empty answer -> 0.4 (retrieval-only)"),
    ]
    for key, note in order:
        v = buckets[key]
        pct = v / n
        print(f"  {key:25s} {v:4d}  ({pct:6.1%})  {note}")
    print()
    print(f"Retrieval hit rate:  {hit / n:.1%}   ({hit}/{n})")
    print(f"Retrieval miss rate: {miss / n:.1%}   ({miss}/{n})")
    print(f"  of misses: {miss_with_answer} returned an answer, "
          f"{miss_without_answer} returned empty")
    print()
    if answer_lengths:
        al = sorted(answer_lengths)
        print(
            f"Answer length (chars): min={al[0]} "
            f"p50={al[len(al)//2]} "
            f"mean={sum(al)/len(al):.0f} "
            f"max={al[-1]}"
        )
    print()
    print("By difficulty:")
    for diff in sorted(by_difficulty):
        sub = by_difficulty[diff]
        total_d = sum(sub.values())
        hits_d = sum(v for k, v in sub.items() if k.startswith("retrieval_hit"))
        exact_d = sub["retrieval_hit_exact"]
        print(
            f"  {diff:4s}  n={total_d:4d}  "
            f"hit={hits_d/total_d:6.1%}  "
            f"exact={exact_d/total_d:6.1%}"
        )
    print()

    est_lower = (full_proxy + 0.4 * partial) / n
    est_upper = (hit + 0.0 * miss) / n   # if AE marked EVERY hit as equivalent
    print("Estimated cloud score range (using exact+substr as proxy for full credit):")
    print(f"  lower bound ~ {est_lower:.3f}   (exact/substr only = full credit)")
    print(f"  upper bound ~ {est_upper:.3f}   (every retrieval hit = full credit)")
    print()
    print("Compare to the equiv_rate printed by `til test` to know where the")
    print("answer-equivalence headroom is:")
    print("  - if equiv_rate is close to the lower bound -> retrieval is the lever")
    print("  - if equiv_rate is well above lower -> QA is finding extra hits in")
    print("    the 'diff' bucket; pushing QA quality / chunking helps more")
    print("  - if equiv_rate is close to upper -> the model is essentially at")
    print("    the ceiling for current retrieval; tune retrieval to lift it")
    print()

    return 0


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    team_name = os.getenv("TEAM_NAME", "")
    team_track = os.getenv("TEAM_TRACK", "novice")
    default_results = Path(f"/home/jupyter/{team_name}/nlp_results.json")
    default_jsonl = Path(f"/home/jupyter/{team_track}/nlp/nlp.jsonl")

    results = Path(args.results) if args.results else default_results
    jsonl = Path(args.jsonl) if args.jsonl else default_jsonl

    if not results.exists():
        print(f"error: results file not found: {results}", file=sys.stderr)
        sys.exit(2)
    if not jsonl.exists():
        print(f"error: ground-truth file not found: {jsonl}", file=sys.stderr)
        sys.exit(2)
    return results, jsonl


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", nargs="?", help="path to nlp_results.json")
    p.add_argument("jsonl", nargs="?", help="path to nlp.jsonl (ground truth)")
    args = p.parse_args()
    r, j = _resolve_paths(args)
    sys.exit(main(r, j))
