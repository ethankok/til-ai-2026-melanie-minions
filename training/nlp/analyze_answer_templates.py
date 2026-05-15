"""Inspect local NLP QA answer patterns.

Usage:
  python training/nlp/analyze_answer_templates.py \
    --data data/novice-nlp-light-20260515/novice/nlp/nlp.jsonl \
    --docs data/novice-nlp-light-20260515/novice/nlp/documents

This script is deliberately stdlib-only so it can run on a laptop without the
NLP image dependencies. It does not train on or ship the local novice data; it
just summarizes question/answer syntax so we can decide which deterministic
post-processing helpers are worth testing.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MONEY_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s+(?:thousand|million|billion|trillion)?\s*"
    r"(?:Phi\s+)?Credits?\b",
    re.I,
)
NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
DATE_RE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
ID_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_docs(path: Path) -> dict[str, str]:
    docs = {}
    for file in sorted(path.iterdir()):
        if file.is_file():
            docs[file.stem] = file.read_text(errors="replace")
    return docs


def norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def classify_question(question: str) -> str:
    q = question.lower()
    if "percentage point" in q or "by what percentage" in q:
        return "percentage_delta"
    if q.startswith(("how many", "what number")) or "how many" in q:
        return "count"
    if q.startswith(("how much", "what amount")) or "credits" in q:
        return "money"
    if q.startswith(("what is", "what are", "define")):
        return "what_is"
    if q.startswith(("where", "in which location")):
        return "where"
    if q.startswith(("who", "which person")):
        return "who"
    if q.startswith(("did", "does", "do ", "is ", "are ", "was ", "were ")):
        return "yes_no"
    if "average" in q or "mean" in q:
        return "average"
    if "total" in q or "sum" in q or "combined" in q:
        return "total_sum"
    if "maximum" in q or "minimum" in q or "highest" in q or "lowest" in q:
        return "max_min"
    if "year" in q and any(w in q for w in ("between", "elapsed", "after", "before", "passed")):
        return "elapsed_years"
    if "day" in q and any(w in q for w in ("between", "elapsed", "after", "before", "separate")):
        return "elapsed_days"
    if "date" in q or "time" in q or q.startswith("when"):
        return "date_time"
    return "other"


def classify_answer(answer: str) -> str:
    words = answer.split()
    if MONEY_RE.search(answer):
        return "money"
    if PERCENT_RE.search(answer):
        return "percent"
    if DATE_RE.search(answer):
        return "date"
    if NUMBER_RE.fullmatch(answer.strip()):
        return "number"
    if ID_RE.fullmatch(answer.strip()):
        return "id_or_acronym"
    if len(words) == 1:
        return "one_word"
    if len(answer) <= 20:
        return "short_phrase"
    return "long_phrase"


def qtype(question: str) -> str:
    q = question.lower().strip()
    for prefix in ("how many", "how much", "what", "which", "where", "when", "who", "why", "how", "by"):
        if q.startswith(prefix):
            return prefix
    return q.split()[0] if q.split() else ""


def literal_status(answer: str, source_text: str) -> str:
    if answer in source_text:
        return "exact_in_src"
    if answer.lower() in source_text.lower():
        return "case_insens_in_src"
    if norm(answer) in norm(source_text):
        return "punct_norm_in_src"
    return "not_literal"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    docs = load_docs(args.docs)

    by_diff = Counter(row.get("difficulty", "?") for row in rows)
    qtypes = Counter(qtype(row["question"]) for row in rows)
    answer_types = Counter(classify_answer(row["answer"]) for row in rows)
    literal = Counter()
    literal_by_diff = defaultdict(Counter)
    templates = Counter()
    template_literal = defaultdict(Counter)
    units = Counter()

    for row in rows:
        answer = row["answer"]
        source_ids = row.get("source_docs") or [row.get("document_id", "")]
        source = "\n".join(docs.get(source_id, "") for source_id in source_ids)
        status = literal_status(answer, source)
        template = classify_question(row["question"])
        literal[status] += 1
        literal_by_diff[row.get("difficulty", "?")][status] += 1
        templates[template] += 1
        template_literal[template][status] += 1
        for match in re.finditer(
            r"\b\d[\d,]*(?:\.\d+)?\s+([A-Za-z][A-Za-z -]{1,24})\b",
            answer,
        ):
            unit = match.group(1).strip().lower()
            unit = re.split(r"[,.();]", unit)[0].strip()
            if unit:
                units[unit] += 1

    print(f"rows {len(rows)} docs {len(docs)}")
    print("difficulty", by_diff)
    print("qtype", qtypes.most_common(20))
    print("answer types")
    for key, count in answer_types.most_common():
        print(f"  {key:16s} {count}")
    print("answer source literal")
    for key, count in literal.most_common():
        print(f"  {key:20s} {count}")
    print("by difficulty")
    for key, counts in sorted(literal_by_diff.items()):
        print(f"  {key:4s} {dict(counts)}")
    print("templates")
    for key, count in templates.most_common(20):
        exactish = (
            template_literal[key]["exact_in_src"]
            + template_literal[key]["case_insens_in_src"]
            + template_literal[key]["punct_norm_in_src"]
        )
        print(f"  {key:18s} {count:4d} literal {exactish:4d}")
    print("units")
    for key, count in units.most_common(25):
        print(f"  {key:24s} {count}")


if __name__ == "__main__":
    main()
