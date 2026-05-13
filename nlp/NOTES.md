# NLP — notes & history

Last updated: 14 May 2026

Per-task working log for NLP (RAG question-answering). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#nlp).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`latest` — official 0.301 / 0.971 (12 May 03:23 SGT, 0 of 700 errors)** — scored under the OLD eval; the response shape no longer matches the FINAL eval rolled out 14 May (see below). The next submission is the first one that runs against the new evaluator.

## New eval (FINAL — pinned 14 May, no more changes)

Per organisers, the NLP evaluator is now frozen in this state:

- Submit top-3 document IDs in `documents`. If any of the first 3 matches the target, retrieval succeeds. >3 → only first 3 considered; <3 → only those.
- Answer cleaned of non-printable chars and truncated to 64 tokens server-side.
- Answer-equivalence threshold raised from 0.5 to **0.9** (same ModernBERT weights).
- Retrieval ✗ → 0.0 outright. Retrieval ✓ + answer ✗ → 0.4. Retrieval ✓ + answer ✓ → 1.0.

Response shape (per-question):
```json
{"documents": ["DOC-0001"], "answer": "This is my answer."}
```
Corpus-load response: `{"predictions": [{"status": "loaded"}]}`.

## Implementation (`v2-hybrid-rag`, unsubmitted)

End-to-end overhaul in [src/nlp_manager.py](src/nlp_manager.py) and [src/nlp_server.py](src/nlp_server.py):

1. **Schema fix (mandatory).** Server now returns `{"status":"loaded"}` on load and `{documents, answer}` per question.
2. **Document IDs.** Positional, 1-indexed, zero-padded 4-digit (`DOC-{i+1:04d}` for the i-th doc in the received list). The endpoint only ever sees raw strings, so positional is the only viable scheme; the evaluator must mirror it.
3. **Chunking.** 3-sentence sliding window with 1-sentence overlap; each chunk carries its parent doc index.
4. **Hybrid retrieval.** BM25Okapi over tokenized chunks ⊕ dense cosine over `BAAI/bge-small-en-v1.5` embeddings (CLS-pooled, L2-normalized; query gets BGE's English search prefix). Per-query z-score normalise then sum. Top-30 chunks → reranker.
5. **Cross-encoder rerank.** `BAAI/bge-reranker-base` scores `(question, chunk)` pairs. Top-10 → QA. Top-3 unique parent docs become the response `documents`.
6. **Extractive QA.** `deepset/roberta-base-squad2` over each of the top reranked chunks; pick the highest-scoring span ≤64 tokens. Extractive (vs generative) is the right play because (a) Clairos has fictional proper nouns pretrained LMs don't know, and (b) the 0.9 ModernBERT threshold rewards near-verbatim spans from the source.
7. **Speed.** GPU half-precision on all three models. Batched embed/rerank/QA. Empty corpus / empty tokens guarded.

Weights baked into the image via [download_models.py](download_models.py). Container runs offline (`TRANSFORMERS_OFFLINE=1` set in the Dockerfile after weights are downloaded). `NLP_MODEL_DIR=/workspace/models`.

Track: **Novice** — no L4/L5 handling code (those don't appear at Novice level).

## Submission history

```text
Tag             Submitted          Score   Speed   Errors    Notes
latest          12/05 03:23        0.301   0.971   0 / 700   Lexical, old eval — schema no longer matches
v2-hybrid-rag   (pending)          ?       ?       ?         Hybrid BM25+BGE + reranker + RoBERTa-SQuAD2
```

## Expected impact

- Schema fix alone: prevents 0.0 from shape mismatch on the new eval.
- Hybrid + reranker top-3 doc recall should push retrieval success high (≥0.85 is plausible given 3 picks); that lifts the floor to ~0.34 even with zero answer credit.
- Extractive answers with 0.9 threshold: harder to estimate; bound is the answer-equivalence rate.

If shipped scoring is poor, the levers in order are: (a) better chunking (paragraph-aware), (b) increase retrieve-K before rerank, (c) sentence-of-best-chunk fallback when SQuAD2 returns low confidence, (d) try `bge-base-en-v1.5` for dense, (e) try `bge-reranker-large`.

## What to edit, what not to

- [src/nlp_manager.py](src/nlp_manager.py) — main logic.
- [src/nlp_server.py](src/nlp_server.py) — schema contract; only edit if the spec changes again (organisers said it won't).
- [Dockerfile](Dockerfile), [requirements.txt](requirements.txt), [download_models.py](download_models.py) — packaging.

## Reproducibility / pointers

- Manager source: [src/nlp_manager.py](src/nlp_manager.py)
- HTTP server: [src/nlp_server.py](src/nlp_server.py)
- Weight bundler: [download_models.py](download_models.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#nlp](../SUMMARY.md)
