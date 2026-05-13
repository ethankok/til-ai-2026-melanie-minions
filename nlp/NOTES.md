# NLP — notes & history

Last updated: 13 May 2026

Per-task working log for NLP (RAG question-answering). For input/output spec
see [README.md](README.md). For submission history across all tasks see
[../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`latest` — official 0.301 / 0.971 (12 May 03:23 SGT, 0 of 700 errors).**

Lexical keyword-overlap baseline. Already in the "correct JSON shape, somewhat
useful" zone — much better starting point than CV's empty-list baseline.
Most of the headroom is in retrieval quality, not plumbing.

## What our model runs on

From [src/nlp_manager.py](src/nlp_manager.py):

- Two-method interface:
  - `load_corpus(documents)` called once per round
  - `qa(question)` called per question
- **Chunking**: sentence-split each document
- **Retrieval**: lexical token-overlap (Counter intersection) with light length
  normalization, return top-scoring sentence
- **Answer extraction**: return the top sentence, truncated to 500 characters
- **No model weights**, no embeddings, no learned components

CPU-bound, near-instant per-question (speed score 0.971 confirms).

## Submission history

```text
Tag       Submitted          Score   Speed   Errors    Notes
latest    12/05 03:23        0.301   0.971   0 / 700   Sentence-split + lexical token-overlap retrieval, 500-char truncation
```

## Detailed timeline

### latest (12 May 03:23) — initial submission

Sentence-level lexical retrieval. Already scoring 0.301 with zero errors on
700 inputs. The baseline is correct-shape and useful enough that there's a
clear measurable next step.

## What needs doing

In priority order (per [../SUMMARY.md "NLP"](../SUMMARY.md)):

### 1. Better chunking — single afternoon

Sliding window of ~2–3 sentences with overlap, not single sentences. Single
sentences lose context (especially for "what was the cause of X?" style
questions where the answer spans multiple sentences). Full paragraphs dilute
retrieval scores. Two-to-three-sentence chunks with 1-sentence overlap is
the standard middle ground.

### 2. BM25 retrieval — fast win

Replace `collections.Counter` token-overlap with `rank_bm25.BM25Okapi`. BM25
adds TF-IDF weighting + length normalization that's far better than the
current Counter heuristic. Single `pip install rank_bm25` + ~20 lines of
code.

Expected gain: 0.301 → 0.45–0.55 range based on standard RAG benchmarks.

### 3. Hybrid retrieval (BM25 + dense embeddings)

Layer dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`) for hybrid
retrieval. Combine BM25 + cosine-sim scores via convex combination
(`alpha * bm25_norm + (1-alpha) * dense_sim`, alpha around 0.5).

Adds ~80 MB to the container, embedding compute is fast on CPU for the
corpus sizes we're seeing (small enough that `load_corpus` can pre-compute
everything).

Cache embeddings in `load_corpus` so `qa()` stays fast.

### 4. Better answer extraction (don't return the whole chunk)

Two options:

- **Extractive QA model**: run `distilbert-base-uncased-distilled-squad` (or
  similar) on the top-k retrieved chunks, return the highest-confidence
  span. Adds ~250 MB to container, real model compute per question — may
  hurt speed score.
- **Sentence-of-best-chunk**: pick the sentence within the retrieved chunk
  with highest overlap / embedding similarity to the question. Much cheaper,
  partial gain.

### 5. Cross-encoder re-ranker (stretch)

Re-rank top-k with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
Slower but usually a clear quality win.

### 6. Watch the answer length cap

Current truncates at 500 chars. Verify the eval doesn't penalize overlong or
under-long answers (different scorers weight these differently).

## State

NLP at 0.301 is the second-biggest absolute opportunity after AE (CV is
absolute zero but has fewer "free" wins; NLP has standard RAG playbook).

A BM25-only upgrade is a **single afternoon of work for ~+0.15–0.25 raw
score**, which translates to **+0.03–0.05 blended qualifier score** at the
20% NLP weight.

## Reproducibility / pointers

- Manager source: [src/nlp_manager.py](src/nlp_manager.py)
- HTTP server (don't edit): [src/nlp_server.py](src/nlp_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#nlp](../SUMMARY.md)
