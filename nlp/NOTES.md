# NLP — notes & history

Last updated: 14 May 2026 05:48 SGT

Per-task working log for NLP (RAG question-answering). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#nlp).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`v3-id-parse` — official 0.000 / 0.888 (14 May 05:33 SGT, 0 of 700 errors).** Pipeline runs cleanly on the cloud but every retrieval misses because our doc-ID convention does not match the cloud's. **Blocked on Ryan's hackoverflow answer.** Local test scores 0.678 with `DOC-XXXX\n` prepended to each document (see "Local verification" below), so the retrieval + rerank + QA model itself is sound — only the ID-extraction step is wrong.

## New eval (FINAL — pinned 14 May)

Per organisers, the NLP evaluator is now frozen in this state:

- Submit top-3 document IDs in `documents`. If any of the first 3 matches the target, retrieval succeeds. >3 → only first 3 considered; <3 → only those.
- Answer cleaned of non-printable chars and truncated to 64 tokens server-side.
- Answer-equivalence threshold raised from 0.5 to **0.9** (same ModernBERT weights).
- Retrieval ✗ → 0.0 outright. Retrieval ✓ + answer ✗ → 0.4. Retrieval ✓ + answer ✓ → 1.0.

Old NLP leaderboard was wiped on rollout.

Response shape (per-question):
```json
{"documents": ["DOC-0001"], "answer": "This is my answer."}
```
Corpus-load response: `{"predictions": [{"status": "loaded"}]}`.

## Doc-ID mystery (the current blocker)

The blank `nlp_manager.py` template Ryan provided types `load_corpus(documents: list[str])` — plain strings, no IDs. The wiki shows the same. But:

- `/home/jupyter/novice/nlp/documents/` has **296 files spanning DOC-0001..DOC-0340 with 44 gaps** in the ID range.
- `nlp.jsonl` `source_docs` uses **real filename IDs** (e.g., `DOC-0193.txt` is at sorted position 166 and is the source for question 0).
- Document content has **no DOC-XXXX prefix** anywhere — verified by reading multiple files.

Our `v3-id-parse` defensive parser handles three plausible cloud encodings: (A) `DOC-XXXX\n` prefix in content, (B) dict shape with `"id"` field, (C) positional fallback. The 0.000 cloud score proves the cloud is **not** in worlds A or B. Worlds we don't yet handle:

- **D — cloud uses sorted-filename IDs.** The eval reads files sorted, then `source_docs` references real filename IDs that happen to have gaps. Our positional `DOC-{i+1:04d}` doesn't account for the gaps. Possible fix: the cloud's held-out corpus might also have gaps, in which case we can't know IDs from content alone — need a side channel.
- **E — cloud sends a side-channel.** Maybe headers, query params, separate endpoint, or paired with the question. Not visible in the wiki spec.
- **F — cloud renumbers consecutively.** Positional would then be correct, but our 0.000 score rules this out (since we tried positional fallback and it failed).

Open question posted to the `ryan helps everyone with the NLP change` Discord thread.

## Implementation (`v3-id-parse`)

End-to-end overhaul in [src/nlp_manager.py](src/nlp_manager.py) and [src/nlp_server.py](src/nlp_server.py):

1. **Server.** Matches Ryan's pre-written async + poll pattern verbatim — `{"status":"loading"}` then `{"status":"loaded"}` on the poll endpoint; per-question response is `{"documents":[...], "answer":"..."}`.
2. **Doc-ID parser.** `_parse_doc_payload` handles three encodings: dict shape with `id` key, first-line `DOC-XXXX` (stripped from body), `DOC-XXXX` within the first 80 chars, and positional fallback `DOC-{i+1:04d}`. Empirically the cloud uses none of A/B/positional — see "Doc-ID mystery" above.
3. **Chunking.** 3-sentence sliding window with 1-sentence overlap; each chunk carries its parent doc index.
4. **Hybrid retrieval.** BM25Okapi over tokenized chunks ⊕ dense cosine over `BAAI/bge-small-en-v1.5` embeddings (CLS-pooled, L2-normalized; query gets BGE's English search prefix). Per-query z-score normalise then sum. Top-30 chunks → reranker.
5. **Cross-encoder rerank.** `BAAI/bge-reranker-base` scores `(question, chunk)` pairs. Top-10 → QA. Top-3 unique parent docs become the response `documents`.
6. **Extractive QA.** `deepset/roberta-base-squad2` over each of the top reranked chunks; pick the highest-scoring span ≤64 tokens. Extractive (vs generative) is the right play because (a) Clairos has fictional proper nouns pretrained LMs don't know, and (b) the 0.9 ModernBERT threshold rewards near-verbatim spans from the source.
7. **Speed.** GPU half-precision on all three models. Cloud speed `0.888` confirms the inference budget is fine.

Weights baked into the image via [download_models.py](download_models.py). Container runs offline (`TRANSFORMERS_OFFLINE=1`). `NLP_MODEL_DIR=/workspace/models`.

Track: **Novice** — no L4/L5 handling code.

## Local verification (workaround for unknown cloud ID format)

`test/test_nlp.py` was patched locally to prepend `DOC-XXXX\n` (from `doc_file.stem`) to each document's content and to sort the glob output. This lets the model see the real filename ID via our parser's "world A" branch. With that in place:

```text
Answer Equivalence Evaluation Summary:
  n=883, equiv_rate=0.678, mean_prob=0.521,
  equivalent_count=598.8, not_equivalent_count=284.2
NLP RAG QA Accuracy: 0.678
```

`equivalent_count` is fractional because retrieval-only successes earn `RETRIEVAL_ONLY_SCORE=0.4` partial credit. The 0.678 number is therefore a blend of full credit (1.0), partial credit (0.4), and zeros. Net: pipeline is correctly retrieving + extracting most of the time when IDs are available.

This patch is **local-only**. The cloud format is unknown until Ryan answers.

## Submission history

```text
Tag           Submitted          Score   Speed   Errors    Local        Notes
latest        12/05 03:23        0.301   0.971   0 / 700   —            OLD EVAL; pre-wipe; lexical baseline
v2-hybrid-rag 14/05 ~04:00       0.000   ~       0 / 700   —            NEW EVAL. Hybrid stack + positional IDs. Failed because positional misaligns with cloud filename IDs (44 gaps)
v3-id-parse   14/05 05:33        0.000   0.888   0 / 700   0.678 (1)    NEW EVAL. Same hybrid stack + defensive parser. Proved cloud is NOT in world A (prefix) or B (dict) — both branches and positional fallback failed
```

(1) Local test with `DOC-XXXX\n` prefix patch. Cloud format unconfirmed.

## What needs doing

1. **WAIT FOR RYAN.** Doc-ID convention is the only thing blocking; everything else (retrieval, rerank, QA) works. Don't burn submissions guessing.
2. Once answered, one targeted patch to `_parse_doc_payload` or `load_corpus` and a rebuild ships a working NLP.
3. Once a non-zero baseline is back, then push retrieval + QA quality:
   - **Batch SQuAD2.** `QA_BATCH=16` is defined but unused — `_extract_answer` processes one context at a time. Batching the forward pass over top-10 reranked chunks frees latency budget.
   - **Better chunking.** Paragraph-aware splitting (split on `\n\n` before sentence-windowing).
   - **Bigger QA model.** `deepset/roberta-large-squad2`.
   - **Bigger embedder/reranker.** `bge-base-en-v1.5`, `bge-reranker-large`.

## What to edit, what not to

- [src/nlp_manager.py](src/nlp_manager.py) — main logic. The next patch lives here once Ryan answers.
- [src/nlp_server.py](src/nlp_server.py) — Ryan's pre-written verbatim; don't drift from upstream.
- [Dockerfile](Dockerfile), [requirements.txt](requirements.txt), [download_models.py](download_models.py) — packaging.
- `test/test_nlp.py` has a local-only prefix patch for verification (sort + `f"{doc_file.stem}\n{content}"`). Keep it locally; do not commit (it doesn't reflect cloud reality).

## Reproducibility / pointers

- Manager source: [src/nlp_manager.py](src/nlp_manager.py)
- HTTP server: [src/nlp_server.py](src/nlp_server.py)
- Weight bundler: [download_models.py](download_models.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#nlp](../SUMMARY.md)
