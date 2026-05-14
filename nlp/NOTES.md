# NLP — notes & history

Last updated: 14 May 2026 13:35 SGT

Per-task working log for NLP (RAG question-answering). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#nlp).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`v4-dict-id` — official 0.483 / 0.888 (14 May 13:29 SGT, 0 of 700 errors).** NEW HIGH on the post-wipe leaderboard. Same image bits as `v3-id-parse`; the 0.000 → 0.483 jump came entirely from Ryan fixing his eval-server bug (it had been sending plain strings instead of `{"id":"DOC-XXXX","document":"..."}` dicts). Our defensive parser caught the dict shape on first try once the real format started flowing.

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

## Doc-ID format (resolved 14 May 09:33 SGT)

Initially the blank `nlp_manager.py` template typed `load_corpus(documents: list[str])` and the wiki showed plain strings, which led `v2-hybrid-rag` and `v3-id-parse` to both score 0.000 on the cloud (`0 / 700` errors but every retrieval missed). Investigation showed:

- `/home/jupyter/novice/nlp/documents/` has **296 files spanning DOC-0001..DOC-0340 with 44 gaps** in the ID range.
- `nlp.jsonl` `source_docs` uses real filename IDs.
- Document content has no DOC-XXXX prefix.

Ryan then confirmed on hackoverflow that his eval server had a bug: it was supposed to send dicts, not plain strings. After his fix, the real format is:

```json
{
  "instances": [{
    "documents": [
      {"id": "DOC-0001", "document": "Text of document one."},
      {"id": "DOC-0002", "document": "Text of document two."}
    ]
  }]
}
```

The template signature was also updated to `load_corpus(documents: list[dict[str, str]])`. Our `_parse_doc_payload` was already defensive across three encodings, including the dict shape with `id`+`document` keys — so `v4-dict-id` (same image as `v3-id-parse`, re-tagged) caught the new format immediately and scored `0.483 / 0.888` on the next submission.

## Implementation

End-to-end pipeline in [src/nlp_manager.py](src/nlp_manager.py) and [src/nlp_server.py](src/nlp_server.py):

1. **Server.** Matches Ryan's pre-written async + poll pattern verbatim — `{"status":"loading"}` then `{"status":"loaded"}` on the poll endpoint; per-question response is `{"documents":[...], "answer":"..."}`.
2. **Doc-ID parser.** `_parse_doc_payload` handles four encodings: (1) dict shape with `id` key (cloud uses this), (2) first-line `DOC-XXXX` stripped from body, (3) `DOC-XXXX` within the first 80 chars, (4) positional fallback `DOC-{i+1:04d}`. Defensive; cost is ~10 lines.
3. **Paragraph-aware chunking.** Each document is split on blank lines into paragraphs first. Short paragraphs (≤3 sentences) become a single chunk; longer ones get a 3-sentence sliding window with 1-sentence overlap *within the paragraph*. Falls back to whole-document sliding-window when there are no blank lines. Each chunk records its parent doc index for ID emission.
4. **Hybrid retrieval.** BM25Okapi over tokenized chunks ⊕ dense cosine over `BAAI/bge-small-en-v1.5` embeddings (CLS-pooled, L2-normalized; query gets BGE's English search prefix). Per-query z-score normalise then sum. Top-30 chunks → reranker.
5. **Cross-encoder rerank.** `BAAI/bge-reranker-base` scores `(question, chunk)` pairs. Top-10 → QA.
6. **Top-3 doc IDs with BM25 backfill.** Walks reranked passages collecting unique parent doc IDs. If the reranker concentrated on <3 unique docs (common when one document has many highly-relevant chunks), backfills from the un-reranked hybrid list. Protects retrieval recall — the new eval gates every case on retrieval, so empty doc slots are pure waste.
7. **Extractive QA (batched).** `deepset/roberta-base-squad2` over the top reranked chunks, **all features tokenized + forwarded in a single batched call** (subject to `QA_BATCH=16`). The previous per-context loop has been replaced by `[question]*N` + `[ctx1,...,ctxN]` tokenization with `overflow_to_sample_mapping` to recover which feature came from which context. Span selection logic is unchanged.
8. **Low-confidence fallback.** If SQuAD2's best span is <3 chars or <2 words, return the highest token-overlap sentence from the top 3 contexts instead. The AE 0.9 threshold rarely passes single-token guesses; a concise source sentence has a real shot.
9. **Speed.** GPU half-precision on all three models. Cloud speed `0.888` on `v4-dict-id`; batching should improve this further on the next submission.

Weights baked into the image via [download_models.py](download_models.py). Container runs offline (`TRANSFORMERS_OFFLINE=1`). `NLP_MODEL_DIR=/workspace/models`.

Track: **Novice** — no L4/L5 handling code.

## Local performance

With the upstream `test_nlp.py` (which now sends `{"id": doc_file.stem, "document": content}` per doc):

```text
Answer Equivalence Evaluation Summary:
  n=883, equiv_rate=0.678, mean_prob=0.521,
  equivalent_count=598.8, not_equivalent_count=284.2
NLP RAG QA Accuracy: 0.678
```

`equivalent_count` is fractional because retrieval-only successes earn `RETRIEVAL_ONLY_SCORE=0.4` partial credit. The 0.678 is a blend of full credit (1.0), retrieval-only (0.4), and zero. Net: pipeline correctly retrieves + extracts most of the time.

Local 0.678 → cloud 0.483 is a `~0.20` gap, consistent with AE/CV local→official gaps on this competition. Held-out corpus is likely distribution-shifted (different topic mix or document length) but otherwise the pipeline transfers cleanly.

## Submission history

```text
Tag           Submitted          Score   Speed   Errors    Local        Notes
latest        12/05 03:23        0.301   0.971   0 / 700   —            OLD EVAL; pre-wipe; lexical baseline (no longer on leaderboard)
v2-hybrid-rag 14/05 ~04:00       0.000   ~       0 / 700   —            NEW EVAL. Hybrid stack + positional IDs. 0.0 — eval server bug; was sending plain strings, not dicts (see v4-dict-id)
v3-id-parse   14/05 05:33        0.000   0.888   0 / 700   0.678        Same hybrid stack + defensive parser. Still 0.0 because eval bug persisted; local 0.678 with prefix-patched test confirmed model was fine
v4-dict-id    14/05 13:29        0.483   0.888   0 / 700   0.678        After Ryan fixed eval to send dicts. SAME IMAGE as v3-id-parse (just re-tagged); recovery to NEW HIGH on post-wipe leaderboard
v5-multi      (pending)          ?       ?       ?         ?            Para-aware chunking + batched SQuAD2 + BM25 doc-diversity backfill + low-conf sentence fallback. NOT yet submitted; run til test first
```

## What's in `v5-multi` (built but not yet submitted)

This bundles Codex's recommended steps 1, 2, 4 (diversity), and 6 in one image. They were combined rather than shipped sequentially because the changes are independent and small; the error-report script lets us still attribute headroom afterward.

1. **Batched SQuAD2 forward pass.** `_extract_answer` now tokenizes all `[question]*N + [contexts]` at once and forwards in batches of `QA_BATCH=16`. Functionally identical span-selection logic to before; should reduce QA latency 50–80% (one-or-two forwards per question vs N). Cloud speed `0.888` → expected ~0.92+.
2. **Paragraph-aware chunking.** `_chunk_document` splits on blank lines first, then sentence-windows within each paragraph. Short paragraphs emit as a single chunk so cross-sentence answer spans aren't fragmented.
3. **BM25 backfill for doc diversity.** `_top_doc_ids` now takes both the reranker output and the un-reranked hybrid list; if the reranker concentrated on <3 unique docs, fills the remaining slots from the hybrid list. Protects against leaving doc-recall on the table.
4. **Low-confidence sentence fallback.** When SQuAD2's best span is <3 chars or <2 words, returns the highest token-overlap sentence from the top contexts instead. The AE 0.9 threshold almost never passes single-token guesses; a concise source sentence has a real shot.

What's **deliberately not changed** in `v5-multi`:
- `TOP_K_RETRIEVE` / `TOP_K_RERANK` / `TOP_DOCS_RETURNED` constants. Codex suggested bumping them; we'll do that in `v6-retrieval-tune` only if the error report shows retrieval is the binding constraint.
- Model choices (`bge-small`, `bge-reranker-base`, `roberta-base-squad2`). Save bigger models for `v7-/v8-` after diagnosing what bucket dominates.
- Server (`nlp_server.py`). Ryan's pre-written verbatim.

## Local diagnostic: [error_report.py](error_report.py)

Run after `til test` to bucket where score is being lost:

```bash
python nlp/error_report.py /home/jupyter/melanie-minions/nlp_results.json /home/jupyter/novice/nlp/nlp.jsonl
```

Buckets:
- `retrieval_miss` — none of the top-3 docs match `source_docs`. Cloud scores 0.0 here.
- `retrieval_hit_exact` / `retrieval_hit_substr` — answer normalised-equal or one-is-substring-of-other. Almost certainly full credit.
- `retrieval_hit_diff` — different answer; the AE 0.9 model decides. The unknown bucket; QA / chunking improvements move score from here to "exact/substr".
- `retrieval_hit_empty` — locked at 0.4 partial credit; any non-empty answer is strictly better.

Use the script to decide which lever to pull for `v6`:
- If `retrieval_miss` dominates → tune retrieval (bigger embedder, asymmetric BM25⊕dense weighting, larger `TOP_K_RETRIEVE`).
- If `retrieval_hit_diff` dominates → tune QA (bigger model, better chunking, span-selection refinements).
- If `retrieval_hit_empty` is large → the low-conf fallback in `v5-multi` should already help; check the bucket on `v5-multi` results.

The script prints estimated cloud-score lower/upper bounds; compare to `til test`'s actual `equiv_rate` to triangulate where the AE model is finding extra credit beyond exact/substr.

## Next priority

1. **Submit `v5-multi`.** Rebuild on Workbench (`til build nlp v5-multi`), run `til test nlp v5-multi`, verify local accuracy is **≥ 0.678** (the v4 baseline; if it drops by more than 0.005 that's a regression somewhere in the new code), submit.
2. **Run `error_report.py` on v4 results** *while* v5 is rebuilding. That gives us the bucket distribution at the 0.483 cloud baseline. The same script run on v5 results tells us which buckets the new changes moved.
3. **`v6` based on what the report shows** — retrieval tuning (`TOP_K_RETRIEVE=50`, asymmetric weighting), or larger embedder/reranker, or larger QA. Decision driven by buckets, not guesswork.
4. **`v7+` only if `v6` is still leaving headroom.** Bigger models (`bge-base`, `roberta-large-squad2`) cost build time and container size; reserve for when the bucket diagnostic justifies them.

## What to edit, what not to

- [src/nlp_manager.py](src/nlp_manager.py) — main logic. All next-priority work lives here.
- [src/nlp_server.py](src/nlp_server.py) — Ryan's pre-written verbatim; don't drift from upstream.
- [Dockerfile](Dockerfile), [requirements.txt](requirements.txt), [download_models.py](download_models.py) — packaging.
- `test/test_nlp.py` is the **updated upstream** (sends dicts with `id` + `document`). No local patch needed any more.

## Reproducibility / pointers

- Manager source: [src/nlp_manager.py](src/nlp_manager.py)
- HTTP server: [src/nlp_server.py](src/nlp_server.py)
- Weight bundler: [download_models.py](download_models.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Input/output spec: [README.md](README.md)
- Strategic context: [../SUMMARY.md#nlp](../SUMMARY.md)
