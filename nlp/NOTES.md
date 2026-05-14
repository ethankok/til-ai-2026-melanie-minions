# NLP — notes & history

Last updated: 14 May 2026 19:50 SGT

Per-task working log for NLP (RAG question-answering). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#nlp).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`v5c-no-para` — official 0.483 / 0.912 (14 May 19:44 SGT, 0 of 700 errors).** New blended high (0.590 vs v4-dict-id's 0.584, +0.006). Same retrieval accuracy as v4 but with batched SQuAD2 + BM25 doc-diversity backfill — the additive parts of the v5 stack after paragraph chunking was identified and reverted as a regressor.

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
v5-multi      (not shipped)      —       —       —         0.628        Para-aware chunking + batched SQuAD2 + BM25 backfill + low-conf fallback. Local REGRESSED -0.050 vs v4; fallback was firing on every single-word answer. NOT submitted
v5b-no-fallback 14/05 19:10      0.456   0.916   0 / 700   0.674        Removed fallback; kept para chunking + batched SQuAD2 + BM25 backfill. Local OK (within 0.005 of v4), but cloud REGRESSED -0.027 vs v4 (speed +0.028 but blended worse -0.013). Para chunking is the suspect — local→cloud gap got bigger
v5c-no-para   14/05 19:44        0.483   0.912   0 / 700   0.678        SHIPPED. Reverted paragraph chunking; kept batched SQuAD2 + BM25 backfill. Score matches v4 exactly + speed kept the +0.024 gain from batching → blended high 0.590 (vs v4 0.584). Confirmed paragraph chunking was the cloud regressor
```

Paragraph chunking is confirmed the v5b regressor; don't revisit on this corpus. v5c is a clean baseline for v6-roberta-large.

## `v5-multi` REGRESSED locally (NOT submitted) — diagnosed

Built `v5-multi` with four changes (paragraph chunking + batched SQuAD2 + BM25 doc-diversity backfill + low-conf sentence fallback). Local `equiv_rate` 0.678 → **0.628** (−0.050). Error-bucket diagnostic identified the cause immediately:

| Bucket | v4 (cloud 0.483) | v5-multi local | Δ |
|---|---:|---:|---:|
| retrieval_miss | 40 (4.5%) | 45 (5.1%) | +5 |
| retrieval_hit_exact | 183 (20.7%) | 105 (11.9%) | **−78** |
| retrieval_hit_substr | 256 (29.0%) | 231 (26.2%) | −25 |
| retrieval_hit_diff | 404 (45.8%) | 502 (56.9%) | **+98** |

Smoking gun in answer-length stats: median answer chars went **12 → 37**, mean **21 → 82**. The low-confidence fallback was firing on a huge fraction of queries (its `< 2 words` clause triggers on every single-word answer — "Velez", "1992", "blue", price tokens) and replacing correct-but-short SQuAD2 spans with too-long sentences that fail the AE 0.9 threshold.

Verdict: the fallback is **net negative**. `v4-dict-id` already had 0 `retrieval_hit_empty` cases, so there was nothing to fall back *from*; its only effect was replacing valid spans with worse ones. Single-word answers are common (especially L1: dates, prices, names) and we were destroying them.

The other three changes look broadly fine:
- Paragraph chunking is slightly worse on retrieval (+5 misses) — possibly noise, not the main loss.
- Batched SQuAD2 is purely a speed change (`til test` ran at `1.56it/s` vs v4's `1.07it/s` — ~46% faster per batch).
- BM25 backfill is purely additive — only adds doc IDs, never removes.

## `v5b-no-fallback` shipped → cloud REGRESSED (-0.027)

Removed the low-confidence sentence fallback; kept paragraph chunking + batched SQuAD2 + BM25 backfill from v5-multi.

Local cleared the gate at 0.674 (within 0.005 of v4's 0.678) but **cloud scored 0.456 / 0.916** — accuracy −0.027 vs v4, speed +0.028, blended ~0.571 vs v4's 0.584 (net −0.013).

The local-cloud gap *widened* from v4's 0.195 to 0.218. That's the diagnostic signal: a change that's near-neutral on the local corpus but worse on the held-out corpus. Paragraph chunking is the prime suspect — local diagnostic on v5-multi already showed +5 retrieval misses (40 → 45) on the local corpus, and the cloud corpus likely amplifies that.

## `v5c-no-para` shipped → recovered (cloud 0.483, +0.024 speed retained)

Reverted `_chunk_document` to v4's plain 3-sentence sliding window with 1-sentence overlap. Kept the two additive changes:

1. **Batched SQuAD2 forward pass** — speed unlock; cloud speed went 0.888 → 0.912 (+0.024).
2. **BM25 doc-diversity backfill** — purely additive (only fills empty top-3 slots, never replaces a reranker pick).

Result: cloud `0.483 / 0.912`. Score matches v4 exactly; speed gain locked in; blended `0.590` — fresh leaderboard high. Paragraph chunking confirmed the v5b regressor, not the other changes.

## Lessons recorded

- **Ship changes sequentially on this task, not bundled.** v5-multi tried four changes at once; we couldn't attribute the regression without bisecting through v5b → v5c. Each bisect was a submission. On the next push (v6+), one knob at a time.
- **Local-cloud gap is a diagnostic, not a number.** v5b local equiv_rate was within 0.005 of v4, but cloud was −0.027. When the gap *widens* on a change, that's the signal — the held-out corpus is responding to the change differently than the local one. Worth checking before submitting.
- **Paragraph-aware chunking does not transfer on this corpus.** The local diagnostic showed +5 retrieval misses (40 → 45); the cloud took it harder. The structural reason (changing BGE embedding distribution of chunks) means it's unlikely to be fixable just by parameter tweaks. Don't revisit.

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
- If `retrieval_hit_empty` is large → SQuAD2 is returning empty/degenerate spans on a non-trivial fraction of cases; revisit a narrower low-conf fallback that only fires on truly empty answers, not single-word ones.

The script prints estimated cloud-score lower/upper bounds; compare to `til test`'s actual `equiv_rate` to triangulate where the AE model is finding extra credit beyond exact/substr.

## Next priority

1. **`v6-roberta-large` — swap the QA model**. The v4/v5c error report puts retrieval at 95.5% hit rate; the dominant loss bucket is `retrieval_hit_diff` at 45.8% (404/883). That's cases where we have the right doc but the AE 0.9 threshold deemed our SQuAD2 answer non-equivalent — i.e. span quality is the bottleneck. Swap `deepset/roberta-base-squad2` → `deepset/roberta-large-squad2`. ~1.4 GB extra container size; per-question latency ~2.5× on GPU fp16, comfortably within budget given the batching headroom. Expected impact: +0.05–0.10 cloud, moving cases from `diff` to `exact/substr`.
2. **If v6 lands a real lift**, *then* consider a retrieval-side change for the last +0.018 of headroom (`bge-base-en-v1.5`, or asymmetric BM25⊕dense weighting). Ship one at a time.
3. **Don't combine changes on this task again.** v5-multi → v5b → v5c bisect cost three submissions; sequential A/B would have been one.

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
