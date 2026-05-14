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

## Implementation (`v4-dict-id`, same image as `v3-id-parse`)

End-to-end overhaul in [src/nlp_manager.py](src/nlp_manager.py) and [src/nlp_server.py](src/nlp_server.py):

1. **Server.** Matches Ryan's pre-written async + poll pattern verbatim — `{"status":"loading"}` then `{"status":"loaded"}` on the poll endpoint; per-question response is `{"documents":[...], "answer":"..."}`.
2. **Doc-ID parser.** `_parse_doc_payload` handles three encodings: dict shape with `id` key, first-line `DOC-XXXX` (stripped from body), `DOC-XXXX` within the first 80 chars, and positional fallback `DOC-{i+1:04d}`. Cloud uses the dict shape; our parser caught it on first try.
3. **Chunking.** 3-sentence sliding window with 1-sentence overlap; each chunk carries its parent doc index.
4. **Hybrid retrieval.** BM25Okapi over tokenized chunks ⊕ dense cosine over `BAAI/bge-small-en-v1.5` embeddings (CLS-pooled, L2-normalized; query gets BGE's English search prefix). Per-query z-score normalise then sum. Top-30 chunks → reranker.
5. **Cross-encoder rerank.** `BAAI/bge-reranker-base` scores `(question, chunk)` pairs. Top-10 → QA. Top-3 unique parent docs become the response `documents`.
6. **Extractive QA.** `deepset/roberta-base-squad2` over each of the top reranked chunks; pick the highest-scoring span ≤64 tokens. Extractive (vs generative) is the right play because (a) Clairos has fictional proper nouns pretrained LMs don't know, and (b) the 0.9 ModernBERT threshold rewards near-verbatim spans from the source.
7. **Speed.** GPU half-precision on all three models. Cloud speed `0.888` confirms the inference budget is fine.

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
```

## What needs doing

1. **Batch SQuAD2 forward pass.** `QA_BATCH=16` is defined but unused — `_extract_answer` processes one context at a time. Batching the forward pass over top-10 reranked chunks should free 50–80% of QA latency budget. Highest-impact, lowest-risk next move.
2. **Paragraph-aware chunking.** Currently sentence-windowed. Splitting on `\n\n` first preserves answer spans that cross sentence boundaries; common failure mode in RAG.
3. **Bigger embedder/reranker** (after #1 frees the latency budget). `bge-base-en-v1.5` instead of `bge-small`; `bge-reranker-large` instead of base. Adds ~700 MB to the image.
4. **Bigger QA model.** `deepset/roberta-large-squad2` (~1.4 GB). Worth trying after #1+#2.
5. **Sentence-of-best-chunk fallback** when SQuAD2 confidence is low. Cheap dropin; helps the "right doc, wrong span" failure mode.

Estimated reachable: cloud `0.55–0.65` with #1+#2 alone, more with #3/#4.

## CODEX recommendation with the way forward

NLP is the cleanest immediate score lever after the `v4-dict-id` recovery. It is worth 20% of the overall qualifier score, the current cloud score is still only `0.483`, and the cloud speed `0.888` leaves room to trade some latency for accuracy. Because the evaluator gates every case on document retrieval before answer equivalence, the way forward should protect retrieval first, then improve answer extraction once the right doc is reliably in the top 3.

Recommended order:

1. **Patch batching first, then re-test `v4` behavior.** `_extract_answer` currently runs RoBERTa-SQuAD2 one context at a time even though `QA_BATCH=16` exists. Batch the QA tokenization/model forward over the top reranked contexts, keep the same span-selection logic, and submit only if local accuracy stays flat or improves. This is mainly a speed-budget unlock: if cloud speed rises, we can spend that budget on stronger retrieval/QA without harming the 25% speed component.
2. **Make chunking paragraph-aware before changing models.** Split documents into paragraphs first, then build overlapping sentence windows inside each paragraph, with a fallback that also emits a short whole-paragraph chunk when the paragraph is not too long. This should improve L1/L2 answer spans without changing dependencies or image size, and it is safer than jumping straight to larger models.
3. **Add a retrieval/answer error report on local `nlp.jsonl`.** For each local question, save target docs, returned docs, answer, reference answer, equivalence probability, and level/metadata if present. Bucket failures into `retrieval miss`, `retrieval hit + answer fail`, and `format/empty fail`. The scoring contract makes this crucial: retrieval misses are `0.0`, retrieval hits with bad answers are still `0.4`, so the report tells us whether to tune BM25/dense/reranker or QA.
4. **Tune retrieval before QA if misses dominate.** Try `TOP_K_RETRIEVE=50`, `TOP_K_RERANK=12-16`, and score weighting such as `1.2*z(BM25)+z(dense)` vs `z(BM25)+1.2*z(dense)`. Also return unique parent docs from the reranker, but consider a fallback that backfills with high-scoring BM25 parent docs when reranker diversity is low. Top-3 doc overlap is the first gate, so a slightly noisier answer is acceptable if document recall improves.
5. **Only then spend the saved latency on bigger models.** If batching keeps speed healthy, A/B `bge-base-en-v1.5` first because it improves the retrieval gate. Try `bge-reranker-large` next if reranker mistakes are visible in the error report. Move to `roberta-large-squad2` only when retrieval-hit/answer-fail is the main bucket.
6. **Add a conservative low-confidence fallback.** When the QA span score is weak or the answer is tiny/generic, return the best sentence or compact paragraph around the highest reranked chunk instead of an empty or over-short span. The equivalence model rewards close source wording and truncates to 64 tokens, so a concise extract from the right chunk can beat an overconfident bad span.

Concrete next tag plan:

```text
v5-qa-batch       batch QA only; expected same accuracy, better speed
v6-para-chunks    QA batch + paragraph-aware chunking; expected accuracy lift
v7-retrieval-ab   v6 + retrieval sweep winner from local error buckets
v8-bge-base       v7 + larger embedder if speed remains acceptable
```

Do not touch `nlp/src/nlp_server.py` unless the upstream contract changes again. Keep `_parse_doc_payload` defensive even though the official corpus-load shape is now dicts with `id` and `document`; it saved the recovery once already.

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
