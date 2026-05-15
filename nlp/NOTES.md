# NLP — notes & history

Last updated: 15 May 2026 19:25 SGT — v9-doc-ensemble NEW HIGH 0.683/0.868

Per-task working log for NLP (RAG question-answering). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#nlp).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`v9-doc-ensemble` — official 0.683 / 0.868 (15 May 19:25 SGT, 0 of 700 errors).** NEW HIGH (+0.004 cloud accuracy vs v8b). Keeps the v8b chunked-context RoBERTa answerer, then adds whole-document BM25 + BGE retrieval as a light prior and reranker seeder. Local moved `0.708` → `0.711`; retrieval misses dropped `40` → `37` and retrieval hit rate improved `95.5%` → `95.8%`. Cloud moved similarly modestly (`0.679` → `0.683`) with a small speed dip (`0.872` → `0.868`), so blended NLP nudged `0.727` → `0.729`.

## Current candidate — `v10-template-lite`

Using the downloaded novice corpus snapshot (`novice-nlp-light-20260515.tgz`, extracted under local gitignored `data/`), added a stdlib analysis helper:

```bash
python training/nlp/analyze_answer_templates.py \
  --data data/novice-nlp-light-20260515/novice/nlp/nlp.jsonl \
  --docs data/novice-nlp-light-20260515/novice/nlp/documents
```

Key result: the local set is not mostly verbatim span extraction. Out of 883 answers, only 316 are exact source substrings, 37 are case-insensitive matches, and 49 are punctuation-normalized matches; 481 are not literal in the source text. L1 itself is mixed (`225/592` not literal), while L2 is mostly non-literal (`256/291`). That explains why `v9` can retrieve 95.8% of questions but still sit at 0.711 local / 0.683 cloud: many retrieved chunks require canonical output or small composition, not just a better span.

Code candidate in [src/nlp_manager.py](src/nlp_manager.py): keep v9 retrieval and RoBERTa as the primary answerer, then run a conservative deterministic layer for obvious arithmetic/date forms:

- elapsed days from `YY-MM-DD` pairs (`"37 days"`)
- elapsed years from PCE/CE year pairs (`"28 years"`)
- percentage-point differences from exactly two percentages (`"12 percentage points"`)

Default `NLP_RULE_MODE=conservative` only overrides the model when the extracted span is clearly diffuse or missing the computed value. `NLP_RULE_MODE=aggressive` is available for a bolder Workbench A/B, and `NLP_RULE_MODE=off` disables this layer. This is deliberately narrow; previous broad fallbacks were net-negative.

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
3. **Sentence-window chunking.** Each document is split into 3-sentence windows with 1-sentence overlap across the whole document. Paragraph-aware chunking was tested earlier and regressed on cloud; the current winning path instead changes the fine-tune context to the answer-containing inference chunk.
4. **Hybrid retrieval.** BM25Okapi over tokenized chunks ⊕ dense cosine over `BAAI/bge-small-en-v1.5` embeddings (CLS-pooled, L2-normalized; query gets BGE's English search prefix). v9 adds whole-document BM25 ⊕ BGE as a second opinion, lightly boosts chunks from high-scoring documents, and seeds the reranker with best chunks from top documents. Top chunk candidates → reranker.
5. **Cross-encoder rerank.** `BAAI/bge-reranker-base` scores `(question, chunk)` pairs. Top-10 → QA.
6. **Top-3 doc IDs with BM25 backfill.** Walks reranked passages collecting unique parent doc IDs. If the reranker concentrated on <3 unique docs (common when one document has many highly-relevant chunks), backfills from the un-reranked hybrid list. Protects retrieval recall — the new eval gates every case on retrieval, so empty doc slots are pure waste.
7. **Extractive QA (batched).** Fine-tuned `deepset/roberta-large-squad2` when `nlp/models/roberta-finetuned-squad2/` is baked into the image; otherwise falls back to stock `roberta-base-squad2`. All features are tokenized and forwarded in batches (`QA_BATCH=16`) with `overflow_to_sample_mapping` to recover which context each feature came from.
8. **Narrow deterministic post-processing (candidate).** `v10-template-lite` computes only high-confidence date/year/percentage-point answers after the learned QA span. Conservative default avoids touching normal span answers unless the span is long or missing the computed value.
9. **No broad low-confidence sentence fallback.** The old fallback was confirmed net-negative because it replaced many correct single-token spans with too-long sentences.
10. **Speed.** GPU half-precision on all three models. Current cloud speed is `0.868` for `v9-doc-ensemble`; `v10-template-lite` should be near-neutral because rules are regex-only.

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
v5c-no-para   14/05 19:44        0.483   0.912   0 / 700   0.678        Reverted paragraph chunking; kept batched SQuAD2 + BM25 backfill. Score matches v4 exactly + speed +0.024 from batching → blended 0.590
v7-finetuned-v1 15/05 11:39      0.517   0.880   0 / 700   0.709        Prior high (+0.034 cloud vs v5c). Fine-tuned roberta-large-squad2 on local nlp.jsonl (353/883 retained, epoch-1 best eval_loss 0.614). Blended 0.608
v7-finetuned-v2 (not shipped)    —       —       —         0.698        Variants+regex+rapidfuzz, 431/883 retained. Local REGRESSED -0.011 vs v1; substr -38, diff +26 (fuzzy spans noisy). NOT submitted
v8b-chunked-context 15/05 18:35   0.679   0.872   0 / 700   0.708        Prior high (+0.162 cloud vs v7-v1). --use-answer-chunk + rapidfuzz off, 353/883 retained; first run had span-misalignment bug fixed in `627c9ce`, retrained. Local looked flat, but cloud strongly rewarded the chunked-context inductive bias. Blended 0.727
v9-doc-ensemble 15/05 19:25       0.683   0.868   0 / 700   0.711        SHIPPED, NEW HIGH (+0.004 cloud vs v8b). Whole-document BM25+BGE retrieval prior/seeding rescued 3 local retrieval misses (40→37), exact unchanged, substr +1, diff +2. Small accuracy lift with small speed cost; blended 0.729
v10-template-lite (candidate)      —       —       —         TBD          v9 + conservative regex arithmetic/date answer layer for elapsed days/years and percentage-point deltas. Build/test before submit; if local exact/substr rises without diff blow-up, ship.
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

## `v7-finetuned-v1` analysis — what worked

Cloud +0.034 (0.483 → 0.517) matches local +0.035 (0.674 → 0.709). The fine-tune transferred near-1:1, a clean signal that:

1. The held-out cloud questions follow the same authoring style as local `nlp.jsonl` (same paraphrase frequency, same answer formats). This makes local equiv_rate a reliable proxy for cloud — at least in the +/− 0.005 band.
2. The bottleneck WAS QA span quality, as the error-bucket diagnostic predicted. With retrieval at 95.5%, moving cases from `retrieval_hit_diff` to `exact/substr` directly lifts the cloud score.
3. roberta-large-squad2 + 318 train examples was enough for a meaningful lift even with aggressive overfitting in epochs 2-3 (`load_best_model_at_end=True` rescued us).

Error-bucket shift (local, v5c → v7-v1):

| Bucket | v5c | v7-v1 | Δ |
|---|---:|---:|---:|
| retrieval_miss | 40 (4.5%) | 40 (4.5%) | 0 |
| retrieval_hit_exact | 183 (20.7%) | **242 (27.4%)** | **+59** |
| retrieval_hit_substr | 256 (29.0%) | 205 (23.2%) | −51 |
| retrieval_hit_diff | 404 (45.8%) | 396 (44.8%) | −8 |
| L1 exact-match | 29.7% | **39.7%** | **+10pp** |
| L2 exact-match | 2.4% | 2.4% | 0 |

The −51 substr / +59 exact shift means the model is producing more strictly-verbatim spans now. L1 single-fact questions are the main beneficiary; L2 multi-fact questions are unchanged (those need composition, not extraction).

## `v7-finetuned-v2` (built; local regression, not shipped)

Same training script + v2 data-prep (variants + flexible regex + rapidfuzz). Retention 353 → 431 (+78). But training eval_loss curve was concerning:

```
epoch 1: eval_loss ?  (not pasted)
epoch 2: eval_loss ?  (not pasted)
epoch 3: eval_loss 1.238   ← v1 had 0.872 at epoch 3
```

The `load_best_model_at_end=True` flag still picks the best epoch, but v2's higher epoch-3 loss suggests it overfit harder. Hypothesis: the rapidfuzz fallback returns window-length spans (not the actual answer text), so some examples now train on slightly-misaligned char ranges. Those noisy examples confuse the model on questions where the correct span is shorter than the fuzzy-matched window.

Outcome: local `0.698`, below v7-v1's `0.709`. The fuzzy fallback was net-negative; do not ship this branch.

## Path to higher scores — `v8` candidates

Status update (15 May, after v7-v2 + v8b/v9 local/cloud tests):

- **v7-v2** (variants + regex + rapidfuzz, 431 examples) → local 0.698, NOT submitted. Fuzzy-matched spans introduced noise; substr -38 / diff +26.
- **v8b-chunked-context** (rapidfuzz off, 353 examples, --use-answer-chunk; first run had a span-misalignment bug that was fixed in `627c9ce`, retrained) → local 0.708, cloud 0.679/0.872. This was the major breakthrough over v7-v1. Local aggregate was flat vs v7-v1 (-0.001), but cloud jumped +0.162, so the hidden corpus must reward the answer-containing chunk distribution much more than the local novice aggregate reveals.
- **v9-doc-ensemble** (whole-document BM25+BGE prior/seeding on top of v8b) → local 0.711, cloud 0.683/0.868. It did exactly what it was designed to do, but only at small scale: local retrieval misses fell 40→37, retrieval hit rose 95.5%→95.8%, and cloud accuracy rose +0.004. Speed fell -0.004, so blended lift is only about +0.002.

**New lesson: local aggregate `equiv_rate` is not enough as a ship gate.** For v8b, local exact/substr/diff buckets looked mostly flat, but hidden cloud accuracy moved massively. Treat local buckets as diagnostics, not as a scalar forecast. Chunked-context training is now proven positive on hidden eval even though the local set did not show it.

Next lever for cloud >0.70:

1. **Protect `v9-doc-ensemble` as the shipped baseline.**
2. **Retrieval is not the main blocker anymore.** Local upper bound is now 0.958 if every retrieved answer were accepted, but actual local is 0.711. That gap is the answer-equivalence / answer-syntax problem, especially `retrieval_hit_diff` staying ~396 cases.
3. **Current A/B: `v10-template-lite`.** Test whether narrow computed-answer rules can rescue a few high-confidence non-literal cases without reintroducing the broad-fallback regression.
4. **Next high-ROI direction after v10: broader answer canonicalization / question-template extraction.** Learn from `nlp.jsonl` question patterns and expected answer forms (dates, money, names, orgs, IDs, measures). Add deterministic post-processing or template-specific extraction on top of the retrieved chunk before trying a slower generative model.
5. **`v8a-genqa` — generative seq2seq head** remains a bigger swing. Train Flan-T5 on all 883 `(question, context, answer)` triples. Risk: AE 0.9 threshold punishes paraphrases and generation costs speed. Reward: possible path beyond 0.70 if outputs stay short and canonical.

Dropped candidates (confirmed not levers):
- `v8c-roberta-large-resume` — moot since v7-v2 was net-negative
- v3 data-prep (drop rapidfuzz without chunked-context) — v7-v2 showed fuzzy noise; v8b's win came from chunked-context, not fuzzy matching
- paragraph-aware chunking, low-confidence sentence fallback (confirmed earlier)

## Submission cadence rule

v5-multi → v5b → v5c bisect cost three submissions; sequential A/B would have been one. **Ship one knob at a time from here on.** v7-v1 already broke this rule by combining "fine-tune" + "roberta-large" in one image, but they're tightly coupled (the fine-tune needed roberta-large's capacity). Going forward: v7-v2 vs v7-v1 should isolate the data-prep delta cleanly.

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
