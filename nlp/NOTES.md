# NLP — notes & history

Last updated: 16 May 2026 14:10 SGT — **NLP HARD-PARKED at v9-doc-ensemble
0.683/0.886.** v12-candidate-ranker (submitted 16/05 13:48) REGRESSED to
0.642/0.829 — worst NLP submission since v5b, blended 0.689 vs v9's 0.734.
Local bucket shift exact 273→204 (-69), substr 178→232 (+54), diff 395→410
(+15) confirmed the predicted failure mode: the ranker promoted doc-mined
short tokens (e.g. "37" over "37 days") that passed the exact/substr training
proxy but failed the 0.9 ModernBERT AE threshold on cloud. Speed also dropped
-0.057 from candidate-mining overhead (18 sentences × 6 regex types per
question, local 4:23 → 10:40). Every post-v9 swing has now regressed
monotonically (v10 0.683, v11 0.680, v8a 0.652, v12 0.642). The NLP
architecture is exhausted; v9 stays on the leaderboard for the rest of the
qualifier. Remaining ~7 days reallocate to AE (40% weight).

## v12 — candidate-answer reranker (16 May)

Implemented in [src/nlp_manager.py](src/nlp_manager.py):

- Default QA mode is extractive even if stale Flan-T5 weights exist. Set
  `NLP_QA_MODE=generative` only to reproduce the regressed v8a path.
- RoBERTa now emits the top answer spans, not only the single best span.
- Candidate pool also includes date/year/percentage rules, v11-style
  canonicalizer outputs, and short literal candidates mined from the top-3
  returned docs: dates, money, codenames, percentages, proper nouns, and
  relation phrases.
- Candidate selection uses a conservative heuristic by default and can load a
  trained lightweight ranker from `nlp/models/answer_ranker.json`.
- [Dockerfile](Dockerfile) now bakes `answer_ranker.json` when present.
- [../training/nlp/train_answer_ranker.py](../training/nlp/train_answer_ranker.py)
  trains that ranker on Workbench from local `nlp.jsonl` using the exact/substr
  proxy.

Runbook:

```bash
python training/nlp/train_answer_ranker.py \
  --data /home/jupyter/novice/nlp/nlp.jsonl \
  --docs /home/jupyter/novice/nlp/documents \
  --out nlp/models/answer_ranker.json

til build nlp v12-candidate-ranker
til test nlp v12-candidate-ranker
python nlp/error_report.py /home/jupyter/melanie-minions/nlp_results.json \
                          /home/jupyter/novice/nlp/nlp.jsonl
til submit nlp v12-candidate-ranker
```

Decision rule: submit if local equiv_rate beats v9 under the same evaluator or
if exact/substr proxy improves with retrieval misses flat. If the learned JSON
ranker overfits, rebuild with `NLP_ANSWER_RANK_MODE=off` to test the heuristic
candidate path alone.

### Cloud result (16/05 13:48) — REGRESSED -0.041

```text
v12-candidate-ranker  16/05 13:48:51   0.642 / 0.829   0 / 700   local 0.663
```

Local was already a -0.048 regression vs v9 (0.711 → 0.663) — the decision rule above
said "submit if local beats v9", and local did NOT beat v9. The submission proceeded
anyway. Cloud confirmed the local signal:

| Bucket | v9 local | v12 local | Δ |
|---|---:|---:|---:|
| retrieval_hit_exact | 273 | **204** | **−69** |
| retrieval_hit_substr | 178 | 232 | +54 |
| retrieval_hit_diff | 395 | 410 | +15 |
| retrieval_miss | 37 | 37 | 0 |

**Smoking gun**: -69 exact and +54 substr means the ranker replaced clean QA-span
answers with shorter doc-mined alternatives. Those passed the exact-or-substr
training proxy ("37" is a substring of "37 days") but failed the 0.9 ModernBERT
AE threshold on cloud (ModernBERT does not rate "37" ≡ "37 days" at 0.9).

This is exactly the v11 failure mode at larger scale — training on a proxy that
doesn't match the cloud scorer. The candidate space was 10× bigger so the regression
was 10× deeper (v11: -0.003, v12: -0.041).

Three contributing factors, in order of impact:

1. **Training proxy mismatch.** `_is_positive` in train_answer_ranker.py used
   exact-or-substr; cloud uses ModernBERT 0.9. Same mistake as v11's replay script.
2. **Heuristic prior put rules above QA.** `source_rule = 4.5 > source_qa = 4.0`
   in `_heuristic_candidate_score`. v10 already showed those rules don't move cloud
   on this corpus; they shouldn't outrank model spans.
3. **Document-mining surface area.** `_document_answer_candidates` generates up to
   ~300 raw candidates per question from 18 sentences × 6 regex types. The dedupe cap
   of 32 keeps too many, and `echoes_question = -2.0` penalty demotes correct QA
   spans whose text overlaps with the question subject.

### Final NLP verdict

Every post-v9 swing regressed monotonically:

```text
v10-template-lite      0.683 / 0.882   neutral (-0.001 blended)
v11-canonical-answer   0.680 / 0.881   -0.004 blended
v8a-genqa              0.652 / 0.836   -0.040 blended
v12-candidate-ranker   0.642 / 0.829   -0.045 blended
```

The architecture is at its ceiling on this corpus. Confirmed dead levers now total
seven: paragraph chunking, low-conf fallback, rapidfuzz spans, narrow rule
templates, full-doc canonicalization, generative answers, candidate reranking.
**NLP is frozen at v9-doc-ensemble** for the rest of the qualifier. Remaining time
to AE.

Per-task working log for NLP (RAG question-answering). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#nlp).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`v9-doc-ensemble` — official 0.683 / 0.886 after the third resubmit (16 May 05:21 SGT, 0 of 700 errors).** Best current NLP blend. The first v9 submit was `0.683 / 0.868`; resubmitting the exact same image later returned `0.683 / 0.883`, then `0.683 / 0.886`, confirming that the speed metric has measurable run-to-run noise. v9 keeps the v8b chunked-context RoBERTa answerer, then adds whole-document BM25 + BGE retrieval as a light prior and reranker seeder. Local moved `0.708` → `0.711`; retrieval misses dropped `40` → `37` and retrieval hit rate improved `95.5%` → `95.8%`.

## `v10-template-lite` — neutral A/B

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

Result (15 May 20:16 SGT): `v10-template-lite` scored `0.683 / 0.882` officially with `0 / 700` errors. Local stayed at `0.711`; buckets shifted only `substr 177→178`, `diff 396→395`, with retrieval unchanged at `95.8%`. Interpretation: the conservative template layer is safe but too narrow to matter. Do not treat this as a new direction by itself; the next move needs broader answer canonicalization or a different QA head, not more tiny regex patches.

## Current candidate — `v11-canonical-answer`

Built from the Workbench v9 failure pack (`nlp-v11-failure-pack.tgz`), which contained:

- `nlp_results.json` from `til test nlp v9-doc-ensemble`
- `nlp_failure_analysis.jsonl` with all retrieval misses and retrieval-hit-diff cases
- `nlp_failure_summary.txt`
- matching `nlp.jsonl`

Baseline pack:

```text
retrieval_miss        37
retrieval_hit_exact  273
retrieval_hit_substr 178
retrieval_hit_diff   395
```

`v11-canonical-answer` keeps all v9/v10 retrieval and RoBERTa behavior, then adds a conservative full-document canonicalizer over the top returned docs. This is different from v10: v10 only looked at top reranked chunks and a few arithmetic forms; v11 uses the full text of the top-3 returned docs to rescue recurring right-document / wrong-phrase cases.

Implemented cases:

- classified/internal codenames from nearby all-caps annex references (`SEASTITCH`)
- penalty answers with financial penalty + mandatory equipment surrender
- event-year differences where question asks "between X and Y"
- singular `Phi Credit` → `Phi Credits`
- `The Edge Research Project` → `Edge Research`
- `four in favor to one against` → `4-1`
- `(...bn)` parenthetical normalization to `N billion Phi Credits`
- Sharpsea Bloc logistics industry phrasing
- confidence level answers (`Medium only` → `medium confidence`)

Replay guardrail:

```bash
python training/nlp/replay_canonicalizer.py \
  --results data/nlp-v11-failure-pack/nlp_results.json \
  --ground data/nlp-v11-failure-pack/nlp.jsonl \
  --docs data/novice-nlp-light-20260515/novice/nlp/documents \
  --changed-out data/nlp-v11-failure-pack/v11_canonicalizer_changed.jsonl
```

Replay result against v9 predictions:

```text
Before: exact 273, substr 178, diff 395, miss 37
After : exact 280, substr 181, diff 385, miss 37
```

That is a +10 exact/substr proxy move with no proxy regressions. Expected cloud gain is modest but real if hidden eval shares these answer-syntax patterns; if cloud stays flat, the next lever is bigger than regex/canonicalization: generative QA or stronger supervised answer formatting.

### Cloud result (15/05 21:26 + 21:39) — REGRESSED -0.003

Submitted v11 twice on 15/05:

```text
21:26:38   0.680 / 0.881   0 / 700
21:39:55   0.680 / 0.873   0 / 700
```

Accuracy stable at 0.680 across both runs, so the -0.003 vs v9 (0.683) is real, not run-to-run variance. Speed dropped slightly on the resubmit (0.881 → 0.873), consistent with the speed noise we already documented.

Two hypotheses for why the +10 replay didn't transfer:

1. **Replay was against the OLD local eval.** The replay script used the pre-14-May rubric (binary equiv ≥ 0.5, plain-string doc shape) on stored v9 predictions. The upstream test_nlp.py we just synced (16/05) reveals the new rubric: threshold 0.9 + 0.4 retrieval-only partial credit. A rewrite that flipped an answer from "substr-of-truth" → "canonicalized form" might still fail the 0.9 ModernBERT threshold, so the proxy gain doesn't translate into cloud points.
2. **Canonicalizer patterns were not sampled on the hidden corpus.** The +10 cases came from specific phrasings in the local pack (codename SEASTITCH, "four in favor to one against", `(2.5bn)`). If those exact patterns are absent from the held-out questions, the rewrites either no-op (best case) or fire incorrectly on lookalike phrasings (worst case, -3 net).

**Verdict:** v11 is not a ship. v9-doc-ensemble keeps the leaderboard slot. **Net cost: 2 submissions and we learned that deterministic canonicalization over top-3 docs is too narrow to overcome the 0.9 threshold.** The next swing has to be a different *answer formatter*, not more regex.

## `v8a-genqa` — generative QA candidate (Flan-T5)

Decided 16/05 after v11 regressed. Hypothesis: deterministic answer rewriting is exhausted; the remaining `retrieval_hit_diff` bucket (~395 cases) needs a model that *generates* the canonical answer form, not extracts a span. Flan-T5 fine-tuned on all 883 `(question, context, answer)` triples — including the ~530 paraphrased answers that extractive training has to skip.

### Scaffolding state (already in place, verified 16/05)

- Training: [training/nlp/finetune_genqa.py](../training/nlp/finetune_genqa.py) — full script, Flan-T5-base default, 3 epochs, `load_best_model_at_end`, outputs to `nlp/models/flan-t5-finetuned/`.
- Manager: [src/nlp_manager.py](src/nlp_manager.py) — QA selection ladder prefers `flan-t5-finetuned` over `roberta-finetuned-squad2` over stock base. Auto-detects `is_encoder_decoder` from config and routes to `_generate_answer` (beam=4, max_new_tokens=64, conditioned on top-1 reranked chunk only).
- Docker: [Dockerfile](Dockerfile) lines 34-36 already bundle `flan-t5-finetuned/` if present.

### Two bugs fixed 16/05 before any v8a-genqa build

1. **Missing `sentencepiece` dep.** Flan-T5 tokenizer is SentencePiece-based; without it `AutoTokenizer.from_pretrained()` fails at runtime. Added to [requirements.txt](requirements.txt).
2. **T5 + `.half()` NaN trap.** [src/nlp_manager.py:368-374](src/nlp_manager.py) unconditionally called `.half()` on all three models. T5 has known fp16 overflow in attention ops → NaN logits at generate time. Patched to keep the generative QA model at fp32; extractive still uses fp16.

### Workbench runbook (user executes)

```bash
# 1. Train (on Workbench GPU). --use-chunk-context matches inference behavior
#    (manager conditions on the top reranked chunk, so train on chunk too).
python training/nlp/finetune_genqa.py --use-chunk-context

# 2. Verify weights landed where the Dockerfile expects.
ls nlp/models/flan-t5-finetuned/

# 3. Build the image.
til build nlp v8a-genqa

# 4. Ship gate: local equiv_rate vs v9 under the NEW test_nlp.py (synced
#    16/05 from upstream — threshold 0.9, 0.4 retrieval-only partial credit).
til test nlp v8a-genqa
# Compare against v9 under the same test_nlp.py first if not already baselined.

# 5. Submit only if local does not regress.
til submit nlp v8a-genqa
```

### Known risks (front and center)

- **Speed.** Flan-T5-base autoregressive generation at beam=4 is ~50-100ms per question on a single chunk. 700 questions → +35-70s vs v9. Speed score `(1 - t/1800)` is currently 0.883 (~211s used); v8a-genqa likely lands ~0.82-0.85. Accuracy needs to lift by **>= +0.02** to keep blended even with a -0.05 speed hit.
- **0.9 threshold punishes paraphrase.** The generative head could paraphrase a correct answer into a form the ModernBERT equivalence model rates < 0.9. Training with `--use-chunk-context` and on the exact answer text (no rapidfuzz, no variants) keeps outputs close to the source phrasing.
- **One-shot generation, not batched.** Current `_generate_answer` processes one question at a time. If speed regression is the only blocker, follow-up tag `v8a-genqa-batched` can batch 8-16 questions through `model.generate` for a 5-10× speedup.

### Decision rule

- Local equiv_rate (new test) ≥ v9 under new test → submit.
- Local equiv_rate (new test) within -0.005 of v9 AND retrieval_hit_diff ↓ by >= 20 → submit (the chunked-context lesson: cloud sometimes rewards distribution shift that local doesn't reveal).
- Otherwise: drop, freeze on v9.

### Cloud result (16/05 05:10) — REGRESSED, dead lever

```text
v8a-genqa     16/05 05:10:19    0.652 / 0.836    0 / 700    local 0.682
v9 baseline   16/05 05:21:57    0.683 / 0.886    0 / 700    local 0.711  (best ever)
```

- **Accuracy** -0.031 vs v9. Local→cloud gap was 0.030 (0.682 → 0.652), almost identical to v9's 0.028 (0.711 → 0.683). Translation: the generative head transferred cleanly; it's just structurally worse than fine-tuned RoBERTa-large extractive on this corpus.
- **Speed** -0.047 vs v9. Local timing was 5:21 vs v9's 4:23 (~22% slower); that 22% materialised on cloud as 257s vs 211s.
- **Blended** ≈ 0.652 × 0.75 + 0.836 × 0.25 = `0.694` vs v9's `0.734`. Net -0.040.

**Why generative lost here, when v7-v1 (RoBERTa-large fine-tune) had won:**

1. **0.9 AE threshold punishes paraphrase.** Extractive spans are verbatim source text — the ModernBERT equivalence model rates them high on lexical overlap. Generative outputs reword (even when correct) and slip below 0.9. We knew this risk going in; it was the deciding factor.
2. **883 examples is too thin for seq2seq.** RoBERTa fine-tunes well at this scale because the span-prediction head is small and pre-conditioned. Flan-T5 has to learn answer *style* from few-shot.
3. **One-shot generation, not batched.** Cost us 22% wall-clock — `v8a-genqa-batched` could recover most of that, but with accuracy already -0.031 it can't win blended even with v9-equivalent speed. Don't build the batched variant.

**Verdict:** generative QA is a dead lever on this corpus. NLP parks at v9-doc-ensemble.

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
8. **Deterministic answer canonicalization.** `v10-template-lite` added narrow chunk-level date/year/percentage rules and was neutral. `v11-canonical-answer` adds a broader but still conservative full-document canonicalizer over the top returned docs for repeated answer-syntax misses.
9. **No broad low-confidence sentence fallback.** The old fallback was confirmed net-negative because it replaced many correct single-token spans with too-long sentences.
10. **Speed.** GPU half-precision on all three models. Current best cloud speed is `0.883` for the resubmitted `v9-doc-ensemble`; v10 was near-neutral at `0.882`. v11 should be near-neutral too because it is regex/sentence scoring over only the top-3 docs.

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
v9-doc-ensemble 15/05 19:46       0.683   0.883   0 / 700   0.711        Same image resubmitted; accuracy unchanged, speed +0.015. Best NLP blend ~0.733. Treat speed deltas of this scale as evaluator noise.
v10-template-lite 15/05 20:16      0.683   0.882   0 / 700   0.711        NEUTRAL. v9 + conservative regex arithmetic/date answer layer. Local substr +1 / diff -1, retrieval unchanged; cloud accuracy unchanged. Safe but too narrow.
v11-canonical-answer 15/05 21:26    0.680   0.881   0 / 700   0.711(old)  REGRESSED -0.003 vs v9. Full-doc canonicalizer; replay +10 exact/substr on OLD-eval proxy did NOT transfer through the 0.9 AE threshold. Blended 0.730.
v11-canonical-answer 15/05 21:39    0.680   0.873   0 / 700   0.711(old)  Same image resubmit; accuracy unchanged → -0.003 is real, not variance. Speed -0.008 within evaluator noise.
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
- **v9-doc-ensemble** (whole-document BM25+BGE prior/seeding on top of v8b) → local 0.711, cloud 0.683/0.868 on first submit and 0.683/0.883 on same-image resubmit. It did exactly what it was designed to do, but only at small scale: local retrieval misses fell 40→37, retrieval hit rose 95.5%→95.8%, and cloud accuracy rose +0.004. The speed improvement on resubmit is evaluator variance, not a code change.

**New lesson: local aggregate `equiv_rate` is not enough as a ship gate.** For v8b, local exact/substr/diff buckets looked mostly flat, but hidden cloud accuracy moved massively. Treat local buckets as diagnostics, not as a scalar forecast. Chunked-context training is now proven positive on hidden eval even though the local set did not show it.

Next lever for cloud >0.70:

1. **Protect `v9-doc-ensemble` as the shipped baseline.**
2. **Retrieval is not the main blocker anymore.** Local upper bound is now 0.958 if every retrieved answer were accepted, but actual local is 0.711. That gap is the answer-equivalence / answer-syntax problem, especially `retrieval_hit_diff` staying ~396 cases.
3. **`v10-template-lite` was safe but too small.** It proves narrow computed-answer overrides do not break the image, but a +1 substring shift is not enough to move cloud accuracy.
4. **Current A/B: `v11-canonical-answer`.** This is the broader deterministic pass we wanted: full-doc canonicalization for repeated syntax failures. Submit only after Workbench `til test`; local replay is positive but not a replacement for container eval.
5. **`v8a-genqa` — generative seq2seq head** remains the next bigger swing. Train Flan-T5 on all 883 `(question, context, answer)` triples. Risk: AE 0.9 threshold punishes paraphrases and generation costs speed. Reward: possible path beyond 0.70 if outputs stay short and canonical.

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
