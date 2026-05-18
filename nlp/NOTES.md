# NLP — notes & history

Last updated: 18 May 2026 ~04:15 SGT — **v15-family is blocked, not
"awaiting a format fix."** The v15 LoRA adapter trained successfully
(8h T4, `bs=1 grad_accum=8`, eval_loss 0.559, mean_token_acc 87.6%),
but there is no working path to ship it from the current Workbench T4:

- `v15-lora-qwen3-8b` direct serving hits the vLLM 0.9.0 Punica/Triton
  LoRA kernel on Turing (`LLVM ERROR: Unsupported rounding mode for
  conversion`). Local silently fell back/corrupted to 0.659; cloud turned
  the same crash into 700/700 HTTP 500s.
- `v14c-qwen3-4b` and `v14d-qwen3-8b` both served locally but timed out
  on cloud. The common factor is the `vllm/vllm-openai` base image and
  cloud startup/throughput, not request JSON shape.
- `v15-merged-qwen3-8b` cannot currently be AWQ-quantized on T4:
  `autoawq` is deprecated/broken, while `llm-compressor` either OOMs at
  DecoderLayer granularity or fails at Linear granularity with a Qwen3-GQA
  symbolic-trace `NoneType` error.

Shipping truth: **v9-doc-ensemble remains best blended** (`0.683/0.886`,
blended 0.734) and **v14-llm-rag remains best raw accuracy**
(`0.734/0.286`). Further NLP only makes sense through one of two paths:
AWQ-quantize the merged Qwen3 LoRA on bigger hardware, or retrain LoRA on
Qwen2.5-7B so it can run on the NGC base image that already survived cloud.

(Command R7B was considered as v14e but **no off-the-shelf AWQ or GPTQ-4bit
quantization exists on HuggingFace** — only MLX and GGUF, neither
vLLM-compatible. BF16 is 14 GB which is too tight on T4 alongside
BGE+reranker+KV cache. Self-quantizing via autoawq is feasible but adds
~1 hr GPU time and a calibration variable before we know if RAG-tuned
beats LoRA-tuned. Skipping in favour of LoRA.)

Historical (pre-v14): NLP was frozen at `v9-doc-ensemble`
(0.683/0.886 official, 0.711 local). `v13b-deberta` failed the local gate
at 0.667 and was 2.4x slower than v9, so it should not be submitted. `v13a`
also confirmed candidate-ranker architecture cannot beat v9 on this corpus.
v13a was tested two ways locally and both lost:

```text
v9-doc-ensemble local         0.711  (baseline)
v13a-ae-ranker  val top-1     0.384  (logistic ranker on AE-labels; NOT BUILT)
v13a-heuristic  local         0.663  (heuristic-only, structural fixes;
                                      identical to v12, NOT SUBMITTED)
```

Key data point: in v13a training, **ORACLE top-1 = 0.814** on the held-out 177
questions, i.e. for 81% of questions there exists a candidate in our pool that
passes ModernBERT AE @ 0.9. But our logistic + 20-feature scoring can only pick
the right one 38% of the time. The candidate pool has +0.10 of headroom; we
just can't rank it. Combined with v13a-heuristic = v12 = 0.663 locally,
candidate-ranker architecture is now ELIMINATED.

The final honest QA-retune swing also lost. Biggest historical lifts were
v7-v1 +0.034 and v8b +0.162, both from training the extractor, but
`v13b-deberta` did not transfer: local 0.667 vs v9's 0.711 and 9:06 vs 3:48
runtime. All post-v9 levers are now dead or below gate: deterministic
post-processing, generative QA, candidate reranking, and DeBERTa-v3-large
extractive retune. Protect v9 and do not rebuild NLP unless the build is forced
back to v9 behavior.

## v14-llm-rag — Qwen2.5-7B-Instruct-AWQ answerer (17 May, in progress)

Rationale recap (see top of file): v9 extractive ceiling is structural, not a
tuning problem. Every post-v9 swing inside the same architecture has regressed
or failed gate. v14 changes the model class.

### Architecture

```
question
  → BM25+BGE hybrid retrieval         (unchanged from v9)
  → bge-reranker-base                  (unchanged)
  → top-3 doc IDs                      (unchanged — retrieval gate)
  → Qwen2.5-7B-Instruct-AWQ via vLLM   ← new answerer
  → answer string (≤ 48 tokens)
```

### Code changes (17 May)

- [src/llm_answerer.py](src/llm_answerer.py) — new module. Boots vLLM with
  `quantization=awq_marlin`, `gpu_memory_utilization=0.78` (tunable via
  `NLP_LLM_GPU_MEM_FRACTION`), `max_model_len=4096`. Greedy decode
  (`temperature=0`, `max_tokens=48`) + stop tokens prevent the paraphrase
  drift that killed v8a-genqa.
- [src/few_shots.json](src/few_shots.json) — 6 hand-picked Q/A/context
  triples covering: money+penalty, codename, PCE date, year-delta with
  `approximately`, bare count, short entity. Threaded through the chat
  template as alternating user/assistant turns.
- [src/nlp_manager.py](src/nlp_manager.py) — `NLP_ANSWERER` env switch
  (`llm` | `extractive`). When `llm`: skip RoBERTa load entirely, init
  vLLM in `_init_models`, warm up inside `load_corpus` (untimed phase).
  `qa_batch` collects all questions in a request and hands them to
  `LLMAnswerer.answer_batch` as a single batched generate call.
- [src/nlp_server.py](src/nlp_server.py) — replaced the per-instance
  sequential loop with a single batched `manager.qa_batch(questions)`.
- [download_models.py](download_models.py) — pulls
  `Qwen/Qwen2.5-7B-Instruct-AWQ` via `huggingface_hub.snapshot_download`
  into `/workspace/models/llm`.
- [Dockerfile](Dockerfile) — `NLP_ANSWERER=llm`, `HF_HUB_OFFLINE=1`,
  build-time warning if the LLM dir didn't land.
- [requirements.txt](requirements.txt) — `vllm>=0.6.6,<0.9`.

### System prompt + few-shot strategy

Stays in `llm_answerer._DEFAULT_SYSTEM_PROMPT`. Key constraints:
- "Answer ONLY from the provided context. If the context does not contain
  the answer, return an empty string." (defensive against L4-style cases,
  even on Novice)
- "Quote the answer using the EXACT wording, dates, numbers, and units from
  the context. Do not paraphrase. Do not add explanation." (this is what
  v8a-genqa lacked — instruction-tuned 7B obeys, Flan-T5-base didn't)
- "Keep the answer as short as possible — typically 1 to 8 words."

Few-shots are intentionally short and verbatim. Each demonstrates a single
answer-form pattern.

### Workbench runbook

```bash
# 1. Build (will pull ~5 GB Qwen weights at build time; first build is slow)
til build nlp v14-llm-rag

# 2. Local A/B vs v9 under the synced upstream test_nlp.py
til test nlp v9-doc-ensemble           # baseline: 0.711, 3:48
til test nlp v14-llm-rag               # gate (see below)

# 3. Inspect failure buckets if local lands below v9
python nlp/error_report.py /home/jupyter/melanie-minions/nlp_results.json \
                           /home/jupyter/novice/nlp/nlp.jsonl

# 4. Submit only after the gate passes
til submit nlp v14-llm-rag
```

### Decision rule

| Local equiv_rate | Local time | Action |
|---|---|---|
| ≥ 0.78 | ≤ 6:00 | Submit. Expected cloud: 0.85–0.92. |
| 0.72–0.78 | ≤ 7:00 | Submit. Expected cloud: 0.78–0.87. |
| ≥ 0.711 but slow | ≤ 8:00 | Submit if blended (`0.75*acc + 0.25*speed`) > v9's 0.733. |
| < 0.711 | any | Do not submit. Diagnose: prompt drift, fp16 OOM fallback, or chunk-context mismatch. |
| any | > 8:00 | Speed regression too large; investigate before submitting. |

### Knobs

| Env var | Default | Purpose |
|---|---|---|
| `NLP_ANSWERER` | `llm` | `llm` or `extractive`. Set `extractive` to roll back to v9. |
| `NLP_LLM_DIR` | `/workspace/models/llm` | Where the AWQ weights live in the image. |
| `NLP_LLM_GPU_MEM_FRACTION` | `0.78` | vLLM `gpu_memory_utilization`. Drop to `0.70` if OOM with retriever+reranker on T4. |
| `NLP_QA_MODE` | `extractive` | Extractive-path knob, ignored when `NLP_ANSWERER=llm`. |

### Risks (front and center)

1. **VRAM** — Qwen-7B-AWQ + BGE + reranker + KV cache on a T4 (16 GB) is
   workable but tight. If `til test` OOMs on Workbench T4, drop
   `NLP_LLM_GPU_MEM_FRACTION` to 0.70 or swap the repo to
   `Qwen/Qwen2.5-3B-Instruct` in `download_models.py` (vLLM tag the build
   `v14-llm-rag-3b`).
2. **Paraphrase past AE 0.9** — if local equiv_rate stalls in the 0.70s
   despite retrieval still at 95.8%, the model is rewording. Tighten by
   prepending two more verbatim few-shots emphasising literal-quote output.
3. **vLLM + torch ABI conflict with the base image** — if `pip install vllm`
   clobbers the torch shipped by `nvcr.io/nvidia/pytorch:25.11-py3`, build
   logs will show a torch version mismatch. Fallback path: install with
   `--no-deps` and pin the few runtime deps vLLM needs (numpy, ray,
   xformers, msgspec, prometheus-client) manually.
4. **Speed regression** — vLLM continuous batching expects multiple
   in-flight requests. If the evaluator sends 1 instance per HTTP call,
   batching only helps when concurrent requests overlap. Mitigations: the
   server-side change already batches *within* a request; we may also need
   `uvicorn --workers 1` (already the default) so all requests hit one
   engine. Generation cap of 48 tokens caps per-question latency at
   ~150–250 ms even uncontested.

### Local pre-build sanity

`few_shots.json` and the prompt template are pure Python/JSON; quick smoke
test of the prompt construction without loading vLLM:

```bash
python -c "
import sys; sys.path.insert(0, 'nlp/src')
# Avoid importing the vllm dep just to check the prompt
from llm_answerer import _DEFAULT_SYSTEM_PROMPT
print(_DEFAULT_SYSTEM_PROMPT[:200])
"
```

### Submission gate — checklist before `til submit`

- [ ] `til test nlp v14-llm-rag` ran end-to-end without error
- [ ] Local equiv_rate ≥ 0.711 (v9 baseline)
- [ ] Local wall-clock ≤ 8:00
- [ ] Container logs show `[llm_answerer] loaded N few-shots` and `warmup complete`
- [ ] Container logs show `[nlp_manager] LLM answerer ready`, not a fallback message
- [ ] `nlp/error_report.py` shows retrieval miss ≤ 37 (v9 floor)

## v14c-qwen3-4b — Qwen3-4B-Instruct-2507-AWQ on vllm-openai base (17 May)

Two-axis change on top of shipped v14-llm-rag (0.734/0.286, blended 0.622):

### Why both axes change at once

1. **Model**: Qwen2.5 → Qwen3. The Qwen3-4B-Instruct-2507 release is roughly
   a year of architecture improvements + a non-thinking instruction tune
   over Qwen2.5-7B. Documented to match or exceed Qwen2.5-7B on QA tasks
   while being 1.75× smaller (4B vs 7B) — directly attacks the speed
   bottleneck.

2. **Base image**: `nvcr.io/nvidia/pytorch:25.11-py3` → `vllm/vllm-openai:v0.9.0`.
   The NGC base shipped pre-compiled `flash_attn` and `torchao` `.so`s
   against its own torch ABI. When vLLM (as a pip dep) downgraded torch,
   those `.so`s became unloadable (`torch.int1` missing, `c10::cuda::*`
   undefined symbol). Working around it required pinning transformers at
   4.46.3 and uninstalling both NGC extensions. Qwen3 needs transformers
   ≥ 4.51, so that workaround is no longer available — we have to fix
   the underlying ABI mismatch, and the cleanest fix is to use the upstream
   image that's tested as a coherent stack.

Coupling both changes in one tag is unusual for this project (the
"sequential A/B" lesson from v5-multi). It's justified because:
- Going to Qwen3 *requires* the transformers bump, which *requires* a
  working flash_attn, which is precisely what `vllm/vllm-openai` provides.
- The intermediate state (Qwen3 + NGC base + transformers 4.51 +
  hand-built flash_attn wheel) would be its own multi-hour rabbit hole
  for no shipping value.

### Code changes (17 May)

- [Dockerfile](Dockerfile) — `FROM vllm/vllm-openai:v0.9.0`, dropped the
  transformers-pin + torchao/flash_attn-uninstall band-aids, added
  `ENTRYPOINT []` to override the base image's OpenAI-API server CMD.
- [requirements.txt](requirements.txt) — slimmed to `rank_bm25==0.2.2`
  only; everything else comes from the base image.
- [download_models.py](download_models.py) — default
  `NLP_LLM_REPO=cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`.
- [src/llm_answerer.py](src/llm_answerer.py) — `apply_chat_template` now
  passes `enable_thinking=False` (Qwen3-specific kwarg, harmless on
  Qwen2.5); added `_strip_boilerplate()` post-processor that strips a
  single leading "Answer:" / surrounding quotes / trailing period from
  the LLM output. Deliberately minimal — v11 taught us aggressive answer
  rewriting regresses cloud AE @ 0.9.

### VRAM and speed math (T4 16GB)

| Component | Bytes | Notes |
|---|---|---|
| Qwen3-4B-AWQ weights | ~2.2 GB | vs Qwen2.5-7B-AWQ ~5.0 GB |
| BGE-small encoder | ~0.13 GB | unchanged |
| bge-reranker-base | ~0.46 GB | unchanged |
| KV cache for 4B @ ctx 4096, gmu 0.78 | ~7 GB | larger than v14 because the weight footprint shrank by ~3 GB |
| Headroom | ~6 GB | |

Per-question wall-clock estimate (back-of-envelope from v14's 2.4s/Q):
- v14 was prefill-bound on T4 sm_75 AWQ kernels
- 4B fewer weight FLOPs → ~1.7× speedup on prefill alone
- Smaller weights → faster activation memory bandwidth → another ~10–15%
- Expected per-Q ~1.3 s → 700 Q ~15 min wall-clock → speed score ~0.50
- Blended target: 0.75 · 0.72 + 0.25 · 0.50 = **~0.665** (still possibly
  short of v9's 0.734 blended; if accuracy holds at 0.73+ it could clear)

### Workbench runbook

```bash
git pull origin main
til build nlp v14c-qwen3-4b           # base image is ~6 GB; first pull is slow
til test nlp v14c-qwen3-4b            # gate: local ≥ 0.711 (v9 baseline)
til submit nlp v14c-qwen3-4b          # only if local clears the gate
```

### Decision rule

| Local equiv_rate | Local wall-clock | Action |
|---|---|---|
| ≥ 0.73 | ≤ 18:00 | Submit. Likely new blended high. |
| 0.71–0.73 | ≤ 16:00 | Submit if blended (`0.75·acc + 0.25·(1 − t/30)`) > v9's 0.734. |
| ≥ 0.73 | > 22:00 | Speed lift didn't show up; investigate vLLM kernel selection or shrink prompt further. |
| < 0.71 | any | Don't submit. Qwen3-4B underperformed; consider Qwen3-8B or speculative-decoding variant. |

### Knobs

| Env var | Default | Purpose |
|---|---|---|
| `NLP_ANSWERER` | `llm` | `llm` or `extractive` (extractive path requires `roberta-finetuned-squad2` to be bundled — not by default in v14c). |
| `NLP_LLM_REPO` | `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit` | Repo to bundle at build time. Set via `--build-arg` not runtime — affects what's baked into the image. |
| `NLP_LLM_DIR` | `/workspace/models/llm` | Where the LLM weights live in the image. |
| `NLP_LLM_GPU_MEM_FRACTION` | `0.78` | vLLM `gpu_memory_utilization`. With 4B weights we have headroom; can raise to `0.85` to grow KV cache. |
| `NLP_LLM_QUANT` | `awq` | `awq` for T4 (sm_75), `awq_marlin` on Ampere+. |
| `NLP_LLM_ENFORCE_EAGER` | `1` | `1` to skip CUDA-graph capture, `0` to enable. For 4B on T4 try `0` once the rest is stable — graphs help more for smaller models. |

### Local test result (17/05 09:52) — NOT SUBMITTED, regressed

```text
v14c-qwen3-4b local      equiv_rate 0.659    wall-clock  5:10  (5.3× faster than v14)
v14   reference local    equiv_rate 0.754    wall-clock 27:29
v9    reference local    equiv_rate 0.711    wall-clock  3:48
```

Projected cloud: blended ~0.684 (0.75·0.635 + 0.25·0.833). That's above
v14's 0.622 but below v9's 0.734 — not a ship.

**Speed unlock is real** — 0.44 s/q vs v14's 2.35 s/q. The base-image swap
also bought us back a ~30s of vLLM init time (no CUDA-graph capture pain,
no enforce_eager workarounds needed once the stack is coherent).

**Accuracy drop of 0.095 vs v14 / 0.052 vs v9 is bigger than projected.**
Three plausible causes, in order of likelihood:

1. **4B capacity floor.** Qwen3-4B-Instruct typically sits ~5–8 pp below
   Qwen2.5-7B-Instruct on closed-book/extractive QA benchmarks. L2 cross-
   document questions ("how many years between X and Y") are where small
   models fall off — they require multi-fact reasoning, not extraction.
2. **Qwen3-Instruct-2507 paraphrases more.** The 2507 release is tuned hard
   for chat helpfulness, which trains the model to *reword*. Our prompt
   says "quote exactly" but a model with strong post-training pull toward
   paraphrasing disobeys at non-trivial rate. With the cloud AE @ 0.9,
   reworded-but-correct answers score 0.
3. **3 few-shots under-anchors a smaller model.** v14 trimmed 6 → 3
   assuming the 7B handled simple patterns zero-shot. A 4B has less
   zero-shot strength; the dropped examples may have been doing real
   work.

Next move: **v14d-qwen3-8b** (one-line change to `download_models.py`,
swap repo to `Qwen/Qwen3-8B-AWQ`). If acc recovers to ≥0.72, hypothesis (1)
was dominant. If it stays at ~0.66, Qwen3 paraphrase tendency is the
structural issue and we either revert to Qwen2.5 family or rebuild the
prompt with hard literal-quote examples.

### Risks (front and center)

1. **Base-image swap may break local `til test`** — the upstream image runs
   a different Python entrypoint by default. `ENTRYPOINT []` plus our
   explicit `CMD` should override cleanly; if `til test` reports the
   container exiting immediately, that's the regression to look for. RESOLVED:
   `python3` not `python` in the new base — fixed in commit; build now
   completes cleanly.
2. **Image size** — vllm/vllm-openai is ~6 GB, plus our ~2.2 GB LLM
   weights and ~0.6 GB retriever stack. Final image ~10 GB (vs v14's
   ~12 GB — actually slightly smaller). Should not hit any submission
   size limit.
3. **Qwen3 chat template emits different control tokens than Qwen2.5** —
   our stop tokens (`\n\n`, `\nQuestion:`, `\nContext:`) are content-based
   not control-token-based, so this is fine, but worth watching the first
   few outputs in the docker logs to confirm we're not getting `<|im_end|>`
   leakage. (vLLM's default decoder strips EOS so this should be a
   non-issue.)
4. **4B model less accurate on L2 questions** — Qwen3-4B is small. If
   local equiv_rate lands below 0.71 specifically because L2 cross-fact
   composition regressed, the answer is Qwen3-8B-AWQ (~5GB) rather than
   going back to v14's Qwen2.5-7B.

## v14d-qwen3-8b — Qwen3-8B-AWQ on vllm-openai base (17 May)

One-line change from v14c: `NLP_LLM_REPO` flipped to `Qwen/Qwen3-8B-AWQ`
(official Qwen quant of the Qwen3 8B base, thinking mode enabled but
suppressed via `enable_thinking=False` in `apply_chat_template`).
Everything else — base image, prompt, 3 few-shots, retrieval — identical
to v14c.

### Local test result (17/05 ~18:25)

```text
v14d-qwen3-8b local      equiv_rate 0.755    wall-clock 18:33  (1.5× faster than v14)
v14   reference local    equiv_rate 0.754    wall-clock 27:29
v14c  reference local    equiv_rate 0.659    wall-clock  5:10
v9    reference local    equiv_rate 0.711    wall-clock  3:48
```

Cloud submission running. Projected blended ~0.643 = 0.75 · 0.73 + 0.25 ·
0.38 (using v14's local-cloud accuracy gap 0.020 and 18:33 → speed score
1 - 18.5/30 = 0.383).

### What v14d resolved

The clean A/B between v14c (4B) and v14d (8B) — same family, same prompt,
same retrieval, same base image — isolates **model capacity** as the
single variable. v14c lost 0.095 accuracy. v14d recovered all of it. So:

- **Qwen3 paraphrase tendency is NOT the problem**. The 2507 / Instruct
  variants do follow "quote exactly" instructions when given enough
  capacity. The v14c failure was structural — 4B is below the QA capacity
  floor for this corpus.
- **Qwen3-8B is the right base for fine-tuning.** Matches Qwen2.5-7B's
  accuracy ceiling (within noise: 0.755 vs 0.754) at 1.5× speed. The
  speed budget bought by going from 7B → 8B-Qwen3 (smaller activations,
  newer kernels) is real even though param count went up — a counterintuitive
  result that came from the architecture generation jump compounding with
  the upstream base-image stack.

### Why v14d is unlikely to be the final submission

Cloud-projected blended ~0.643 still loses to v9's 0.734 blended. The 18:33
local wall-clock translates to a speed score around 0.38, and accuracy at
0.73 cloud can't compensate for that gap. Unless cloud is significantly
faster than local (we've seen 5-10% variance, not 30%), v14d's blended
will not beat v9's.

So v14d's role is **not to ship**, it's to **be the LoRA-fine-tune base**.

## v15-lora-qwen3-8b — QLoRA fine-tune of v14d on the 883 local examples

The lever: in v7→v8b, fine-tuning RoBERTa-large on the local nlp.jsonl was
worth +0.034 cloud accuracy first try and +0.162 when we matched the
inference chunking distribution at training time. Applying the same lever
to the 30× larger Qwen3-8B should compound — the LLM has more capacity to
absorb Clairos vocabulary, the PCE date format, the "quote verbatim"
behaviour, and the L2 cross-fact composition style. Speed unchanged from
v14d (~18:33 local) — the entire bet is on accuracy.

### Architecture

```
                            v14d (un-tuned base)
question
  → BM25+BGE hybrid retrieval       (unchanged from v9)
  → bge-reranker-base               (unchanged)
  → top-3 doc IDs                   (unchanged)
  → Qwen3-8B-AWQ via vLLM           ← model class from v14d
  → answer

                            v15-lora (this section)
question
  → BM25+BGE hybrid retrieval       (unchanged)
  → bge-reranker-base               (unchanged)
  → top-3 doc IDs                   (unchanged)
  → Qwen3-8B-AWQ + LoRA adapter     ← LoRA is the only new thing
  → answer
```

The original plan loaded the LoRA adapter at inference via vLLM's
`LoRARequest` (`enable_lora=True`, `max_loras=1`, `max_lora_rank=16`),
with no model merge or re-quantisation. That plan is now falsified on the
T4/cloud stack: the vLLM Punica/Triton LoRA kernel crashes, so the only
remaining Qwen3 route would be offline merge + AWQ re-quantization.

### Code changes (17 May)

- [training/nlp/finetune_lora.py](../training/nlp/finetune_lora.py) — new
  QLoRA training script. Loads Qwen3-8B in 4-bit nf4 via bitsandbytes
  (~4.5 GB VRAM for the base), attaches r=16 LoRA adapters to
  `q_proj/k_proj/v_proj/o_proj`, trains 2 epochs with
  `gradient_checkpointing=True`. The original default was
  `bs=2 × grad_accum=4`; actual T4 run needed `bs=1 × grad_accum=8`.
  Inference-distribution training: for each (q, gold_answer, source_docs)
  row in `nlp.jsonl`, builds the *exact* same prompt structure
  `[system, *few_shots, user(question + 3 chunks)]` that `llm_answerer.py`
  uses at inference, with the gold answer as the assistant turn. Loss is
  masked to the assistant turn only via `SFTConfig(completion_only_loss=True)`.
  Chunks are picked from `source_docs` using the v8b-style
  "answer-containing chunk" heuristic; fallbacks to first chunk when the
  answer is paraphrased.
- [src/llm_answerer.py](src/llm_answerer.py) — added `lora_dir` constructor
  param and `NLP_LLM_LORA_DIR` env reading. If the dir contains an
  `adapter_config.json`, vLLM is initialised with `enable_lora=True` and a
  `LoRARequest("v15-lora", 1, lora_dir)` is created once and passed to
  every `generate()` call. Backward-compatible: no adapter dir → identical
  to v14d.
- [Dockerfile](Dockerfile) — added `ENV NLP_LLM_LORA_DIR=/workspace/models/lora`
  and a RUN block that bundles `nlp/models/lora/` into
  `/workspace/models/lora/` if `adapter_config.json` is present at build
  time. Falls through to base-model-only if not.
- [requirements-dev.txt](../requirements-dev.txt) — added `bitsandbytes>=0.43.0`
  and `trl>=0.12.0` (peft and accelerate were already there for ASR).

### Runbook (on the Workbench instance)

Step 1 — pip-install the new training deps:

```bash
cd ~/til
pip install -r requirements-dev.txt
```

Step 2 — train the adapter (actual T4 runtime was ~8h, not 45 min):

```bash
python training/nlp/finetune_lora.py \
    --base Qwen/Qwen3-8B \
    --data /home/jupyter/novice/nlp/nlp.jsonl \
    --docs /home/jupyter/novice/nlp/documents \
    --out  nlp/models/lora \
    --epochs 2 \
    --batch-size 1 \
    --grad-accum 8

# Output: nlp/models/lora/{adapter_config.json,adapter_model.safetensors,
#                          tokenizer*,BASE_MODEL}
```

Actual console output from the completed run:
- `loaded 883 training examples`
- `trainable params: 15,335,424 || all params: 8,206,070,784 || trainable%: 0.1869`
- Eval loss curve: 0.645 → 0.580 → 0.561 → 0.559 over 200 steps
- `load_best_model_at_end=True` retains the lowest eval_loss checkpoint

Step 3 — sanity-check the adapter loads on the base:

```bash
python - <<'PY'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto", device_map="cpu")
model = PeftModel.from_pretrained(base, "nlp/models/lora")
print("adapter loaded OK, adapter modules:", len(list(model.named_modules())))
PY
```

(The base re-download here is BF16 ~16 GB if HF cache is empty. Skip this
step if you trust the trainer output.)

Step 4 — build the v15 image (the LoRA adapter gets bundled because step 2
left it at `nlp/models/lora/`):

```bash
til build nlp v15-lora-qwen3-8b
```

Build cache state: base image cached from v14d, vLLM install cached,
`download_models.py` cached (LLM repo unchanged), `COPY models` and the
LoRA-bundle RUN are new layers (~30s). Total rebuild ~1–2 min.

Step 5 — local A/B vs v14d:

```bash
til test nlp v14d-qwen3-8b           # baseline: 0.755, 18:33
til test nlp v15-lora-qwen3-8b       # gate: local equiv_rate ≥ 0.78
```

Container logs should show:
- `[llm_answerer] LoRA adapter detected at /workspace/models/lora (max_rank=16)`
- `[llm_answerer] loading vLLM ...`  *(includes enable_lora=True in kwargs)*
- `[nlp build] bundled v15 LoRA adapter (Qwen/Qwen3-8B)` *(from the build log)*

Step 6 — superseded. Do not submit this direct-LoRA image now:

```bash
# Historical only:
# til submit nlp v15-lora-qwen3-8b
```

### Superseded direct-LoRA decision rule

This table is retained to explain the original plan, but the cloud/runtime
failures above overrule it.

| Local equiv_rate | Local wall-clock | Action |
|---|---|---|
| ≥ 0.80 | ≤ 22:00 | Submit. New accuracy and likely new blended high. |
| 0.78–0.80 | ≤ 22:00 | Submit. Likely beats v14 cloud (0.734); blended depends on cloud speed. |
| 0.75–0.78 | ≤ 22:00 | Submit only if blended (`0.75·acc + 0.25·(1 − t/30)`) > v14d's projected 0.643. |
| < 0.75 | any | LoRA underfit. Inspect eval_loss curve — if still trending down at end of training, raise to 3 epochs. If flat, raise r=16 → r=32 + extend `target_modules` to MLP layers. |
| any | > 25:00 | Adapter inference overhead higher than expected. Try `max_loras=1` is already set; raise `NLP_LLM_GPU_MEM_FRACTION` to 0.85 to give vLLM more KV cache. |

### Risks (front and center)

1. **VRAM during training.** QLoRA on Qwen3-8B should sit at ~10–12 GB on
   T4. If OOM, halve `--batch-size` to 1 and double `--grad-accum` to 8
   (effective batch size stays at 8). Last resort: reduce `MAX_SEQ_LEN`
   from 2048 → 1536.
2. **Adapter doesn't apply at inference.** Symptom: `til test` runs but
   answers look identical to v14d (no improvement). Check container logs
   for `[llm_answerer] LoRA adapter detected ...`. If absent, the COPY at
   build time didn't pick up `nlp/models/lora/adapter_config.json`.
3. **Quant-format mismatch (BnB-4bit train base vs AWQ inference base).**
   Both quantise the same underlying BF16 weights from different
   starting points. LoRA matrices were tuned against BnB-nf4 dequantised
   activations; they're applied against AWQ-int4 dequantised activations.
   Small quality drift is possible (~1pp); should be far below the
   training lift.
4. **Train/inference prompt drift.** The system prompt in
   `finetune_lora.py:SYSTEM_PROMPT` MUST match
   `llm_answerer.py:_DEFAULT_SYSTEM_PROMPT` exactly. If they diverge
   silently, the model sees a different prompt at inference and the
   adapter underperforms. Currently duplicated; consider a single source
   of truth if iterating further.
5. **vLLM 0.9 LoRA+AWQ combo.** Documented to work but historically had
   bugs. If we see crashes, downgrade to v0.8.x or fall back to
   merging+requantizing offline (slower path but more robust).

### Expected lift (superseded projection)

Comparing the v7→v8b lift (+0.162 cloud on RoBERTa-large extractive) to
what a generative 8B LoRA fine-tune should give:

- The RoBERTa-large fine-tune could only move *extracted-span quality* —
  it was structurally capped at literal source substrings.
- An LLM fine-tune additionally moves *generation style* — PCE date
  format, codename casing, "approximately N years" prefixes, money unit
  scaling. These cover the 481/883 cases where the gold answer is
  non-literal.
- Conservative estimate: **+0.05 cloud accuracy** over v14d. Realistic:
  **+0.05 to +0.10**. Optimistic: **+0.10 to +0.15** (matching v8b's
  lift scaled by capacity).

Projected v15 cloud blended: `0.75 · 0.78 + 0.25 · 0.38 = 0.68` (conservative)
to `0.75 · 0.85 + 0.25 · 0.38 = 0.73` (optimistic, near v9 blended).

### Update — cloud reality vs projection (18 May ~04:15)

All projections above assumed v15 would actually serve requests on cloud
like v14d does locally. **Cloud reality: vllm/vllm-openai base image
times out or crashes on every submission**, regardless of model:

```text
v14c-qwen3-4b      18/05 cloud   TIMEOUT  (local was 5:10)
v14d-qwen3-8b      18/05 cloud   TIMEOUT  (local was 18:33)
v15-lora-qwen3-8b  18/05 10:12   0.000 / 1.000 / 700/700 errors
```

v14 (Qwen2.5-7B-AWQ, NGC base) ran cloud at 21 min and succeeded —
that's our only proven cloud path. Most likely cause for the v14c/d
timeouts is image-pull / cold-start overhead on the novel
vllm/vllm-openai base; v15-lora's 700/700 errors are the broken Triton
LoRA kernel exceptions propagating to FastAPI (vs. silent fallback locally).

The offline AWQ-merge path also broke in two ways: **autoawq** is
deprecated (final dev release has broken `from awq import
AutoAWQForCausalLM`), and **llm-compressor 0.10** OOMs on T4 at default
sequential_targets, then throws `TypeError: 'NoneType' object is not
subscriptable` inside the symbolic-trace subgraph forward when sliced
at Linear granularity — likely a Qwen3 GQA edge case.

Update: the script now defaults to **GPTQ W4A16** (`--quant-method gptq`,
`--max-seq-length 256`) as the T4-safe retry path. GPTQ avoids the AWQ
smoothing/propagation pass that produced the Qwen3 GQA `NoneType` failure.
Keep GPTQ on the default block-level sequential pipeline; forcing per-Linear
sequencing hits the same symbolic-trace failure at `o_proj`.
GPTQ then OOMed consistently at `model.layers.29.mlp.down_proj` during
`torch.cholesky_inverse(H)` with only ~250 MiB free, so the script now defaults
to `--gptq-ignore-down-proj-from-layer 29`. That leaves the seven late
`down_proj` modules (`29-35`) in BF16 and still quantizes the rest of the model,
which is the only T4 path left that can plausibly produce a bootable artifact.

Follow-up result: the layer-29 skip completed quantization and saved
`nlp/models/llm-merged/`. Docker build copied a 6.61 GB model context and the
container reached healthy state. The next `til test` failures were host
evaluator environment issues, not model/container failures:

- Running `til test` while `~/quant-venv` is active fails immediately with
  `ModuleNotFoundError: No module named 'dotenv'` because the quant venv is not
  the normal Workbench test env.
- Running from base env then fails loading `ModernBertForSequenceClassification`
  because broken optional `torchvision` is still installed after the earlier
  `llmcompressor` torch downgrade (`RuntimeError: operator torchvision::nms
  does not exist`). Fix by deactivating the quant venv and uninstalling
  `torchvision` from the host env before rerunning `til test`.

**Net**: the LoRA training payoff is still unproven until the host evaluator
env is repaired and `til test nlp v15-merged-qwen3-8b` reaches scoring. Two
real paths forward:

1. **Repair the Workbench host test env and score the GPTQ-merged image.**
   If it scores above v14/v14d locally, decide whether to submit despite the
   known vllm-openai cloud-timeout risk.
2. **Retrain LoRA on Qwen2.5-7B**, ship on NGC base. 8h retrain, but
   uses the proven cloud-shippable v14 stack and applies the v8b
   fine-tune lever cleanly. Highest-EV remaining path if NLP is to be
   pushed further.

For the qualifier as-is: v9-doc-ensemble holds blended (0.683/0.886 =
0.734), v14-llm-rag holds accuracy (0.734/0.286). Neither moves with
v15-family until one of the two paths above lands.

Follow-up score: after repairing the host evaluator env, `v15-merged-qwen3-8b`
scored **0.659** locally. That is exactly the v14c 4B / v15 runtime-corruption
bucket and far below the v14d 8B base score of 0.755, so it is not a submit.
This does **not** prove the LoRA training was ineffective by itself; the score
is too pathological for that conclusion. The likely failure chain is one of:

1. The image did not actually serve the intended merged GPTQ artifact.
2. The merged artifact was built from a stale / wrong adapter.
3. The T4-safe GPTQ escape hatch damaged the model enough to erase the 8B base
   behavior (`calib_n=32`, seq_len=256, and late `down_proj` modules skipped).

`llm_answerer.py` now prints model provenance at boot (`model_type`,
`quant_method`, and `MERGED_FROM`) so the next `docker logs ...` can separate
"wrong thing served" from "right thing served but quantization/merge failed".

## v16-deberta-v3 — extractive-reader retry (18 May)

Decision: try one more fast-reader path before spending more time on LLM
serving. The old `v13b-deberta` run already tried DeBERTa-v3-large and failed
locally (`0.667` vs v9's `0.711`, 9:06 vs 3:48). The only reason to reopen it
is a materially different gate:

- restore the Dockerfile default to `NLP_ANSWERER=extractive` so the image
  actually serves the QA reader instead of the v14/v15 LLM path;
- skip the LLM download for this tag so build/runtime stay fast;
- bundle `nlp/models/deberta-finetuned-squad2/` if present;
- reduce runtime QA max sequence length to 256 via `NLP_QA_MAX_SEQ_LEN=256`;
- train for **one epoch at lr=1e-5**, because v13b's eval loss was best at
  epoch 1 and then overfit.

Run:

```bash
python training/nlp/finetune_qa.py \
  --base-model deepset/deberta-v3-large-squad2 \
  --use-answer-chunk \
  --epochs 1 \
  --lr 1e-5 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --gradient-checkpointing \
  --output nlp/models/deberta-finetuned-squad2

til build nlp v16-deberta-v3
til test nlp v16-deberta-v3
```

Gate: only submit if local `NLP RAG QA Accuracy` clears v9's 0.711. If not,
do not keep iterating DeBERTa; v13b plus this v16 gate are enough evidence.

Result: **0.692 local**, question-answering loop **8:27**. This improved over
the old DeBERTa run (`v13b` was 0.667) but still missed the v9 local gate
(`0.711`) and remained far slower than v9 (`~3:48`). Verdict: do not submit.
The lower-LR one-epoch recipe reduced the damage but did not overturn the core
finding that RoBERTa-large transfers better than DeBERTa-v3-large on this
small, synthetic Clairos span-extraction corpus.

## v17-modernbert — stock vs fine-tuned QA A/B (18 May)

Purpose: answer the skepticism about whether QA-head training is still useful.
ModernBERT gives a clean fast-reader A/B:

1. **Stock** `smangla/ModernBERT-base-squad2` with no Clairos fine-tune.
2. **Fine-tuned** same checkpoint on our retained chunked-context examples.

Code changes:

- `download_models.py` now downloads `smangla/ModernBERT-base-squad2` into
  `/workspace/models/modernbert-base-squad2`.
- `nlp_manager.py` now supports `modernbert-finetuned-squad2` and
  `modernbert-base-squad2`, plus a direct `NLP_QA_MODEL_DIR` override.
- Extractive priority:
  override > modernbert-finetuned > modernbert-base-squad2 >
  deberta-finetuned > roberta-finetuned > roberta-base.

Run stock first, before creating `nlp/models/modernbert-finetuned-squad2`, so
the manager naturally falls through to the stock ModernBERT QA model:

```bash
til build nlp v17-modernbert-stock
til test nlp v17-modernbert-stock
```

Then train and test:

```bash
python training/nlp/finetune_qa.py \
  --base-model smangla/ModernBERT-base-squad2 \
  --use-answer-chunk \
  --epochs 1 \
  --lr 1e-5 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --output nlp/models/modernbert-finetuned-squad2

til build nlp v17-modernbert-ft
til test nlp v17-modernbert-ft
```

Interpretation: if stock and fine-tuned are both below v9's 0.711 local, stop
training extractive readers. If stock is decent and fine-tuning adds a clear
lift, ModernBERT may be the next shippable blended-score path.

The older projection above is now superseded by cloud/runtime evidence.
Do not spend more time on prompt trimming for Qwen3 until the serving path
itself is solved.

### Training result (18 May ~01:30 SGT)

```text
config:    bs=1 grad_accum=8  (T4 VRAM ~14GB peak; bs=2 OOMed at step 1)
duration:  8h 06m on T4
steps:     200 / 200 (full 2 epochs)
final:     train_loss 0.57   eval_loss 0.559   mean_token_acc 0.876
```

Loss curve (monotonic):

```text
step  10  loss 3.04                                  acc 0.520
step  50  loss 0.68  eval 0.645                      acc 0.864
step 100  loss 0.58  eval 0.580                      acc 0.872
step 150  loss 0.57  eval 0.561                      acc 0.876
step 200  loss 0.57  eval 0.559                      acc 0.876
```

The model is correctly predicting **87.6% of the gold answer tokens**
under teacher-forcing on the eval split. For comparison, v8b's RoBERTa
fine-tune which delivered the +0.162 cloud accuracy lever had its best
eval_loss at 0.872 — v15's 0.559 is 36% lower. Generative SFT loss is
not directly comparable to extractive span CE, but the curve shape and
plateau timing are textbook healthy.

Eval_loss was still drifting down at epoch 2 (0.561 → 0.559) so a third
epoch would have given diminishing returns at proportionally large
wall-clock cost. Calling 2 epochs the right stopping point. `load_best_model_at_end=True`
retained the epoch-2 checkpoint.

Practical notes from this run for any future LoRA training:
- `bs=2` OOMs on T4 16GB despite QLoRA. Default to `bs=1 grad_accum=8`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation
  (set it in the shell before launching).
- Each eval pass takes ~9 min (89 examples × bs=1 bf16 fwd). 4 eval
  passes per run cost ~36 min. For future iterations, increase
  `eval_steps` from 50 to 100 (2 evals instead of 4) to save ~18 min.
- Total budget on T4: ~8h for 2 epochs on this size dataset. If running
  multiple LoRA experiments, kick them off overnight.

Adapter saved to `nlp/models/lora/`:
- `adapter_config.json`
- `adapter_model.safetensors`  (~60 MB)
- `tokenizer*`
- `BASE_MODEL` (contains `Qwen/Qwen3-8B`)

### Superseded direct-LoRA build + test sequence

This sequence is kept for provenance only. It is **not** the current
recommended path because vLLM's LoRA kernel crashes on the T4/cloud stack.

```bash
# Verify adapter is in place
ls nlp/models/lora/adapter_config.json && echo OK

# Build (mostly cached from v14d; only the COPY models + LoRA-bundle layers re-run)
til build nlp v15-lora-qwen3-8b
# Expect in build log:
#   [nlp build] bundled v15 LoRA adapter (Qwen/Qwen3-8B)

# Local A/B vs v14d
til test nlp v14d-qwen3-8b           # baseline: 0.755, 18:33
til test nlp v15-lora-qwen3-8b       # target: ≥ 0.78
# Expect in container log (early in load):
#   [llm_answerer] LoRA adapter detected at /workspace/models/lora (max_rank=16)
```

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

### Final NLP verdict on post-processing layers

Every post-v9 post-processing swing regressed monotonically:

```text
v10-template-lite      0.683 / 0.882   neutral (-0.001 blended)
v11-canonical-answer   0.680 / 0.881   -0.004 blended
v8a-genqa              0.652 / 0.836   -0.040 blended
v12-candidate-ranker   0.642 / 0.829   -0.045 blended
v13a-heuristic         (0.663 local, NOT SUBMITTED — same as v12)
```

The POST-PROCESSING architecture is at its ceiling on this corpus. Confirmed
dead levers: paragraph chunking, low-conf fallback, rapidfuzz spans, narrow rule
templates, full-doc canonicalization, generative answers, candidate reranking.

The final **EXTRACTOR retrain** path is also now exhausted. v7-v1 (+0.034) and
v8b (+0.162) both came from training the QA head, but `v13b-deberta` did not
transfer: local 0.667 vs v9's 0.711, 9:06 vs 3:48. See section below.

## v13a — AE-trained candidate ranker (16/05 — NOT SUBMITTED)

Local result: val top-1 0.384, val oracle 0.814 (177 held-out questions).

Diagnosis: the ranker has access to candidates that would pass AE for 81% of
val questions (oracle), but the logistic scoring only picks the right one 38% of
the time. The 20-feature set can't discriminate among same-type candidates
("2178" vs "2178 CE" vs "around 2178" for a year question — all match
`wants_years` + `has_year_unit`).

Heuristic-only test (no learned JSON, with structural fixes — `source_qa=5.0`,
`source_rule=4.0`, dropped `_RELATION_PHRASE_RE`, `n_sentences=8`, `echoes_question=-1.0`):
local 0.663 — identical to v12. The candidate POOL itself dilutes v9
regardless of how it's scored. Candidate-ranker architecture confirmed dead.

Three artefacts kept for future ranker work if we ever revisit:
- [training/nlp/train_answer_ranker.py](../training/nlp/train_answer_ranker.py) — now does ModernBERT-AE labeling + 80/20 split + oracle/top-1 reporting.
- `nlp/models/answer_ranker.json` — fitted weights from the AE-labeled training run.
- `_answer_candidates()` machinery in `nlp_manager.py` stays in tree; disable with `NLP_ANSWER_RANK_MODE=off` (default heuristic-only path still routes through it).

## v13b — DeBERTa-v3-large QA retune (16/05, failed local gate)

Hypothesis: the answer-form gap (`retrieval_hit_diff` ~395 cases) is a QA-head
problem, not a post-processing problem. Every post-processing swing has
regressed; every QA retrain has hit (+0.034 v7-v1, +0.162 v8b). DeBERTa-v3 is
documented +1-2% over RoBERTa-large on extractive QA benchmarks. With the v8b
chunked-context training recipe held fixed and just the base model swapped,
the only hypothesis at risk is "does DeBERTa-v3 transfer here as well as it
does on SQuAD-style benchmarks".

Manager + Dockerfile changes (16/05):
- Manager: `QA_DEBERTA_FINETUNED_DIR = MODEL_DIR / "deberta-finetuned-squad2"`
  added as the highest-priority QA dir; fallback ladder is now
  `deberta > flan-t5-finetuned (gen) > roberta-finetuned-squad2 > stock`.
- Dockerfile: bundles `nlp/models/deberta-finetuned-squad2/` when present.
- Both safe: if the dir doesn't exist locally, image falls through to v9's
  RoBERTa-large weights.

Workbench runbook:

```bash
# Tokenizer needs sentencepiece (already installed). Train on Workbench GPU:
python training/nlp/finetune_qa.py \
  --base-model deepset/deberta-v3-large-squad2 \
  --use-answer-chunk \
  --epochs 5 \
  --output nlp/models/deberta-finetuned-squad2

# Build & local A/B
til build nlp v13b-deberta
til test  nlp v9-doc-ensemble                  # baseline 0.711
til test  nlp v13b-deberta                     # final result: 0.667, failed

# Do not submit: final local result was 0.667 vs v9's 0.711.
```

Decision rule: same as before — local `equiv_rate` must clear v9's 0.711 under
the synced upstream `test_nlp.py`. If `v13b-deberta` local < 0.711, drop and
freeze on v9. It landed at 0.667, so v13b is dropped and NLP is frozen.

### Training run (16/05 ~23:30 SGT) — completed in ~6 min on T4

OOM at default batch=8 on the 14.5 GB T4 (DeBERTa-v3-large disentangled
attention uses ~3× the activation memory of RoBERTa-large). Patched
`training/nlp/finetune_qa.py` with `--gradient-accumulation-steps` and
`--gradient-checkpointing` flags (commit `8296c29`). Ran:

```bash
python training/nlp/finetune_qa.py \
  --base-model deepset/deberta-v3-large-squad2 \
  --use-answer-chunk \
  --epochs 5 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --gradient-checkpointing \
  --output nlp/models/deberta-finetuned-squad2
```

Result: 338 train / 37 val (same examples as v8b: 375 retained after
`answer_not_in_doc=508` filter). 220 optimizer steps × 1.76 s/step = 6:26.

```text
epoch 1   train 0.96-1.51   eval_loss 1.58   ← BEST, saved
epoch 2   train 0.53-0.64   eval_loss 1.62
epoch 3   train 0.24-0.27   eval_loss 1.76
epoch 4   train 0.09-0.29   eval_loss 2.83
epoch 5   train 0.01-0.09   eval_loss 2.34
```

`load_best_model_at_end=True` → epoch 1 weights are what got saved to
`nlp/models/deberta-finetuned-squad2/`.

**Yellow flag**: eval_loss 1.58 is ~2.5× v7-v1's epoch-1 0.614 on the same
data. DeBERTa-v3-large didn't calibrate to our QA distribution as cleanly as
RoBERTa-large did. Train loss collapse 1.51→0.01 across 5 epochs is classic
overfit on 338 examples. eval_loss is span-token CE though, not equiv_rate, so
the only honest test is `til test nlp v13b-deberta` against v9's 0.711.

### Local test result — NOT SUBMITTED

```text
v9-doc-ensemble    0.711  3:48  (baseline)
v13b-deberta       0.667  9:06  ← FAILED GATE; -0.044 accuracy, 2.4× slower
```

Below the 0.711 gate. DeBERTa-v3-large fine-tuned on 338 examples and
load_best picked epoch 1 (eval_loss 1.58), but the model still underperforms
RoBERTa-large on this corpus. Speed also regressed significantly (9:06 vs 3:48)
from DeBERTa's disentangled attention overhead at inference.

**NLP is frozen at v9-doc-ensemble (0.683/0.886) for the rest of the qualifier.**
Confirmed dead levers now total eight: paragraph chunking, low-conf fallback,
rapidfuzz spans, narrow rule templates, full-doc canonicalization, generative
answers (Flan-T5), candidate reranking (v12/v13a), DeBERTa-v3-large QA retune.

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
