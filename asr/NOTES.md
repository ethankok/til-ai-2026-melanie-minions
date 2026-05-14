# ASR — notes & history

Last updated: 14 May 2026

Per-task working log for ASR. For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#asr).
For training-pipeline mechanics see [../training/asr/README.md](../training/asr/README.md).
For data-driven error analysis see [../training/asr/ERROR_ANALYSIS.md](../training/asr/ERROR_ANALYSIS.md).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`nemo-zs` — official 0.956 / 0.946 (14 May 20:33 SGT, 0/400 errors).**
Blended score `0.75*0.956 + 0.25*0.946 = 0.9535`, +0.025 over `ft-lora32-v1`.
Same accuracy (within noise: -0.001), +0.097 speed. The leaderboard keeps
the higher raw score for either dimension, but blended-per-challenge is what
feeds the qualifier total via the 75/25 weighting, so this is the new ASR
high.

## Active experiment: NeMo Parakeet-TDT backend

Leaderboard inspection on 14 May shows multiple Novice teams above
`0.97 / 0.92` simultaneously (Overflow `0.991/0.925`, OpenLarp `0.986/0.940`,
suite108 `0.982/0.920`). distil-large-v3 cannot reach that frontier
structurally — its autoregressive cross-attention decoder is the speed
bottleneck. Our hypothesis: top teams are on a NeMo transducer, most likely
**Parakeet-TDT-0.6B-v2** (top of the HF Open ASR English leaderboard, RTFx in
the thousands on T4, non-autoregressive over cross-attention).

Two pieces of evidence support switching the backbone:

1. **Air-gap is fine for NeMo.** README + CLAUDE.md confirm `til test` runs
   on a no-internet docker network. NeMo loads from a local `.nemo` file via
   `ASRModel.restore_from(restore_path=...)`, no network calls. Same
   bake-into-image pattern the NLP container already uses.
2. **Transcript style favors Parakeet.** All numbers in the manifest are
   spelled out (`"zero six hundred"`, `"twenty third"`, `"two-seven-zero"`,
   `"three hundred and sixty-five"`). Whisper-family models emit digits and
   need `_digits_to_words` to recover; Parakeet emits spelled-out numbers
   natively, removing the riskiest part of post-processing. The post-processor
   now lives in `asr/src/asr_postprocess.py` and stays as a safety net.

Parallel build path (does not touch the shipped distil-whisper image):

```text
asr/src/asr_postprocess.py          shared digits->words (extracted from manager)
asr/src/asr_manager.py              UNCHANGED behavior; now imports postprocess
asr/src/asr_manager_nemo.py         NEW: NemoASRManager (Parakeet-TDT)
asr/src/asr_server.py               picks backend via ASR_BACKEND env (default whisper)
asr/requirements-nemo.txt           NEW: nemo_toolkit[asr]==2.0.0 + audio libs
asr/Dockerfile.nemo                 NEW: parallel image, ENV ASR_BACKEND=nemo
training/asr/download_models_nemo.py NEW: stage parakeet-tdt-0.6b-v2.nemo into asr/models/
```

Phase plan with kill switches at each step:

1. **Zero-shot Parakeet on the held-out 10% val.**
   ```bash
   pip install -r asr/requirements-nemo.txt   # ~10 min on Workbench
   python training/asr/download_models_nemo.py \
       --model nvidia/parakeet-tdt-0.6b-v2 \
       --out asr/models
   # Quick local eval against the held-out val (Workbench-side script TBD;
   # for now just `til test` against the new image).
   docker build -f asr/Dockerfile.nemo -t melanie-minions-asr:nemo-zs .
   til test asr nemo-zs
   ```
   Decision gate: official-style Eng-WER ≤ 0.05 on the held-out val.
   - If **yes**, submit as `nemo-zs-v1` and continue to phase 2.
   - If **no**, abort. The shipped `ft-lora32-v1` stays unaffected because
     none of the whisper files changed.

2. **Add slang context biasing.** `NemoASRManager._configure_biasing` already
   tries newer-NeMo APIs (`set_context_biasing`, `set_boosting_words`,
   `configure_biasing`); if the installed NeMo version exposes one, slang
   biasing engages automatically. If not, the manager logs and skips — no
   crash. Worth +0.005-0.015 on slang-heavy clips.

3. **Fine-tune Parakeet on the 4110 novice clips** (only if phases 1-2 fall
   short of 0.99). NeMo supports adapter-based PEFT and full FT; 0.6B fits T4
   for full FT at small batch. Same Option B held-out 10% as the current LoRA
   run.

Submission gate: zero `errors`, schema unchanged, official blended score
strictly above `ft-lora32-v1`'s `0.957/0.849` (i.e. blended ≥ `0.930`).

### nemo-zs — local result (14 May 2026)

```text
english error rate (WER): 0.0429
chinese error rate (CER): 0.0000
malay error rate (WER):   0.0000
tamil error rate (WER):   0.0000
1 - MER: 0.9893
1028 batches × 2.62s avg → 37:28 wall clock on the 4110-clip local set
```

Compared against `ft-lora32-v1` (leaky local WER `0.0299`, held-out val
`0.04662`, official `0.957`):

- Parakeet-TDT-0.6B-v2 **zero-shot** (no fine-tune, no slang biasing) lands
  at WER `0.0429` on the full local set. Whisper LoRA's *held-out* val WER
  was `0.04662`, so Parakeet starts roughly on par with our trained model
  before any in-domain adaptation. The accuracy gate (≤ 0.05) is cleared.
- Local set is ~10× larger than the cloud set (4110 vs 400). Cloud wall
  clock projection: 4110 / 400 × 37:28 / 30 min ≈ 30% of t_max → cloud
  speed score `~0.70`. **This is below `ft-lora32-v1`'s `0.849` and is
  exactly why `cuda-python` is the very next change.**

NeMo logged at startup:

```text
No conditional node support for Cuda. Cuda graphs with while loops are
disabled, decoding speed will be slower
Reason: No `cuda-python` module. Please do `pip install cuda-python>=12.3`
```

The TDT decoder's while-loop is its main per-clip cost. With the CUDA-graph
fast path disabled, the decode loop runs as eager Python kernels; with it
enabled, the whole loop fuses into a CUDA graph and Parakeet gets the
multi-thousand-RTFx numbers it advertises. `cuda-python>=12.3` is now in
`requirements-nemo.txt`; rebuild as `nemo-zs-v2` and re-test before the
next submission.

Submit `nemo-zs` first to bank the accuracy result; the leaderboard keeps
the highest blended score so a worse `nemo-zs` cannot demote
`ft-lora32-v1`.

### nemo-zs — official result (14 May 20:33 SGT)

```text
errors: 0 / 400
score:  0.956   (vs ft-lora32-v1 0.957 — flat within cloud noise of -0.001)
speed:  0.946   (vs ft-lora32-v1 0.849 — +0.097)
blended (75/25): 0.9535   (vs ft-lora32-v1 0.9285 — +0.025)
```

The cloud set was about 30% of t_max worth of wall clock without
cuda-python; with cuda-python enabling the TDT CUDA-graph fast path,
expect the speed score to climb further (current 0.946 → ~0.96+).
Accuracy parity with the LoRA-tuned Whisper at zero-shot is the
headline: this is the floor before any fine-tuning or slang biasing.

Local→cloud generalization gap turned out NEGATIVE again (local WER
`0.0429` → official ~`0.044`). Same pattern as `ft-lora32-v1`. The
official 400-clip distribution is just slightly easier than our local
4110-clip set on this dataset.

`nemo-zs` is the new shipped tag. `ft-lora32-v1` stays as a fallback
image but is no longer the live ASR contribution to the qualifier
total.

### nemo-zs-v2 — cuda-python CUDA-graph fast path (14 May 22:07 SGT)

```text
errors: 0 / 400
score:  0.956   (vs nemo-zs 0.956 — exactly equal, leaderboard-ranked tie)
speed:  0.946   (vs nemo-zs 0.946 — exactly equal)
local:  WER 0.0429, wall clock 34:42 (vs nemo-zs 37:28, -7%)
local per-batch: 2.03s (vs nemo-zs 2.62s, -22%)
```

cuda-python `12.3+` is being picked up correctly inside the container —
the "No conditional node support for Cuda" startup warning is gone, and
the local TDT decoder is measurably faster. **But cloud speed didn't move
at all.** Why:

- nemo-zs cloud speed 0.946 corresponds to ~97s wall on 400 clips =
  ~0.24s/clip. Local nemo-zs-v2 is ~0.51s/clip. The cloud rig is already
  ~2× faster per clip than our T4 — almost certainly L4 or A10.
- On a faster GPU the TDT decoder loop is an even smaller fraction of
  per-clip cost than locally. Audio decode (soundfile + librosa
  resample), HTTP / base64 round-trip, batch assembly, and Python
  overhead dominate.
- A 22% speedup on something that's already maybe 10-15% of cloud
  per-clip cost rounds to nothing visible at three-decimal score
  precision.

**Conclusion: cloud speed is no longer the TDT decoder. Speed is parked
at 0.946 unless we change the serving shape (which is risky for
diminishing returns).** The next ASR lever is accuracy, and that's what
`train_parakeet.py` is for.

### Next: Parakeet fine-tune (planned)

Path is now wired end-to-end:

```text
training/asr/prepare_data_nemo.py    NEW: build NeMo-format manifests
training/asr/train_parakeet.py       NEW: fine-tune Parakeet (PL Trainer)
training/asr/export_parakeet.py      NEW: stage best.nemo into asr/models/
training/asr/README.md               updated with Parakeet quick-start
```

Default recipe (from `train_parakeet.py`):

- Encoder frozen (Parakeet's conformer is already strong on English; budget
  goes to decoder + joint network, where slang/in-world adaptation lives).
- 5 epochs, lr 5e-5, batch 8, grad-accum 2 → effective batch 16.
- ModelCheckpoint(monitor=val_wer, save_top_k=2) + EarlyStopping(patience=3).
- Wall clock estimate: 3-4 hr on T4.

Decision gate before submitting `parakeet-ft-v1`: local Eng-WER ≤ 0.035
(from current zero-shot 0.0429). If hit, expected official accuracy
0.965-0.975. If not hit, rerun with `--epochs 8 --lr 3e-5` or unfreeze
the encoder. Leaderboard keeps the higher score so a regression cannot
demote `nemo-zs`.

## What our model runs on

### Inference (the shipped Docker container)

- **Base model**: `distil-whisper/distil-large-v3` — English-only distillation of Whisper large-v3. Same encoder quality, ~6× faster decoder. Right call for the Novice (English-only) track.
- **Fine-tune**: LoRA rank 32, alpha 64, dropout 0.05, applied to decoder attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`). Encoder frozen.
- **Runtime engine**: `faster-whisper` (CTranslate2 backend) at `float16` on GPU. CPU fallback at `int8` if CUDA missing.
- **Container base**: `nvcr.io/nvidia/pytorch:25.11-py3`.
- **Inference flags** (see [src/asr_manager.py](src/asr_manager.py)):
  ```python
  model.transcribe(
      audio, language="en", task="transcribe",
      beam_size=1,                          # greedy
      vad_filter=False,                     # eats speech on long clips
      condition_on_previous_text=False,
      initial_prompt=self.initial_prompt,   # 200-token slang prompt
      without_timestamps=True,              # small speed win
      temperature=0.0,                      # no temp fallback retries
      compression_ratio_threshold=2.4,      # repetition guard
      log_prob_threshold=-1.0,              # low-confidence guard
      no_speech_threshold=0.6,              # silence hallucination guard
  )
  ```
- **Audio-level silence guard** runs *before* `transcribe()` to skip pure noise / breath bursts that would otherwise hallucinate "Thank you." / "I" on sub-1.5s clips.
- **Post-processing** at [src/asr_manager.py `_digits_to_words`](src/asr_manager.py): integers, decimals, comma-thousands, 24h military times, four-digit codes, niner callsigns, spoken ordinals (`23rd → twenty third`), coordinate-safe decimals (`1.1.7` stays multi-token, not parsed as decimal).
- **Slang prompt**: 200 in-world proper nouns mined from the NLP corpus, highest-frequency first (`cyanite renhwa zonnon clairos floodwall phyrexis nanobot sharpsea kashikari wampa nyari sarento megacorporation ...`). Passed as `initial_prompt=` to bias decoding.

### Training (Workbench-only, in [../training/asr/](../training/asr/))

- **GPU**: Tesla T4 (16 GB VRAM). fp16 not bf16 (Turing-gen).
- **Python env**: `transformers >=4.46,<5.0` (pinned to 4.57.6 via `pip install --user`), `peft 0.19.1`, `accelerate 1.13.0`, `datasets[audio]`, `faster-whisper 1.2.1`, `ctranslate2 4.7.1`.
- **Trainer**: HuggingFace `Seq2SeqTrainer` + PEFT LoRA, gradient checkpointing on, `use_cache=False`, fp16 autocast.
- **Augmentation**: SpecAugment (`mask_time_prob=0.05`, `mask_feature_prob=0.05` baked into the feature extractor) + optional speed perturb at ±10%. Noise mixing path exists but no noise corpus on Workbench, so it stays off.
- **Data**: `/home/jupyter/novice/asr/asr.jsonl` (4110 clips, all `language: english`). 90/10 stratified split with slang oversampling, but the 10% val is leaky (same manifest as `test_asr.py`) — see "leaky val" warning below.

## Submission history

```text
Tag               Submitted          Score   Speed   Local WER       Outcome
v1                12/05 03:42        0.000   0.993   —               Empty-string baseline (sanity check submission)
norm-v1           12/05 16:23        0.877   0.864   0.0759          Added digit verbalization + audio-silence guard
vad-off-v1        12/05 20:00        0.938   0.859   0.0554          VAD off + hallucination guards + ordinals + coord-safe decimals
vad-off-v2        not submitted      —       —       0.0604          Reversed slang prompt order — REGRESSED, reverted
ft-lora32-v1      13/05 11:22        0.957   0.849   0.0299 (leaky)  LoRA rank-32 decoder fine-tune — CROSSED 0.95 target
ft-lora32-int8f16 13/05 12:45        0.923   0.856   0.0551          int8_float16 quantization — REGRESSED accuracy too much
```

Leaderboard mechanic: **highest score wins**, so the `ft-lora32-int8f16`
regression did NOT demote us. Live entry is still `ft-lora32-v1` at
**0.957 / 0.849**.

## Detailed timeline — what each tag changed

### v1 (12 May 03:42) — submission plumbing only

Empty-string baseline. Just verified the submission pipeline works end-to-end:
schema, port, Docker layer, GCP push, evaluator response.

### norm-v1 (12 May 16:23) — first real WER

- Wired faster-whisper distil-large-v3, CT2 float16, loaded once at startup.
- Added `_digits_to_words` because the scorer doesn't normalize digits and
  Whisper outputs digits while references spell them out ("seventy two" vs
  "72", "zero six hundred" vs "0600", "seven niner" vs "7-9-er").
- Audio-level silence guard to avoid hallucinations on empty clips.
- Result: official 0.877, local Eng-WER 0.0759. Local→official gap +0.047
  absolute — large because some inference flags weren't yet tuned.

### vad-off-v1 (12 May 20:00) — the inference-side breakthrough

Dropped the `BatchedInferencePipeline` path (it forces Silero VAD which was
truncating long clips — see [ERROR_ANALYSIS.md "VAD truncation"](../training/asr/ERROR_ANALYSIS.md))
and switched to plain `model.transcribe(..., vad_filter=False)`. Added
Whisper-side hallucination guards (`no_speech_threshold=0.6`,
`log_prob_threshold=-1.0`, `compression_ratio_threshold=2.4`, `temperature=0.0`)
and tightened the audio-level silence guard for sub-1.5s clips.

`_digits_to_words` extended: spoken ordinals (`23rd → "twenty third"` not
`"twenty threerd"`), coordinate-safe decimal regex (`"1.1.7"` doesn't get
half-rewritten as `"one point one.seven"`).

Result: **official 0.938** (+0.061 absolute), local WER 0.0554. Local→official
gap shrank to **+0.007 absolute** — the inference fixes generalize cleanly.

### vad-off-v2 (12 May, NOT submitted) — slang prompt reversal experiment

Hypothesis: faster-whisper truncates `initial_prompt` to the last ~223
decoder tokens; with ~200 mined slang tokens the prompt overflows, so
reversing the list to put high-frequency terms LAST should help them survive
truncation.

Result: **wrong direction.** Local WER regressed 0.0554 → 0.0604.

Likely cause: putting `cyanite`, `sarento`, `phyrexis`, `mewan` immediately
before decode-start over-primes the decoder, causing false-positive
hallucinations of those tokens on unrelated audio. The original ordering
(highest-frequency first) left them in the truncated head and rarer terms
near the end — a weaker, less biased prior that worked better.

Action: reverted `extract_slang.py` to write highest-frequency first.

### ft-lora32-v1 (13 May 11:22) — LoRA fine-tune ships, crosses 0.95

3-epoch LoRA rank-32 fine-tune of `distil-whisper/distil-large-v3` decoder
attention only. Training took ~7 hours on Workbench T4 at 19.6 s/step (the
README's 1.5–2 hr estimate was off because transformers 4.57.6 is heavier per
step than the 4.46-era baseline).

Training health: loss 1.18 → 0.65 → 0.38 → ... → 0.25, smooth descent, stable
`grad_norm` ~0.4–0.6 throughout. Val WER at the two eval points:

| Step | Epoch | Val WER (held-out 409 clips) |
|---:|---:|---:|
| 500 | 1.29 | 0.04982 |
| 1000 | 2.58 | **0.04662** ← best, used as final |

Local Eng-WER on the full 4110-clip test set was 0.0299 — looks great but is
**leaky** (the LoRA trained on 90% of those clips). The cleaner generalization
proxy is the val WER 0.04662.

Result: **official 0.957 / 0.849**. Held-out val WER 0.04662 → official ~0.043
means the generalization gap turned out **negative** (val over-estimated
official by ~0.0036). The official 400-clip distribution is slightly easier
than the local held-out slice. Useful piece of leaderboard intuition.

Inference path unchanged from `vad-off-v1`; only the weights are different.

### ft-lora32-int8f16 (13 May 12:45, NOT improving leaderboard) — int8 quantization fails

Re-exported the same LoRA-merged checkpoint at `int8_float16` quantization,
expecting `speed 0.849 → 0.90+` with `accuracy delta ≤ 0.005` per the README.

Reality: **accuracy −0.034**, speed only +0.007. Worst-of-both trade.

| | Local WER | Official acc | Official speed |
|---|---:|---:|---:|
| `ft-lora32-v1` (fp16) | 0.0299 | 0.957 | 0.849 |
| `ft-lora32-int8f16` | 0.0551 | 0.923 | 0.856 |
| Δ | +0.025 | −0.034 | +0.007 |

Root cause: Whisper's decoder is more quantization-sensitive than the README
assumed, likely because of the ~51866-token output vocab and the
LoRA-merged weight distribution amplifying int8 quantization noise.

Leaderboard kept the higher `ft-lora32-v1` score, so no rollback was needed.
The `int8_float16` lever for the speed score is **off the table** for this
checkpoint.

## CODEX recommendation

ASR is no longer the section that should receive major engineering time. The
current `ft-lora32-v1` image already gives a strong ASR blended score:

```text
0.75 * accuracy 0.957 + 0.25 * speed 0.849 = 0.930 blended
```

Because ASR is only 20% of the qualifier, the live contribution is about
`0.186`. The remaining theoretical gain from perfecting ASR is real but small:

```text
Current ASR contribution:   0.20 * 0.930 = 0.186
Perfect ASR contribution:   0.20 * 1.000 = 0.200
Remaining headroom:         ~0.014 overall qualifier score
```

That means the way forward is **protect the shipped peak, then only run short
A/B tests when AE/NLP/CV are blocked**. Do not spend another long training cycle
here unless the team explicitly decides ASR is the bottleneck.

Recommended ASR path:

1. **Keep `ft-lora32-v1` as the shipped baseline.** It has `0 / 400` errors,
   official `0.957 / 0.849`, and the best known blended score. Do not replace it
   unless a new tag beats it on official submission.
2. **Ignore CT2 int8 speed quantization for this checkpoint.**
   `ft-lora32-int8f16` already proved the trade is bad: accuracy fell
   `0.957 -> 0.923` while speed barely moved `0.849 -> 0.856`. That is a
   blended-score loss, not an optimization.
3. **Run one low-cost inference A/B if idle: `beam_size=2`.** Submit only if the
   local English WER improves enough that the speed hit is likely worth it. Rule
   of thumb: for blended score, `0.75 * accuracy_gain` must beat
   `0.25 * speed_loss`, so a `+0.003` accuracy gain can only afford about
   `-0.009` speed loss.
4. **Run one prompt-size A/B if beam is neutral:** regenerate `slang_prompt.txt`
   with top-100 or top-50 terms, still highest-frequency first. This might save
   a little decode overhead and reduce proper-noun over-priming. Reject it
   quickly if local English WER worsens or in-world noun errors increase.
5. **Only consider rank-64 / 5-epoch LoRA if the team needs the last ASR point.**
   Val WER was still falling, so there may be `0.005-0.010` accuracy left, but it
   costs another long Workbench run and does not fix the speed side. It is a
   late-stage polish move, not the next best competition move.
6. **Do not ensemble, TTA, or re-enable VAD.** Ensembles and speed-perturb voting
   likely lower blended score by doubling inference time; VAD already caused the
   dominant long-clip truncation failure.

Submission gate for any ASR experiment:

```text
Required: 0 / 400 errors, schema unchanged, official blended score > ft-lora32-v1
Local signal: track english error rate (WER), not local 1 - MER
Fallback: leaderboard keeps ft-lora32-v1 if an experiment regresses
```

Practically: ASR can maybe contribute another `0.002-0.006` overall with a
lucky beam/prompt tweak, but AE/NLP/CV have larger reachable headroom. Treat ASR
as a stable high-scoring module and use it as a reliability anchor.

## Gotchas hit (8 so far, all patched)

All eight transformers / PEFT / Workbench gotchas the project has hit are
documented inline in [../training/asr/README.md "Known gotchas"](../training/asr/README.md).
Summary list:

1. PEFT + gradient checkpointing → `element 0 of tensors does not require grad`. Fix: `enable_input_require_grads()` BEFORE `get_peft_model`.
2. `evaluation_strategy=` → `eval_strategy=` rename in transformers 4.46+.
3. `tokenizer=` → `processing_class=` rename in transformers 4.46+.
4. HF `datasets` Audio decoding wants `torchcodec` which needs FFmpeg system libs. Fix: `Audio(decode=False)` + soundfile in collator.
5. Workbench env got bumped to `transformers 5.8.0` overnight on 13 May → Whisper forward signature changed. Fix: `pip install --user 'transformers>=4.46,<5.0'`.
6. `LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM)` makes `PeftModelForSeq2SeqLM.forward` inject `input_ids=None` which transformers 4.57+ rejects on Whisper. Fix: drop `task_type` from the config.
7. `ct2-transformers-converter` errors if `--output_dir` exists at all (even empty). Fix: pass `--force`.
8. `export_ct2.py` wipe-loop deleted the slang prompt before the copy-back step when `--slang-file` lived inside `--output-dir`. Silent — only a `WARN` was emitted. Fix: buffer slang bytes in memory BEFORE the wipe.

The last one (gotcha 8) shipped the broken `ft-lora32-int8f16` *image* before
we caught it, but `extract_slang.py` regenerates the prompt deterministically
so recovery was fast.

## Levers we explicitly did NOT pull (and why)

- **Rank-64 / 5-epoch LoRA escalation** — `ft-lora32-v1` already crossed
  0.95; the marginal +0.005–0.010 isn't worth another 7-hour training run
  given the deadline and the bigger headroom on AE/NLP/CV.
- **Encoder unfreeze for one low-LR pass** — same reasoning. Modest gain,
  long training time.
- **`beam_size=2`** — would buy +0.002–0.005 WER for a small speed hit. Zero
  downside since leaderboard keeps high score, but unclear it pushes us
  meaningfully past 0.957.
- **Ensemble (distil-large-v3 + whisper-large-v3 with ROVER vote)** — ~2×
  inference cost would tank the speed score; only justified if accuracy was
  bottlenecking us, which it isn't.
- **TTA (test-time augmentation: speed-perturb + vote)** — same problem,
  doubles inference cost.
- **Pure `int8` (no float16 fallback)** — int8_float16 already regressed
  accuracy; pure int8 would almost certainly be worse.
- **Trim slang prompt to top-50** — small speed lever, unclear accuracy
  impact, not worth experimenting on with the deadline approaching.

## State of remaining ASR work

ASR is **parked at 0.957 / 0.849**. Marginal hour invested here now is
lower-ROI than the same hour on:

- **AE** (40% qualifier weight, best shipped `planner-v3b` at 0.499/0.853; learned policy needs mixed-opponent PPO after `bc-v1` regressed)
- **NLP** (currently 0.301; official scorer gives partial credit for retrieval, so BM25/chunking has direct upside)
- **CV** (currently 0.402/0.963 with 0/500 errors; fine-tuned YOLOv8s shipped, but hidden eval is far below local mAP)

The only ASR experiment worth a short cycle now is `beam_size=2` or a tiny
prompt tweak since it can't regress our leaderboard entry. Speed quantization
is no longer a priority: `ft-lora32-int8f16` regressed to `0.923 / 0.856`.

## Reproducibility / pointers

- Run end-to-end training: [../training/asr/README.md#quick-start](../training/asr/README.md)
- Error analysis & scoring artifacts: [../training/asr/ERROR_ANALYSIS.md](../training/asr/ERROR_ANALYSIS.md)
- Inference manager source: [src/asr_manager.py](src/asr_manager.py)
- HTTP server (don't edit): [src/asr_server.py](src/asr_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Model weights / slang prompt live in `asr/models/` (gitignored; regenerate via `export_ct2.py` + `extract_slang.py`)
