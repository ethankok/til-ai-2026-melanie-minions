# ASR — notes & history

Last updated: 25 May 2026 — **Qualifier closed. `nemo-ft-v1` is the final high score at `0.969 / 0.946` (blended `0.96325`).**

Per-task working log for ASR. For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#asr).
For training-pipeline mechanics see [../training/asr/README.md](../training/asr/README.md).
For data-driven error analysis see [../training/asr/ERROR_ANALYSIS.md](../training/asr/ERROR_ANALYSIS.md).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`nemo-ft-v3` — official 0.970 / 0.947 (27 May 13:26 SGT, 0/400 errors).**
Blended score `0.75*0.970 + 0.25*0.947 = 0.96425`.
This model builds on `nemo-ft-v2` by adding post-processing rules for additional spelling and phonetic variants (e.g. zonen/sono -> Zonnon, mewn -> Mewan, pullwalker -> Fullwalker).

Decision: **closed.**

## NGPU-LM n-gram fusion prototype (02 Jun 2026 — Semis prep, default-OFF)

Design of record: [docs/superpowers/specs/2026-06-02-asr-ngram-lm-fusion-design.md](../docs/superpowers/specs/2026-06-02-asr-ngram-lm-fusion-design.md).

Adds GPU-resident n-gram (NGPU-LM) shallow fusion to the Parakeet-TDT decode
path to attack the residual in-world proper-noun WER. **Default-OFF**: with
`ASR_NGRAM_LM` unset the decoder stays greedy and the image is behaviourally
identical to `nemo-ft-v3`. Code shipped locally (unit-tested on Mac); the build,
sweep, and `til test` run on Workbench.

Local Mac validation done: `test/test_build_ngram_lm.py` +
`test/test_ngram_lm_decoding_cfg.py` (10 tests, the pure text-collection and
decoding-config logic).

### Environment reality (probed 02 Jun 2026)

- **Container** (pinned commit `ccbbfbb…`): **has** NGPU-LM `malsd_batch`
  (confirmed: the commit ships `tdt_malsd_batched_computer.py`). This is where
  serving runs, so fusion works in prod.
- **Workbench host `(base)`**: NeMo **2.0.0** — has only the *classic*
  n-gram/`maes` beam decoders, **no `malsd_batch`**. Too old to run the NGPU-LM
  sweep, but new enough to **build the n-gram LM** (it can load the TDT tokenizer
  + run KenLM). The base env is **conda**, so KenLM binaries are one
  `conda install` away.

Key fact that makes the cheap path work: the n-gram ARPA is keyed to the **BPE
tokenizer** (baked into the `.nemo`, identical across NeMo versions), so an ARPA
built on host NeMo 2.0.0 is valid for the container's `malsd_batch`, which
accepts a `.ARPA` directly. **Do NOT** `pip install -r requirements-nemo.txt`
into base (documented torch-downgrade cascade).

### Workbench runbook — cheap & blind path (build ARPA on host, tune via til test)

No local sweep (host can't run `malsd_batch`); ship defaults `alpha=0.3 beam=4`
and tune alpha across 1–2 `til test` runs.

```bash
cd /home/jupyter/til && git pull origin main
export TIL_FOLDER=/home/jupyter/til

# 1. KenLM binaries + the train_kenlm.py matching the HOST NeMo (2.0.0).
conda install -y -c conda-forge kenlm
curl -sSL -o /tmp/train_kenlm.py \
  https://raw.githubusercontent.com/NVIDIA/NeMo/v2.0.0/scripts/asr_language_modeling/ngram_lm/train_kenlm.py

# 2. Build the n-gram LM as a .ARPA (NO --save-nemo: 2.0.0 lacks NGPU-LM .nemo
#    packaging; the container loads the .ARPA directly). The script prints the
#    exact output file paths at the end.
python training/asr/build_ngram_lm.py \
    --asr-jsonl /home/jupyter/novice/asr/asr.jsonl \
    --nlp-dir /home/jupyter/novice/nlp \
    --nemo-model asr/models/parakeet-tdt-0.6b-v2.nemo \
    --out asr/models/ngram_lm \
    --ngram-length 6 \
    --train-kenlm /tmp/train_kenlm.py \
    --kenlm-bin "$(dirname "$(which lmplz)")"
# -> note the printed .ARPA path; e.g. asr/models/ngram_lm.tmp.arpa

# 3. Point the image at that ARPA. In asr/Dockerfile uncomment + set:
#      ENV ASR_NGRAM_LM=/workspace/models/asr/<the .arpa filename>
#      ENV ASR_NGRAM_LM_ALPHA=0.3   ENV ASR_BEAM_SIZE=4
#      ENV ASR_LM_STRATEGY=malsd_batch ENV ASR_LM_PRUNING=late
#      ENV ASR_LM_BLANK_MODE=lm_weighted_full

# 4. Gate the OFF image first (LM unset) — this is the "any errors?" check.
#    (Leave ASR_NGRAM_LM commented out for this build.)
til build asr ngram-lm-off
til test  asr ngram-lm-off        # must match nemo-ft-v3 behaviour exactly

# 5. Flip ON (uncomment the ENV block) and test/submit against the gate.
til build asr ngram-lm-on
til test  asr ngram-lm-on         # check WER ↑ and speed ≥ 0.92
til submit asr ngram-lm-on
# Tune: if alpha=0.3 helps but speed is fine, try 0.2 / 0.4 in 1–2 more til tests.
```

**Promotion gate:** `til test` speed ≥ 0.92 AND blended > `nemo-ft-v3` 0.964
(0/400 errors, schema unchanged). Otherwise keep `nemo-ft-v3`; leaderboard keeps
the higher score so a regression cannot demote us.

**Container support is confirmed** — the pinned commit ships
`tdt_malsd_batched_computer.py`, so `malsd_batch` NGPU-LM fusion works at serve
time without any pin bump. If `malsd_batch` ever misbehaves at runtime the
manager falls back to greedy (no crash), and `preserve_arpa` keeps a `.ARPA`
usable by the classic `maes` strategy (`ASR_LM_STRATEGY=maes`) as a manual
fallback. (Verified 02 Jun 2026 against commit `ccbbfbb…` on GitHub.)

## nemo-ft-v3 (27/05) — additional spelling post-processing fixes (new blended high)

Status: current shipped tag and blended high score.

Why this candidate:
- Identified residual spelling and phonetic errors in predictions via the `scan_errors.py` script.
- Added corrections for `zonen`/`sono` -> `Zonnon`, `mewn` -> `Mewan`, and space-less `pullwalker(s)` -> `Fullwalker(s)`.
- Corrected unit test assertions for casing behavior of `copy is sil`.
- Passed local unit tests and Workbench `til test` with English WER `0.0209` (improved from `0.0210`) and `1 - MER` `0.994785` (improved from `0.994753`).

Submit gate:
- Cloud result: 0/400 errors, Score `0.970`, Speed `0.947` (improved from `0.943`).
- Blended result: `0.96425` (new overall blended high score!).

## nemo-ft-v2 (27/05) — proper-noun post-processing fixes (prior accuracy high)

Status: prior accuracy high score.

Why this candidate:
- Integrated spelling and regex group post-processing fixes to `asr/src/asr_postprocess.py` based on error analysis of `asr_results.json`.
- Corrected regex capture groups for Sim Jiahong, Blackshore, and Clairos (e.g. `Clayro's` -> `Clairos'`).
- Added corrections for `Takeshi Oelaren` -> `Takeshi Oyelaran`, `Devika Runyan` -> `Devika Oranyan`, `Parks and Hyun` -> `Park Soo-Hyun`, `sadentu` -> `in Sarento`, `SEC37`/`CEC87` -> `TEC`, etc.
- Passed local unit tests and Workbench `til test` with English WER `0.0210` and `1 - MER` `0.9947535430276239`.

Submit gate:
- Cloud result: 0/400 errors, Score `0.970`, Speed `0.943`.
- Blended result: `0.96325`.

## nemo-ft-v1 (24/05) — fine-tuned model (prior high score)

Status: final submission.

Why this candidate:
- Fine-tuned Parakeet-TDT-0.6B-v2 on the novice dataset for ~1.5 epochs, reaching a val WER of 0.0856 at step 713.
- Extracted weights to a standalone `.nemo` format using a CPU fallback conversion script and baked it into the Docker context.
- Passed local `til test` with an English error rate (WER) of 0.0213.

Submit gate:
- Cloud result: 0/400 errors, Score `0.969`, Speed `0.946`.
- Blended result: `0.96325` (new overall blended high score!).

## nemo-zs-v7 (24/05) — residual replay candidate (accuracy high / blended tie)

Status: shipped and parked as the final ASR candidate.

Why this candidate:
- Keep the same Parakeet-TDT-v2 model/runtime as `nemo-zs-v6`; no decoder, prompt, or Docker risk.
- Replay the actual saved `nemo-zs-v6` local outputs from Workbench against `/home/jupyter/novice/asr/asr.jsonl`.
- Only keep residual rules that make sense after a second review and do not create replay regressions under the local alignment check.

Replay gate:
- `nemo-zs-v6` saved-output replay: 8532 edit errors, approximate WER `0.02912`.
- `nemo-zs-v7` post-process replay: 7549 edit errors, approximate WER `0.02576`.
- Delta: 983 fewer edit errors, 497 changed predictions, 494 improved lines, 0 worsened lines.

Changes staged in `asr/src/asr_postprocess.py`:
- Join high-confidence compounds: `launchpad`, `launchpads`, `waystation`, `supersoldiers`, `megacorp(s)`, `blackrock`, `stellarcore`.
- Repair Opted/Opting/Optin hyphenation and a few residual ASR forms (`petrol` -> `patrol`, `Marcos` -> `Marcus`, `Fair Ex` -> `Phyrexis`, `Neari` -> `Nyari`, `Tai Dak`/`Dida` -> `Tidak`).
- Add context-limited possessives for `Caulfield's`, `Cyanite's`, `Sim's`, and `Dreamer's` while preserving known plural exceptions.
- Normalize maritime bearing phrases such as `bearing ninety five degrees` -> `bearing zero nine five degrees` and `heading one hundred eighty degrees` -> `heading one eight zero degrees`.

Submit gate:
- Workbench `til build asr nemo-zs-v7`: passed.
- Workbench `til test asr nemo-zs-v7`: passed in about 15:18 with English WER `0.0270`, `1 - MER 0.9932495352686306`.
- Workbench `til submit asr nemo-zs-v7`: completed cleanly and handed off to automatic evaluation.
- Cloud result: 0/400 errors, Score `0.969`, Speed `0.941`.
- Blended result: `0.9620`, tied with `nemo-zs-v6`.

Decision:
- Keep `nemo-zs-v7` as the latest/raw-accuracy-high tag.
- Keep `nemo-zs-v6` as the faster blended-tie fallback.
- Park ASR; no further model/runtime/post-processing work is recommended before deadline.

## nemo-zs-v6 (24/05) — residual cleanup high

Submitted 24 May 2026 17:39 SGT.

Results:
- Errors: 0/400
- Score (Accuracy): 0.967
- Speed: 0.947
- Blended score (75/25): 0.9620 (first blended high; now tied by `nemo-zs-v7`)
- Workbench local test before submit: English WER `0.0296`, `1 - MER 0.9926082235489664`

Why this candidate:
- Kept `nemo-zs-v5`'s fast Parakeet-TDT-v2 backend and only added post-processing repairs discovered from saved-output replay.
- Fixed `%` outputs to preserve `percent` after scorer punctuation removal.
- Removed standalone filler hallucinations (`uh`, `um`, `mm`).
- Fixed hundreds/thousands ordinals such as `123rd` -> `one hundred twenty third`.
- Added cleanup for The CUBE spacing, First Dreamer spacing, and a small set of residual proper-noun/style variants.
- Replay against saved local outputs reduced approximate WER from `0.03187` to `0.02960` before the full Workbench run.

## nemo-zs-v2 (22/05) — phonetic post-corrections accuracy peak

Submitted 22 May 2026 21:05 SGT.

Why this candidate:
- Added casing-preserving phonetic post-corrections for proper nouns (e.g., Sorrento -> Sarento, Phyrexis, Mewan, etc.) in `asr/src/asr_postprocess.py`.
- Intended to run on top of the fast `parakeet-tdt-0.6b-v2.nemo` model.

Results:
- Errors: 0/400
- Score: 0.962 (Accuracy improved from 0.956 to 0.962, verifying the post-corrections work!)
- Speed: 0.911 (dropped from 0.946)
- Local English WER: 0.0384 (vs baseline 0.0429)
- Local 1 - MER: 0.9904 (vs baseline 0.9893)

Diagnosis:
- The default model in `asr/Dockerfile` was set to `parakeet-unified-en-0.6b.nemo` instead of `parakeet-tdt-0.6b-v2.nemo`. The unified model has higher inherent latency (speed 0.911-0.915) due to its non-TDT architecture.
- Reverting the default model in both `Dockerfile` and `Dockerfile.nemo` to `parakeet-tdt-0.6b-v2.nemo` will recover the baseline speed to `0.946`.

Action:
- Switched default model variables to `parakeet-tdt-0.6b-v2.nemo` in `asr/Dockerfile`, `asr/Dockerfile.nemo`, `asr_manager_nemo.py`, and `download_models_nemo.py`.
- Simplified the NeMo sanity check in Dockerfiles (removed `att_chunk_context_size` check).
- Built and submitted the next iteration as `nemo-zs-v3`.

## nemo-zs-v3 (22/05) — blended score high SGT

Submitted 22 May 2026 21:46 SGT.

Why this candidate:
- Reverted the default model to `parakeet-tdt-0.6b-v2.nemo` to recover from `nemo-zs-v2`'s latency.
- Retained all the casing-preserving phonetic post-corrections from `nemo-zs-v2`.

Results:
- Errors: 0/400
- Score (Accuracy): 0.960 (slightly below unified model's 0.962, but significantly above original zero-shot 0.956)
- Speed: 0.945 (recovers the speed score from 0.911 back to baseline levels)
- Local English WER: 0.0378
- Local 1 - MER: 0.9905
- Blended score (75/25): 0.95625 (new overall blended high score!)

## nemo-zs-v4 (23/05) — slang prompter & post-processing refinement

Results:
- Errors: 0/400
- Score (Accuracy): 0.962 (tied our raw accuracy peak, verifying the slang and Zonnon/Caulfield corrections work!)
- Speed: 0.942
- Blended score (75/25): 0.957 (new overall blended high score!)

Why this candidate:
- Fixed slang prompter extraction pollution in `training/asr/extract_slang.py` by embedding a `FALLBACK_WORDS` list (1500 common English words) to prevent silent fallback failures from filling the bias prompt with common words when download dependencies fail.
- Refined the proper noun rules in `asr/src/asr_postprocess.py` (specifically `Ashcastle` rules) by splitting them into a prefix-required rule and a standalone rule. This successfully resolves the space-eating bug where preceding spaces were collapsed when the prefix was absent (e.g., converting `is Ashcastle` to `isAshcastle`).
- Added case-preserving phonetic/spelling post-corrections for `Zonnon` (e.g., Zonan, zonon, Zonanun, Zonal, zondun's -> Zonnon) and `Caulfield` (e.g., Coalfields, callfields, Coffield's -> Caulfield) to capture common mistakes.
- Local validation on `asr_results.json` shows 335/4110 lines modified, successfully fixing boundary collapsing, word merging, and phonetically close proper nouns without affecting correct substitutions.

## nemo-zs-v5 (23/05) — v5 post-processing & refined Phi rules

Results:
- Errors: 0/400
- Score (Accuracy): 0.966 (Accuracy improved from 0.962 to 0.966, a new high!)
- Speed: 0.944
- Blended score (75/25): 0.9605 (new overall blended high score!)

Why this candidate:
- Added refined proper noun rules in `asr/src/asr_postprocess.py` to capture remaining phonetic and spacing mismatches against gold transcript patterns:
  - Canian (e.g. kanyan, canaanian, Canadian -> Canian)
  - Hegemony (e.g. hegemoni, Hegmoni -> Hegemony)
  - Sharpsea Bloc / routes (e.g. sharp sea, SHARP C BLOCK -> Sharpsea Bloc)
  - Nyari (e.g. niari, niyari -> Nyari)
  - Dreamer (e.g. streamer -> Dreamer)
  - Fullwalker (e.g. full walker, pull walkers -> Fullwalker)
  - Edgedancer (updated to match both `c` and `s` spelling variations `d[ae]n[cs]ers?`)
  - Floodwall (e.g. flood wall -> Floodwall)
  - TEC (e.g. tech command -> TEC command, for tech -> for TEC)
  - CYPHER (e.g. cipher requires -> Cypher requires, give cipher -> give Cypher)
  - Bloc (e.g. block tensions -> bloc tensions, Accommodationist block -> Accommodationist bloc)
- Refined Phi currency / metric context-specific replacements:
  - Standalone pi/fi/fai -> Phi
  - Preceded by scale words (million/thousand/hundred/billion five/fai) checked against currency contexts (bribes, credits, somatic clinic, somatic enhancement, biodealers, cost/price/prices, funds transfer, etc.) and explicitly excluded in telemetry contexts (bearing/vector/heading degrees, wind knots, coordinates, latitude/longitude).
  - Followed by currency suffixes (five ledger, file credits, fi movements).
  - Matches context phrases (got five to drop, sold file for, bleeding/funneling/saving up five, throwing five around, five in bribes/at blackjack/for/minimum).
- Validated locally on `asr_results.json` showing 195 repaired mismatches and 0 regressions against gold data.
- Corrected unit test assertions in `test/test_asr_postprocess.py` to enforce strict casing preservation behavior (e.g., lowercased input to lowercased output like "tech command" -> "tec command", and capitalized input to capitalized/uppercase output like "Tech command" -> "TEC command").



## Completed A/B: Parakeet unified zero-shot

**`parakeet-unified-zs` — rejected 22 May 2026.** This was a zero-shot
replacement for the current Parakeet-TDT-v2 checkpoint, not a fine-tune.

Why this candidate:

- Same practical size class as the current high: 600M parameters.
- Same NeMo-style offline `.nemo` packaging, so it is much lower risk than
switching to Cohere/Granite/Qwen runtimes.
- Model-card OpenASR offline WER is slightly better than TDT-v2
  (`5.91` vs `6.04/6.05`), while keeping a transducer-style architecture.
- It supports punctuation/capitalization, but the official scorer strips
  punctuation, so this should not hurt.

Code/build changes:

```text
asr/Dockerfile                      now uses NeMo + ASR_NEMO_MODEL=parakeet-unified-en-0.6b.nemo
asr/Dockerfile.nemo                 mirror explicit NeMo build path
asr/requirements-nemo.txt           NeMo pinned to GitHub main commit ccbbfbb for unified-model support
asr/src/asr_manager_nemo.py         default local model file -> parakeet-unified-en-0.6b.nemo
training/asr/download_models_nemo.py default model -> nvidia/parakeet-unified-en-0.6b
```

22 May startup diagnosis:

- The direct Hugging Face `.nemo` download works; `ASRModel.from_pretrained`
  was removed from the host download path so Workbench no longer needs to
  instantiate the model just to stage weights.
- The first built image still installed released `nemo_toolkit[asr]==2.7.3`.
  That release's `ConformerEncoder` does not accept `att_chunk_context_size`,
  but `nvidia/parakeet-unified-en-0.6b` includes that key in its config.
- Result: the container never reached healthy state, cloud saw 400/400 error
  exits, and the score was `0.000 / 0.996`. This is a startup/runtime mismatch,
  not an ASR quality measurement.
- The Docker build now checks for `att_chunk_context_size` immediately after
  installing NeMo, before copying the multi-GB model directory. If this check
  fails, do not submit; the build is intentionally stopping a bad image early.

Final result after runtime fix:

```text
Workbench til test:
  1028/1028 complete in 54:12
  English WER: 0.0453
  1 - MER:     0.9886804088933244

Cloud:
  tag:    parakeet-unified-zs
  errors: 0/400
  score:  0.956
  speed:  0.915
```

Decision: **reject**. Unified tied the current high's accuracy but lost speed
(`0.915` vs `0.946`) and local English WER was worse than the `nemo-zs` gate
(`0.0453` vs `0.0429`). Blended score is `0.75*0.956 + 0.25*0.915 = 0.9458`,
below `nemo-zs` at `0.9535`.

Workbench commands:

```bash
cd /home/jupyter/til
git pull origin main
export TIL_FOLDER=/home/jupyter/til

# Host side only needs this to fetch the .nemo. Keep <1.0 so transformers in
# the Workbench env does not become dependency-conflicted.
python -m pip install -U "huggingface_hub>=0.34,<1.0"

python training/asr/extract_slang.py \
    --nlp-dir /home/jupyter/novice/nlp \
    --out asr/models/slang_prompt.txt

python training/asr/download_models_nemo.py \
    --model nvidia/parakeet-unified-en-0.6b \
    --out asr/models

til build asr parakeet-unified-zs
til test asr parakeet-unified-zs

# Reproduce only. The final cloud probe tied accuracy but lost speed, so this
# tag is not the ASR high.
til submit asr parakeet-unified-zs
```

Decision gate:

```text
Promote if: no longer applicable; final result did not clear the gate.
Rejected:  local English WER 0.0453, cloud 0.956 / 0.915, blended 0.9458.
Fallback:  `nemo-zs` remains the shipped high. To rebuild the old TDT-v2 path,
           set ASR_NEMO_MODEL=parakeet-tdt-0.6b-v2.nemo and download that file.
```

## Historical experiment: NeMo Parakeet-TDT backend

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
asr/src/asr_manager_nemo.py         NemoASRManager (generic local NeMo `.nemo`)
asr/src/asr_server.py               picks backend via ASR_BACKEND env (default whisper)
asr/requirements-nemo.txt           NeMo runtime + audio libs
asr/Dockerfile.nemo                 explicit NeMo image, ENV ASR_BACKEND=nemo
training/asr/download_models_nemo.py stage local `.nemo` files into asr/models/
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

## Creative options for pushing past 0.99

Decoder-only FT + NeMo word-boosting probably caps at ~0.97-0.98. To
break 0.99 (official WER ≤ 0.009) the levers below attack what's left,
which per [../training/asr/ERROR_ANALYSIS.md](../training/asr/ERROR_ANALYSIS.md)
is almost entirely in-world proper-noun substitutions (Sarento→Sorrento,
Cyanite→cyanide, Phyrexis→Pyrex's, Mewan→Mee-one, Kestrelian→Castilian,
Belford Straits→Belford Streets). Leaderboard reference: Overflow
0.991/0.925, OpenLarp 0.986/0.940, suite108 0.982/0.920 — so 0.99 is
empirically reachable.

Roughly ordered by expected impact × creativity. Not yet tried.

### A. Phonetic + world-frequency post-correction (cheap, no retrain)

Pure inference-side. For every token in the hypothesis:

- If OOV for an English word list AND within Double-Metaphone edit
  distance ≤ 1 of an NLP-corpus term → substitute.
- If the token IS an English word but a phonetically-equivalent
  NLP-corpus term has ≫ in-world frequency (`cyanide` vs `cyanite`) →
  substitute, gated by Parakeet token confidence (NeMo exposes
  `hypothesis.score`).

Add as a new stage in [src/asr_postprocess.py](src/asr_postprocess.py).
Runs in milliseconds, no GPU. Directly targets the
[ERROR_ANALYSIS.md "Proper-noun substitutions"](../training/asr/ERROR_ANALYSIS.md)
failure bucket. Expected +0.005-0.015 absolute.

### B. CTC + TDT dual-head ensemble (free inside Parakeet)

Parakeet-TDT exposes both a TDT decoder and a CTC head trained jointly.
They make different proper-noun errors. ROVER-vote at word level inside
one model invocation — no extra encoder forward pass. NeMo has a
`hybrid_rnnt_ctc_bpe` path that does this natively. Expected +0.002-0.005
official for ~0 speed cost.

### C. Retrieval-aware ASR (most creative, untried, blue-sky upside)

Slang prompt was mined from the NLP corpus, so audio transcripts likely
share phrasing with corpus sentences. Pipeline:

1. Parakeet emits hypothesis.
2. Embed hypothesis with BGE-small (already in the NLP container).
3. Retrieve top-1 corpus sentence by cosine.
4. If cosine > 0.92 AND word-level WER (hyp vs retrieved) ≤ 0.2, replace
   hypothesis with the retrieved sentence.

If transcripts are paraphrased from corpus content, one corpus hit fixes
every slang error in that sentence at once. 30-minute A/B on local data
to check whether retrieval similarities cluster high before investing
further. Risk: paraphrase too loose → false replacements; gate hard.

### D. KenLM in-domain shallow fusion (canonical, not yet wired)

Train a 4-gram KenLM on the NLP corpus, plug into NeMo's beam decoder.
TDT supports beam + LM rescoring. Every published ASR benchmark gets
+0.5-2 WER points absolute from this; on a vocabulary-driven failure
profile like ours it could be +1-2. Tradeoff: beam decoding is slower
than greedy. Cloud speed is currently 0.946 — there's some headroom but
not infinite. Pair with option H (conditional beam) to preserve speed.

### E. TTS-augmented training for slang coverage

The 4110 clips have median 1-3 hits per rare in-world noun. Synthesize
50-100 additional clips per top-200 slang term using a fast TTS (Piper,
Coqui, XTTS) reading templates like
`"Approach the Sarento checkpoint at zero six hundred."` Mix into FT at
~10% of batches. Targets the failure distribution directly. Risk:
TTS-bias leakage — small because Parakeet wasn't pretrained on this
synth data.

### F. Active-learning targeted FT

Run zero-shot Parakeet over all 4110 clips, identify the ~200 worst-WER
ones, oversample 8× alongside slang oversampling. Direct attack on the
long-tail failure distribution. Adds nothing to wall clock; just a
manifest change in
[../training/asr/prepare_data_nemo.py](../training/asr/prepare_data_nemo.py).

### G. Conditional Whisper fallback (uses both backbones)

Run Parakeet first. For each clip, check TDT confidence
(`hypothesis.score`). For the bottom ~5% by confidence, fall back to
`ft-lora32-v1` Whisper and ROVER-vote at word level. ~5% extra wall
clock (Whisper only fires sometimes). The two models miss different
things — Whisper's vocab attention captures some slang Parakeet munges,
and vice versa. Fallback image already exists; just need to load both
managers.

### H. Conditional N-best beam rescoring (speed-preserving)

Greedy TDT for high-confidence clips (~90% of test), beam-4 + KenLM
rescoring only for low-confidence ones. Cherry-picks the accuracy gain
of option D while keeping the speed score intact.

### I. Vocab extension on the SentencePiece tokenizer

Parakeet splits `Sarento` into pieces it never saw co-occur. Add ~200
in-world proper nouns as new SentencePiece tokens, randomly initialize
their embeddings, fine-tune the embedding matrix + joint head for 1
epoch. Now `Sarento` is one emission, not three risky pieces. NeMo
supports tokenizer extension via `change_vocabulary`. Higher
implementation cost; addresses the root cause.

### J. External pretraining-data fine-tune (orthogonal)

LibriSpeech / CommonVoice / GigaSpeech mixed in at ~10% weight during
FT for one epoch reduces generic residual ~1% WER (not slang, but
real-world variety). Won't break 0.99 alone but stacks.

### Recommended stack if we commit to chasing 0.99

These compose. Rough 9-day budget to deadline:

1. **Day 1**: A (phonetic post-correction) + B (CTC+TDT hybrid head) —
   both pure inference-time, A/B against `nemo-zs` without retraining.
   If either lands +0.01, ship immediately.
2. **Day 2-3**: C (retrieval-aware ASR) — fastest creative experiment
   with the biggest blue-sky upside. Validate retrieval similarity
   distribution on local first.
3. **Day 3-5**: queued Parakeet FT + F (active-learning oversampling) +
   E (TTS-augmented data). Single training run, three levers stacked.
4. **Day 6-7**: D (KenLM shallow fusion) for the final accuracy push.
   Validate speed stays ≥ 0.92 — pair with H if it doesn't.
5. **Day 8-9**: only if still below 0.99 — G (conditional Whisper
   fallback) or I (vocab extension).

The two ideas that exploit competition-specific assets and most likely
distinguish 0.99-tier teams from 0.97-tier teams:

- **C (retrieval-aware ASR)** and **D (KenLM on the NLP corpus)** both
  leverage the fact that we have the full in-world text corpus.
  Generic ASR teams don't have this lever.

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

ASR is **parked at the `nemo-zs-v7` / `nemo-zs-v6` blended tie**:

- `nemo-zs-v7`: `0.969 / 0.941`, blended `0.9620`, raw accuracy high.
- `nemo-zs-v6`: `0.967 / 0.947`, blended `0.9620`, faster fallback.

Marginal ASR time is no longer justified before the deadline. Do not run beam,
prompt, model, or runtime experiments unless organisers change scoring or a hard
failure appears. Put time into unresolved task scores or final packaging.

## Reproducibility / pointers

- Run end-to-end training: [../training/asr/README.md#quick-start](../training/asr/README.md)
- Error analysis & scoring artifacts: [../training/asr/ERROR_ANALYSIS.md](../training/asr/ERROR_ANALYSIS.md)
- Inference manager source: [src/asr_manager.py](src/asr_manager.py)
- HTTP server (don't edit): [src/asr_server.py](src/asr_server.py)
- Container build: [Dockerfile](Dockerfile), [requirements.txt](requirements.txt)
- Model weights / slang prompt live in `asr/models/` (gitignored; regenerate via `export_ct2.py` + `extract_slang.py`)
