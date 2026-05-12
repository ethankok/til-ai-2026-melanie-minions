# PLEASEEEREAD — TIL-AI 2026

We need to submit **5 separate Docker services** by:

**24 May 2026, 11:59:59 PM SGT**

The repo looks huge, but most of it is official scaffolding. For now, we mainly edit **5 manager files**.

---

## The only files that matter right now

```text
SUMMARY.md                   read this
RESULTS.md                   latest leaderboard/submission scores
asr/src/asr_manager.py       ASR baseline / our ASR code
cv/src/cv_manager.py         CV baseline / our CV code
noise/src/noise_manager.py   Noise baseline / our noising code
nlp/src/nlp_manager.py       NLP baseline / our RAG code
ae/src/ae_manager.py         AE baseline / our agent code
```

I also patched this once for reset robustness:

```text
ae/src/ae_server.py
```

Don't edit the rest unless we specifically need to.

---

## What the important folders/files mean

```text
asr/       speech recognition task
cv/        object detection task
noise/     adversarial image noising task
nlp/       RAG question-answering task
ae/        autonomous exploration / Bomberman agent task

test/      official local evaluators used by til test
README.md  official starter repo instructions
requirements-dev.txt  dev dependencies
.gitmodules  submodule config, don't touch

til-26-ae/      official AE environment, don't edit
til-26-finals/  official finals code, don't edit
```

Inside each task folder:

```text
src/*_manager.py  OUR model/inference logic — edit this
src/*_server.py   HTTP wrapper — usually don't edit
Dockerfile        packaging for submission — later
requirements.txt  task-specific Docker deps — later
README.md         task input/output spec — reference only
```

Mental model:

```text
server.py receives request -> manager.py returns prediction
```

Plain English:

- The evaluator sends data to our Docker container using HTTP.
- `*_server.py` handles the web/API part.
- `*_manager.py` is where our actual AI/model/rules go.
- The server calls the manager once per input and wraps the answer into JSON.
- If the JSON shape is wrong, we can get 0 even if the model is smart.

What a "baseline" means:

- A baseline is a simple working version.
- It is not meant to win.
- It proves the service starts, accepts input, and returns valid output.
- Once baselines work, we replace the dumb logic with better models one task at a time.

---

## What we need to submit

Submit these 5 tasks:

1. **ASR** — transcribe audio
2. **CV** — detect/classify objects in images
3. **Noise** — produce adversarially noised images
4. **NLP** — answer questions from provided documents using RAG
5. **AE** — control an agent in a Bomberman-like environment

Scoring weight:

```text
AE   40%
ASR  20%
CV   20%
NLP  20%
Noise: required/useful, but no direct qualifier points observed
```

Each scored task is roughly:

```text
75% accuracy/reward
25% speed
```

First goal: **working valid submissions for everything**. Then optimize.

Important terms:

- **Docker service**: a packaged mini-app for one task. It runs a web server and waits for inputs.
- **Endpoint/route**: the URL path the evaluator calls, e.g. `/asr` or `/nlp`.
- **Port**: the network number the service listens on, e.g. ASR uses `5001`.
- **Schema**: the exact input/output JSON format. Matching schema is non-negotiable.
- **GCP Workbench**: the official Google Cloud machine where data/submission tools live.
- **`til build`**: builds a Docker image for one task.
- **`til test`**: runs official local evaluation against that image.
- **`til submit`**: uploads/submits that image for scoring.

Task ports/routes:

```text
ASR    /asr    port 5001
CV     /cv     port 5002
Noise  /noise  port 5003
NLP    /nlp    port 5004
AE     /ae     port 5005, plus /reset
```

---

## Current baselines

### ASR

File: `asr/src/asr_manager.py`
Endpoint: POST `/asr` on port 5001
Input: list of `{key, b64}` where b64 is base64 WAV bytes
Output: `{"predictions": ["transcript1", "transcript2", ...]}` (same order as input)

What we built (Novice track is English-only, official score 12 May = 0.877 / 0.864 on `norm-v1`. **`vad-off-v1` is ready to submit** — local English WER dropped from 0.0759 to 0.0554 (~27% relative), predicted official ~0.91-0.92. `vad-off-v2` regressed and was reverted. Next major lever is LoRA fine-tune — see [training/asr/README.md](training/asr/README.md) Quick-start. See [RESULTS.md](RESULTS.md)):

```text
faster-whisper distil-large-v3 + slang prompt mined from NLP corpus.
Loaded once at startup, CT2 float16 on GPU (int8 CPU fallback).
greedy beam, condition_on_previous_text=False, language="en" forced.
vad_filter=False + audio-level silence guard (long-clip truncation was the
  dominant error in norm-v1 — see ERROR_ANALYSIS).
Hallucination guards: no_speech_threshold=0.6, log_prob_threshold=-1.0,
  compression_ratio_threshold=2.4, temperature=0.0, without_timestamps=True.
Post-processing: digits + ordinals + niner callsigns + coordinate-safe
  decimals verbalized so the scorer's word-level metric counts them correctly.
```

Server passes the whole HTTP batch to `ASRManager.asr_batch(list[bytes])` in
one call — see `asr/src/asr_server.py`.

Training (Workbench-only, in `training/asr/`):

1. `extract_slang.py` — mines NLP corpus → `slang_prompt.txt`, baked next to model weights. **Defaults**: `--top-k 200 --min-count 2`. Writes highest-frequency first; reversing was tested in `vad-off-v2` and regressed WER (over-primes decoder), so we keep the original order. Passed as `initial_prompt=` at inference.
2. `prepare_data.py` — reads `/home/jupyter/novice/asr/asr.jsonl` (4110 entries), 90/10 stratified split, oversamples slang-containing clips. **Note**: there is no separate training manifest — same file is what `test_asr.py` evaluates against, so post-FT local numbers will be inflated by memorization. Treat the official submission as the only real validator.
3. `train_distil_whisper.py` — LoRA fine-tune (decoder attn, encoder frozen), SpecAugment + optional noise mixing + speed perturb, jiwer WER aligned with official scorer transforms.
4. `export_ct2.py` — merge LoRA → `ct2-transformers-converter` → CT2 float16 dir + slang prompt copied alongside.
5. Push `asr/models/` to a private GCS bucket; pull before `til build asr`.

See [training/asr/README.md](training/asr/README.md) for end-to-end commands, the smoke-test path (zero-shot CT2 export, no training required), and environment variables (`ASR_DEVICE`, `ASR_COMPUTE_TYPE`) to override device/precision. See [training/asr/ERROR_ANALYSIS.md](training/asr/ERROR_ANALYSIS.md) for full data-driven error analysis, the scoring-artifact explanation, and the route toward `0.95+` score / `0.90+` speed.

Workbench facts (confirmed 12 May 2026):

- GPU: **Tesla T4** (16 GB) — fp16, not bf16.
- Manifest: 4110 entries at `/home/jupyter/novice/asr/asr.jsonl`, schema `{"key","audio","transcript","language":"english"}`. No separate `train.jsonl`.
- No accessible noise corpus on Workbench — augmentation falls back to SpecAugment + speed perturb only.
- Scoring artifact: local `1 - MER` is misleading high because the local manifest has only `english`-labeled samples and the scorer divides by 4 (the other three buckets contribute 0). Track the bare `english error rate (WER)` line in `test_asr.py` output instead.

Stretch (only after LoRA-FT + int8_float16 re-export land):
- Beam=2 (not 5) if speed has margin.
- Encoder unfreeze for a final low-LR pass.
- Ensemble distil-large-v3 + whisper-large-v3 (~2× inference cost) only if accuracy ceiling becomes the bottleneck.

---

### CV

File: `cv/src/cv_manager.py`
Endpoint: POST `/cv` on port 5002
Input: list of `{key, b64}` where b64 is base64 JPEG bytes
Output:

```python
{"predictions": [[{"bbox": [x, y, w, h], "category_id": int}, ...], ...]}
```

`bbox` is [x, y, w, h] (top-left + width/height, COCO style — NOT [x1, y1, x2, y2]).
Empty list per image is valid if nothing is detected.

Current baseline:

```text
returns [] = no detections
```

Valid but scores 0.

What we need to do:

1. Pick a detector. Default: `ultralytics` YOLOv8/v11 (easiest), or `RT-DETR` if accuracy > speed. Pretrained COCO weights first, then fine-tune on the provided training images if categories don't match COCO.
2. Confirm category mapping: the eval has its own `category_id` set. Map model class IDs → eval `category_id` (probably needs a lookup table from the dataset metadata).
3. Load model once in `CVManager.__init__`.
4. In `cv(image_bytes)`:
   - Decode JPEG (`PIL.Image.open(io.BytesIO(image_bytes))`).
   - Run detector.
   - Convert each detection to `{"bbox": [x, y, w, h], "category_id": id}`. Make sure to convert from xyxy→xywh if the detector returns xyxy.
   - Apply a confidence threshold (start ~0.25, tune).
5. Add deps to `cv/requirements.txt`: `ultralytics` (pulls torch), `pillow`. Use a CUDA base image in `cv/Dockerfile` if GPU is available on eval — check what the wiki says about GPU access.
6. Test: `til build cv && til test cv`.

Stretch:
- Fine-tune on the training set rather than relying on COCO weights.
- TTA (test-time augmentation) if speed budget allows.
- NMS tuning per class.

---

### Noise

File: `noise/src/noise_manager.py`
Output: base64-encoded JPEG string

Current baseline:

```text
re-encodes input as a clean JPEG, returns base64
```

Valid and safe.

What we need to do (do this last — no qualifier weight):

1. Read the noise spec carefully: there's a perturbation budget (likely L∞ or SSIM/PSNR threshold) AND a fairness/validity check.
2. Pick a method:
   - **Cheap**: random bounded noise within budget. Easy, modest impact.
   - **Better**: untargeted FGSM/PGD against a public surrogate classifier (ResNet/ViT), clipped to budget.
   - **Best**: ensemble attack across multiple surrogates for transferability.
3. In `noise(image_bytes)`:
   - Decode → tensor.
   - Compute perturbation, clip to budget.
   - Re-encode as JPEG (JPEG re-compression can wipe high-freq adversarial signal — be aware, may need to compensate).
   - Return base64.
4. Add deps to `noise/requirements.txt` only if needed (torch, torchvision).
5. Verify the fairness check still passes locally before submitting.

---

### NLP

File: `nlp/src/nlp_manager.py`
Has two methods: `load_corpus(documents)` (called once per round) and `qa(question)` (per question).

Current baseline:

```text
sentence-split each doc, lexical token-overlap retrieval with light length normalization,
return the top-scoring sentence (truncated to 500 chars)
```

This is our best starting point — it's already correct shape and somewhat useful.

What we need to do:

1. **Better chunking**: sliding window of ~2–3 sentences with overlap, not single sentences. Single sentences lose context; full paragraphs dilute retrieval.
2. **Better retrieval**: replace token-Counter overlap with BM25 (`rank_bm25` package) as a fast win. Then layer dense embeddings (e.g. `sentence-transformers/all-MiniLM-L6-v2`) for hybrid retrieval.
3. **Better answer extraction**: don't return the whole chunk. Either:
   - Run an extractive QA model (`distilbert-base-uncased-distilled-squad` or similar) on the top-k chunks, or
   - Pick the sentence within the chunk with highest overlap/embedding sim to the question.
4. Cache embeddings in `load_corpus` so `qa()` is fast.
5. Add deps to `nlp/requirements.txt`: `rank_bm25`, `sentence-transformers`, `transformers`.
6. Watch the answer length cap — current truncates at 500 chars; verify the eval doesn't penalize too-long answers.

Stretch:
- Re-ranker (cross-encoder) on top-k.
- Tiny local LLM for generative answers, **only** if it fits the runtime budget.

---

### AE

File: `ae/src/ae_manager.py`
Endpoint: POST `/ae` (and `/reset`) on port 5005
Server resets the manager when a fresh round is detected (`step == 0` or empty POST).

Actions:

```text
0 forward
1 backward
2 left  (rotate?)
3 right (rotate?)
4 stay
5 place bomb
```

Observation keys we already use: `action_mask`, `frozen_ticks`, `team_bombs`, `step`. There are more — inspect a real observation early.

Current baseline:

```text
respects action_mask, mostly forward, periodic rotates, bombs every 20 turns if legal,
stays put while frozen
```

AE is **40%** of the qualifier — biggest lever, do this first.

What we need to do:

1. **Inspect observations**: dump one real observation locally (run `til-26-ae`'s env, print keys/shapes). We need the full schema before planning. Likely has: agent position, orientation, occupancy grid, enemy positions, bomb timers, walls, items.
2. **Phase 1 — rule-based planner**:
   - Build a grid map from the observation each step.
   - Pathfind (BFS/A*) toward the nearest unexplored cell or objective.
   - Translate the next step in the path into the right action given current orientation.
   - Bomb only when (a) blocked by a destructible wall on the path, or (b) an enemy is in the blast line and we have an escape route. Always plan the escape before placing.
   - Avoid stepping into bomb blast zones — predict explosions N ticks ahead.
3. **Phase 2 — heuristic scoring**: score candidate actions by (progress toward goal) + (safety from bombs) + (item pickup) and pick argmax. Easier to tune than a hard rule tree.
4. **Phase 3 (only if time) — RL**:
   - Train PPO or DQN against `til_environment.bomberman_env` using `til-26-ae`.
   - Export policy weights, load in `AEManager.__init__`.
   - Inference must still be fast — tiny MLP/CNN, not a giant transformer.
5. Reset hygiene: anything stateful (turn counter, map memory) MUST live on `self` — the server re-instantiates `AEManager` on reset, which already handles this. Don't add module-level globals.
6. Test: run `python test/test_ae.py` after `til build ae` (it spins up the env and POSTs observations to the live container).

Stretch:
- Opponent modeling: track the enemy's recent moves and avoid their predicted next position.
- Bomb-chain planning (chain reactions).

---

## Rough outline (what to do, in order)

**Week 0 — plumbing (everyone, in parallel)**

1. Get GCP Workbench access working for everyone.
2. Run `til build <task>` and `til test <task>` for all 5 tasks against the current baselines. Goal: every container starts, every endpoint returns valid-shape JSON, every `til test` exits cleanly.
3. `til submit` all 5 baselines so we have a non-zero submission on the board.

**Week 1 — first real models**

4. **AE** (40%): replace random-walk baseline with a rule-based planner (BFS over the occupancy grid + bomb safety check). Big jump expected.
5. **NLP** (20%): swap lexical Counter for BM25 + better chunking. Single afternoon of work, meaningful score gain.
6. **CV** (20%): drop in pretrained YOLOv8/v11 with COCO weights, map class IDs to the eval's `category_id`. Real work is the class mapping, not the model.
7. **ASR** (20%): drop in faster-whisper (small), load once in `__init__`, batch instances. Done.

**Week 2 — push scores**

8. **AE**: heuristic scoring or fine-tuned policy. Test against `til-26-ae` env locally.
9. **NLP**: hybrid retrieval (BM25 + dense embeddings) + extractive QA model on top-k.
10. **CV**: fine-tune on the provided training set if categories don't match COCO.
11. **ASR**: try larger Whisper variant if speed budget allows.
12. **Noise**: bounded perturbation (FGSM against a surrogate classifier).

**Always**

- Re-run `til test` after every change.
- Don't commit model weights to git — bake them into the Docker image build instead.
- If a change makes a container fail to start, **revert immediately**. A working bad model > broken good model.

Why this order:

- AE is 40% — every hour spent there is worth ~2x the same hour on a 20% task.
- NLP's lexical baseline can jump significantly with retrieval upgrades alone.
- CV/ASR are mostly "swap in a pretrained model" once the plumbing works.
- Noise has no direct qualifier weight; do it last.

What "done for now" means for a task:

```text
1. service starts
2. endpoint responds
3. output JSON has correct shape
4. Docker builds on GCP
5. til test runs without crashing
6. til submit succeeds
```

Only after all 5 hit "done for now" should anyone chase better scores.

---

## Who edits what

```text
ASR owner:   asr/src/asr_manager.py
CV owner:    cv/src/cv_manager.py
Noise owner: noise/src/noise_manager.py
NLP owner:   nlp/src/nlp_manager.py
AE owner:    ae/src/ae_manager.py
```

If unsure, don't edit server/Docker/submodule files yet.

---

## Local setup

Check imports:

```bash
python - <<'PY'
import sys
sys.path.extend(['asr/src','cv/src','nlp/src','noise/src','ae/src'])
from asr_manager import ASRManager
from cv_manager import CVManager
from nlp_manager import NLPManager
from noise_manager import NoiseManager
from ae_manager import AEManager
print('ok')
PY
```

---

## GCP setup later

Official loop:

```bash
til build asr
til test asr
til submit asr
```

Repeat for:

```text
asr cv noise nlp ae
```

---

## Do not touch unless needed

```text
til-26-ae/
til-26-finals/
.gitmodules
src/*_server.py
Dockerfile
requirements.txt
```

Do not commit huge model weights to Git.
