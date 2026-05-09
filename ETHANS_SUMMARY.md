# PLEASEEEREAD — TIL-AI 2026

We need to submit **5 separate Docker services** by:

**24 May 2026, 11:59:59 PM SGT**

The repo looks huge, but most of it is official scaffolding. For now, we mainly edit **5 manager files**.

---

## The only files that matter right now

```text
PLEASEEEREAD.md              read this
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

File:

```text
asr/src/asr_manager.py
```

Current baseline:

```text
returns "" for every audio file
```

Valid but scores badly.

Improve with:

```text
Whisper / faster-whisper / multilingual ASR
```

---

### CV

File:

```text
cv/src/cv_manager.py
```

Current baseline:

```text
returns [] = no detections
```

Valid but scores 0.

Output format:

```python
[
    {"bbox": [x, y, w, h], "category_id": category_id}
]
```

Improve with:

```text
YOLO / RT-DETR / other object detector
```

---

### Noise

File:

```text
noise/src/noise_manager.py
```

Current baseline:

```text
returns the image basically unchanged
```

Valid and safe.

Improve later with:

```text
small bounded perturbations that still pass fairness checks
```

Do this after scored tasks work.

---

### NLP

File:

```text
nlp/src/nlp_manager.py
```

Current baseline:

```text
loads documents, keyword-matches the question, returns best matching sentence
```

This is our best starting point.

Improve with:

```text
better chunking + embeddings + retrieval + extractive answer
```

Maybe later use a small local LLM, only if it fits Docker/runtime.

---

### AE

File:

```text
ae/src/ae_manager.py
```

Current baseline:

```text
uses action_mask, moves forward when possible, turns when stuck, bombs occasionally
```

AE is worth the most, so improve this early.

Actions:

```text
0 forward
1 backward
2 left
3 right
4 stay
5 place bomb
```

Improve with:

```text
rule-based planner first, RL only if we have time
```

---

## Our priority plan

1. **Get GCP working enough to run official `til test`**
2. **Submit all 5 baselines early** if allowed
3. Improve **AE** first because it is 40%
4. Improve **NLP** next because baseline can become decent quickly
5. Improve **CV**
6. Improve **ASR**
7. Improve **Noise** last

Why this order:

- AE is worth the most, so small improvements matter a lot.
- NLP can improve quickly because retrieval baselines are already useful.
- CV/ASR may need larger pretrained models and more setup.
- Noise is useful but not the main qualifier score target.

What "done for now" means for a task:

```text
1. service starts
2. endpoint responds
3. output JSON has correct shape
4. Docker builds on GCP
5. til test runs without crashing
6. til submit succeeds
```

Only after that should we chase better scores.

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

On Ethan's Mac:

```bash
cd /Users/ethankok/Projects/til
source .venv/bin/activate
```

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

When Workbench works, clone this repo:

```bash
cd /home/jupyter
git clone https://github.com/kushmics/til-ai-2026.git til
cd til
git submodule update --init
micromamba activate env
pip install -r requirements-dev.txt
```

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
