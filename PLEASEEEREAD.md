# PLEASEEEREAD — TIL-AI 2026

We need to submit **5 separate Docker services** by:

**24 May 2026, 11:59:59 PM SGT**

This repo already has **valid baselines** for all 5 tasks. They are dumb but useful: they let us build/test/submit something first, then improve task by task.

---

## What we need to do

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

So first goal: **working valid submissions for everything**. Then optimize.

---

## Current baseline files

These are the only files we should mainly edit for now:

```text
asr/src/asr_manager.py
cv/src/cv_manager.py
noise/src/noise_manager.py
nlp/src/nlp_manager.py
ae/src/ae_manager.py
```

I also patched:

```text
ae/src/ae_server.py
```

for more robust reset handling.

---

## What each baseline currently does

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

Need to improve with:

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

Need to improve with:

```text
YOLO / RT-DETR / other object detector
```

Output format must be:

```python
[
    {"bbox": [x, y, w, h], "category_id": category_id}
]
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

Need to improve with:

```text
small bounded perturbations that still pass fairness checks
```

Do this later, after scored tasks work.

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

Need to improve with:

```text
better chunking + embeddings + retrieval + extractive answer
```

Maybe later add a small local LLM, but only if it fits Docker/runtime.

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

Need to improve with:

```text
rule-based planner first, RL only if we have time
```

Actions:

```text
0 forward
1 backward
2 left
3 right
4 stay
5 place bomb
```

---

## Our likely plan

Priority:

1. **Get GCP working enough to run official `til test`**
2. **Submit all 5 baselines early** if allowed
3. Improve **AE** first because it is 40%
4. Improve **NLP** next because baseline can become decent quickly
5. Improve **CV**
6. Improve **ASR**
7. Improve **Noise** last

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

Avoid editing/deleting:

```text
til-26-ae/
til-26-finals/
.gitmodules
```

Those are official submodules.

Do not commit huge model weights to Git.

---

## Immediate team split

Suggested owners:

```text
AE: highest score, planner/agent logic
NLP: retrieval + answering
CV: detector
ASR: speech model
Noise + GCP integration: whoever has bandwidth
```
