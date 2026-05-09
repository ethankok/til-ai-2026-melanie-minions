# PLEASEEEREAD — BrainHack 2026 TIL-AI Team Guide

This repo is our working codebase for **BrainHack 2026 TIL-AI**.

Short version: we need to build **5 Dockerized AI services** and submit them before the Virtual Qualifiers deadline.

**Deadline:** 24 May 2026, 11:59:59 PM SGT  
**Official repo:** https://github.com/til-ai/til-26  
**Wiki/specs:** https://github.com/til-ai/til-26/wiki  
**Curriculum:** https://github.com/til-ai/til-26-curriculum

---

## 1. What is this hackathon?

TIL-AI is a defence/security-themed AI competition where teams build models/services for multiple tasks:

1. Speech recognition
2. Computer vision object detection
3. Adversarial image noising
4. RAG question answering
5. Autonomous exploration in a Bomberman-like gridworld

The competition infrastructure expects each task as a **separate HTTP service inside a Docker image**. The official `til` CLI on the GCP Workbench builds, tests, and submits each image.

The goal is not just to train cool models. The goal is to submit services that:

- start reliably
- expose the correct endpoint/port
- return the exact required JSON schema
- run fast enough
- score well on hidden evaluation data

---

## 2. Competition modes / tracks

There are at least two tracks visible in the starter/evaluation setup:

- **Novice** — easier data/tasks; current GCP folder seen as `/home/jupyter/novice`.
- **Advanced** — harder data/tasks; includes more challenging multilingual/noisy/cross-document cases.

The local tests and GCP data paths use `TEAM_TRACK`, usually one of:

```bash
novice
advanced
```

The official Workbench may already have `/home/jupyter/novice` or `/home/jupyter/advanced` data folders.

---

## 3. What we have to submit

We submit **5 separate challenge images/services**:

### ASR — Automatic Speech Recognition

- Folder: `asr/`
- Manager file: `asr/src/asr_manager.py`
- Route: `/asr`
- Port: `5001`
- Input: base64 WAV audio
- Output:

```json
{"predictions": ["transcript one", "transcript two"]}
```

Scoring:

- English/Malay/Tamil: WER, lower is better
- Chinese: CER, lower is better
- approximate score: `max(0, 1 - mean_error_rate)`

Likely plan:

- Start with valid blank baseline.
- Then try Whisper / faster-whisper / wav2vec-style ASR.
- Need multilingual support if our track includes Chinese/Malay/Tamil.

---

### CV — Object Detection

- Folder: `cv/`
- Manager file: `cv/src/cv_manager.py`
- Route: `/cv`
- Port: `5002`
- Input: base64 JPEG image
- Output per image:

```json
[
  {"bbox": [x, y, w, h], "category_id": 0}
]
```

Scoring:

- COCO-style mAP from IoU 0.50 to 0.95
- predicted boxes must be accurate and category IDs must match

Known target categories from the wiki:

```text
0 cargo aircraft
1 commercial aircraft
2 drone
3 fighter jet
4 fighter plane
5 helicopter
6 light aircraft
7 missile
8 truck
9 car
10 tank
11 bus
12 van
13 cargo ship
14 yacht
15 cruise ship
16 warship
17 sailboat
```

Likely plan:

- Start with valid empty-detection baseline.
- Fine-tune or adapt YOLO/RT-DETR/DETR-style detector if data is available.
- Prioritize getting any boxes/classes working before chasing mAP.

---

### Noise — Adversarial Image Noising

- Folder: `noise/`
- Manager file: `noise/src/noise_manager.py`
- Route: `/noise`
- Port: `5003`
- Input: base64 JPEG image
- Output:

```json
{"predictions": ["BASE64_ENCODED_NOISED_IMAGE"]}
```

Important:

- Noise is required/useful, but appears to have **no direct qualifier points**.
- It matters for finals and for making competitors' CV models worse.
- If the noised image violates fairness thresholds, it may be rejected or replaced by the original.

Observed local fairness thresholds:

```text
L2/RMSE <= 67
L2 inside boxes <= 50
SSIM inside boxes >= 0.3
```

Likely plan:

- Current baseline passes image through unchanged.
- Later: add bounded noise/blur/compression/patch perturbations while staying under fairness limits.
- Do not spend too much time here before ASR/CV/NLP/AE have valid submissions.

---

### NLP — RAG Question Answering

- Folder: `nlp/`
- Manager file: `nlp/src/nlp_manager.py`
- Route: `/nlp`
- Port: `5004`

First request loads corpus:

```json
{"instances": [{"documents": ["doc text 1", "doc text 2"]}]}
```

We must return:

```json
{"predictions": ["loaded"]}
```

Then questions arrive:

```json
{"instances": [{"question": "..."}]}
```

We return:

```json
{"predictions": ["answer"]}
```

Scoring:

- Uses an answer-equivalence model, not exact string match.
- Empty string may be correct for unanswerable questions in harder settings.

Likely plan:

- Current baseline does simple keyword retrieval and returns the best matching sentence.
- Improve to chunk documents, embed chunks, retrieve top-k, then answer with a small local LLM or extractive QA.
- Must work offline in Docker. Assume evaluation has no internet.

---

### AE — Autonomous Exploration

- Folder: `ae/`
- Manager file: `ae/src/ae_manager.py`
- Route: `/ae`
- Port: `5005`
- Reset route: `/reset`
- Environment: Bomberman-like gridworld

Output:

```json
{"predictions": [{"action": 0}]}
```

Actions:

```text
0 FORWARD
1 BACKWARD
2 LEFT
3 RIGHT
4 STAY
5 PLACE_BOMB
```

Observation includes:

```text
action_mask
agent_viewcone
base_viewcone
direction
location
base_location
health
frozen_ticks
base_health
team_resources
team_bombs
step
```

Scoring:

- AE is the biggest component: **40% of total qualifier score**.
- Reward comes from exploration/objectives/combat/base interactions depending on the environment.

Likely plan:

- Current baseline respects `action_mask`, moves forward when possible, rotates when stuck, bombs occasionally.
- Next improvement: write a proper heuristic planner using viewcone/action mask.
- Later: train RL or imitation policy only if time/data/setup permits.

---

## 4. Overall scoring weights

Qualifier score weights:

```text
ASR: 20%
CV: 20%
NLP: 20%
AE: 40%
Noise: required/useful, but no direct qualifier points observed
```

For scored tasks:

```text
75% accuracy/reward
25% speed
```

Qualifier speed cap observed:

```text
t_max = 30 minutes
absolute max runtime = 1 hour
```

Meaning: a slow but accurate model can lose points. A working fast baseline is better than a fancy model that times out.

---

## 5. How to run locally

Local path on Ethan's Mac:

```bash
cd /Users/ethankok/Projects/til
source .venv/bin/activate
```

If `.venv` is missing:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

Quick sanity check:

```bash
python - <<'PY'
from asr.src.asr_manager import ASRManager
from cv.src.cv_manager import CVManager
from nlp.src.nlp_manager import NLPManager
from noise.src.noise_manager import NoiseManager
from ae.src.ae_manager import AEManager
print('manager imports ok')
PY
```

Local limitation: the official `til build`, `til test`, and `til submit` commands are expected on GCP Workbench, not necessarily on local Mac.

---

## 6. How to run on GCP Workbench later

GCP is slow/unreliable right now, so we are developing locally first. But final validation/submission must happen on Workbench.

Expected Workbench commands:

```bash
cd /home/jupyter
git clone https://github.com/kushmics/til-ai-2026.git til
cd til
git submodule update --init
micromamba activate env
pip install -r requirements-dev.txt
```

Then for each task:

```bash
til build asr
til test asr
til submit asr
```

Repeat for:

```bash
asr
cv
noise
nlp
ae
```

Use tags when trying multiple versions:

```bash
til build nlp lexical-v1
til test nlp lexical-v1
til submit nlp lexical-v1
```

---

## 7. Current state of this repo

This repo currently has **valid baseline manager files**:

```text
asr/src/asr_manager.py      blank transcript baseline
cv/src/cv_manager.py        empty detections baseline
noise/src/noise_manager.py  pass-through JPEG baseline
nlp/src/nlp_manager.py      keyword/sentence retrieval baseline
ae/src/ae_manager.py        simple action-mask heuristic baseline
ae/src/ae_server.py         reset handling patched for robustness
```

These baselines are not meant to win. They are meant to ensure every service has valid output and can be improved independently.

---

## 8. Our likely strategy

Priority order:

1. **Get all five services building and schema-valid.**
2. **Submit early once GCP works**, even with dumb baselines, if multiple submissions/best-score-counts policy applies.
3. **Improve AE**, because it is 40% of the score.
4. **Improve NLP**, because a simple retrieval baseline can become decent quickly.
5. **Improve CV**, likely with a pretrained detector/fine-tuning.
6. **Improve ASR**, likely with Whisper/faster-whisper or another multilingual ASR model.
7. **Noise last**, unless finals/strategy demands it earlier.

Concrete first-week plan:

- Day 1: make all services start and return valid outputs.
- Day 2: get first official `til test` results on GCP.
- Day 3: NLP retrieval upgrade.
- Day 4: AE heuristic upgrade.
- Day 5: CV baseline detector.
- Day 6: ASR model integration.
- Day 7+: iterate based on leaderboard scores and runtime.

---

## 9. Team rules / warnings

- Do not commit huge model weights/checkpoints to Git.
- Do not delete or edit these submodule folders unless we know exactly why:

```text
til-26-ae/
til-26-finals/
.gitmodules
```

- Keep commits frequent. The rules warn against suspicious single-commit large submissions.
- Use AI tools responsibly and credit AI assistance if required by submission rules.
- Assume Docker evaluation has no internet. Bundle required weights/assets inside the image or use the competition-prescribed storage method.
- Always test schema first before optimizing model quality.

---

## 10. Who should work on what

Suggested split:

- Person A: **AE** — highest weight, heuristic/RL/planning.
- Person B: **NLP** — retrieval, chunking, extractive/generative QA.
- Person C: **CV + Noise** — detector and adversarial image perturbations.
- Person D: **ASR + integration** — ASR model, Docker builds, GCP testing.
- Everyone: help with `til test`, runtime, debugging, and documentation.

If we have fewer people, combine:

- AE owner
- NLP owner
- CV/ASR owner
- Integration/GCP owner

---

## 11. Commands cheat sheet

Local:

```bash
cd /Users/ethankok/Projects/til
source .venv/bin/activate
git status
git add .
git commit -m "message"
git push
```

GCP:

```bash
cd /home/jupyter/til
git pull
git submodule update --init
micromamba activate env
pip install -r requirements-dev.txt
```

Official loop:

```bash
til build TASK
til test TASK
til submit TASK
```

Replace `TASK` with:

```text
asr cv noise nlp ae
```

---

## 12. Immediate next tasks

- Verify Docker builds once GCP is accessible.
- Create first private GitHub repo and invite teammates.
- Decide track: novice or advanced.
- Get access to official data folders on Workbench.
- Run official `til test` for all five baselines.
- Start improving NLP and AE first.
