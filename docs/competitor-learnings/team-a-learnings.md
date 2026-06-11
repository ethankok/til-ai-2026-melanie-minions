# DSTA BrainHack TIL-AI 2025 — Team A Learnings

This document outlines the key design patterns, model configurations, and hyperparameters from a public TIL-2025 team repository (Team A, 5th place in Finals) and evaluates how they compare to our current **TIL 2026** codebase.

---

## 🎤 Automatic Speech Recognition (ASR)

### 1. Model Selection & Architecture
* **Team A Approach:** Initially experimented with fine-tuning `facebook/wav2vec2-large-960h` using CTC loss, but their final submitted pipeline (`asr_manager.py`) deployed a fine-tuned Whisper model (`WhisperForConditionalGeneration`) with 4-bit quantization (using `BitsAndBytesConfig`) to balance transcription accuracy and memory constraints.
* **Our Current Approach:** Zero-shot inference with `faster-whisper` (`distil-large-v3`) in `CTranslate2` format on GPU/CPU, biased using an in-world slang prompt.
* **Comparison & Key Learnings:**
  * **Quantized Whisper vs. Wav2Vec2:** While Wav2Vec2 is smaller and faster, the team's final choice to deploy Whisper with 4-bit quantization indicates that Whisper's robustness against noise was preferred over Wav2Vec2's speed.
  * **Quantization Lever:** Using 4-bit quantization (via `bitsandbytes` or CTranslate2's `int8`/`float16`) is a crucial optimization to load large models under constrained GPU environments.

### 2. Audio Preprocessing
* **Team A Approach:** Standardized audio preprocessing by resampling to 16kHz, converting to mono (single-channel), and normalizing volume.
* **Our Current Approach:** Identical resampling to 16kHz using `librosa`/`numpy` interpolation and mono-conversion.
* **Comparison & Key Learnings:** Our preprocessing matches best-practices; however, we also incorporate an aggressive sub-1.5s silence gate to suppress Whisper hallucinations, which is a major robustness lever.

---

## 🖼️ Computer Vision (CV)

### 1. The P2 Layer (High-Resolution Feature Maps)
* **Team A Approach:** Utilized a custom YOLOv8 configuration `yolov8n-p2.yaml` for training.
* **Our Current Approach:** Standard pretrained YOLOv8/RT-DETR or OWLv2, using a multi-pass tiled inference mode (`2x2`, `3x2`) to catch small objects.
* **Comparison & Key Learnings:**
  * **P2 Detection Lever:** The P2 layer adds a stride-4 feature map to the YOLO backbone (compared to the standard P5 layers with strides 8, 16, 32). This preserves fine details of very small objects early in the network.
  * **Latency Advantage:** Our current codebase relies on overlapping image cropping (tiling) to detect small objects. While highly accurate, tiling scales latency linearly with the number of tiles (e.g., 4x or 6x slower). Fine-tuning a `yolov8-p2` or `rtdetr-p2` model would allow us to detect small objects in a single forward pass, saving valuable GPU milliseconds and boosting speed score.

---

## 🤖 Reinforcement Learning / Autonomous Exploration (AE)

### 1. Model Architecture
* **Team A Approach:** Implemented a Convolutional Deep Q-Network (CNN-DQN) combining spatial viewcone processing with state features.
  * **CNN Backbone:** Processed the viewcone using two Conv2D layers (`Conv2d(8, 16, k=3, s=1, p=1)` -> `ReLU` -> `Conv2d(16, 32, k=3, s=1, p=1)` -> `ReLU`).
  * **MLP Feature Fusion:** Flattened CNN outputs and concatenated them with direction, normalized location, scout role, and normalized step before feeding into a two-layer MLP (`128 -> 128 -> 5`).
* **Our Current Approach:** A stateful, rule-based heuristic planner (`AEManager`) that tracks frontiers, objects, base defense, and plans paths via Dijkstra/A*.
* **Comparison & Key Learnings:**
  * **Feature Engineering:** Team A unpacked the integer tile values from the 7x5 viewcone into 8 binary channels representing the presence of wall segments, agents, and resources. Our current heuristic uses similar channel unpacking but does not feed them to a neural network.
  * **Determinism vs. Adaptability:** RL approaches are highly adaptive to dynamic patterns (e.g. dodging guards) but lack the path-planning guarantees of Dijkstra. Our rule-based heuristic is deterministic, respects action masks perfectly, and avoids training overhead, but could benefit from hybridizing with a trained value function to estimate "long-term reward density" when choosing distant frontiers.

### 2. Training Optimization
* **Team A Approach:** Implemented Double DQN (to stabilize Q-value targets), Prioritized Experience Replay (PER) with a SumTree, and decay epsilon-greedy exploration.
