# DSTA BrainHack TIL-AI 2025 — Team D Learnings

This document outlines the key algorithmic strategies, framework extensions, and audio/vision preprocessing pipelines from a public TIL-2025 team repository (Team D, Semifinalists) and compares them to our current **TIL 2026** codebase.

---

## 🎤 Automatic Speech Recognition (ASR)

### 1. Model Configuration & Settings
* **Team D Approach:** Used `faster-whisper` (`WhisperModel`) in `float16` precision with `beam_size=5`.
* **Our Current Approach:** `faster-whisper` (`distil-large-v3`) with `beam_size=1`, `temperature=0.0` (greedy decoding), and no-speech thresholds.
* **Comparison & Key Learnings:**
  * **Beam Search vs. Greedy Decoding:** Beam search with size 5 improves transcription quality for highly ambiguous or noisy audio, but at the cost of significantly higher latency. In our current code, we use greedy decoding (`beam_size=1`) to maximize the speed score. Since the test dataset has Gaussian noise, testing whether `beam_size=3` or `beam_size=5` improves accuracy on GPU without violating the latency budget is a valuable lever.

### 2. Post-processing Text Normalization
* **Team D Approach:** Implemented a custom regex corrector (`word_correction.py`) to map special tokens, remove punctuation, and normalize spacing. However, while imported in `asr_manager.py`, it was not actually called in their final runtime inference pipeline.
* **Our Current Approach:** Digits-to-words conversion and basic whitespace cleaning.
* **Comparison & Key Learnings:** While written, failing to call the cleanup script in production shows a mismatch between training/validation logic and server deployment. We must ensure our text normalization steps are fully integrated and tested in the active server script.

---

## 🖼️ Computer Vision (CV)

### 1. Slicing Aided Hyper Inference (SAHI)
* **Team D Approach:** Integrated **SAHI** using `slice_height=320`, `slice_width=320`, and `overlap_ratio=0.2`.
* **Our Current Approach:** Custom tiled inference splits the image into `2x2`, `2x1`, or `3x2` overlapping grids.
* **Comparison & Key Learnings:**
  * **Standard Slicing Library:** SAHI is a highly optimized library specifically built for sliced inference. It contains built-in NMS merging algorithms (like `greedy_nms`) that handle boundary box merging across slices more gracefully than custom intersection-over-union algorithms.
  * **Latency Tradeoff:** Running SAHI sliced inference on a 320x320 patch size creates a large number of slices (e.g. 15-20 slices for a 1920x1080 image), which can run very slowly. Our custom tiled inference (`2x2` or `2x1`) is a lighter alternative that balances small-object recall and latency.

### 2. Preprocessing & Aspect Ratio Preservation
* **Team D Approach:** Implemented a custom `LetterboxTransform` to resize and pad images to 1280x1280 while preserving the aspect ratio.
* **Our Current Approach:** Standard PIL/OpenCV resizing.
* **Comparison & Key Learnings:** preserving aspect ratio via letterboxing (padding empty dimensions with black pixels) ensures that the aspect ratio of the bounding boxes does not distort, which is crucial for YOLO backbones trained with letterboxed data.

---

## 🤖 Reinforcement Learning / Autonomous Exploration (AE)

### 1. Temporal Dynamics via Frame Stacking
* **Team D Approach:** Experimented with **Frame Stacking** (`rl_manager_stacked.py`), combining multiple consecutive observation states into a single model input. Specifically, they stacked the last 4 viewcones (leading to `INPUT_FEATURES = 1128`).
* **Our Current Approach:** Heuristic planning based on the current step's map state.
* **Comparison & Key Learnings:**
  * **Temporal Context:** In a dynamic environment with moving guards, a single step's observation does not convey velocity or direction. Stacked frames (stacking the last 4 viewcones) allow a feedforward neural network to capture the movement vectors of enemy agents. This is a critical lever if using a model-free RL approach.

### 2. Multi-Agent Reinforcement Learning (MARL)
* **Team D Approach:** Concurrently trained scout and guard agents in the custom environment using CTDE (Centralized Training with Decentralized Execution) DQN and PPO.
