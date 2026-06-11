# DSTA BrainHack TIL-AI 2025 — Team B Learnings

This document outlines the design decisions, system architectures, and model deployment strategies from a public TIL-2025 team repository (Team B, Novice Category) and compares them to our current **TIL 2026** codebase.

---

## 🖼️ Computer Vision (CV)

### 1. Model Selection & TensorRT Compilation
* **Team B Approach:** Used `RT-DETRv2-M` for fast and accurate object detection and compiled the model into a **TensorRT Engine** (`model.engine`).
* **Our Current Approach:** Running standard PyTorch/Ultralytics models via native Python wrappers (`YOLO` or `RTDETR` classes).
* **Comparison & Key Learnings:**
  * **TensorRT Acceleration:** TensorRT optimizes neural network graphs by merging layers, performing precision calibration (FP16/INT8), and generating target-specific kernels for the execution hardware.
  * **Latency Optimization:** Since the MBS desktop has a Blackwell RTX 5070 Ti, compiling our final models to TensorRT engines is the ultimate lever for speed. It allows the vision model to run in a few milliseconds per image, leaving substantial time budget for other tasks.
  * **Post-processing NMS:** Team B used PyTorch's native `torchvision.ops.nms` on GPU to merge boxes across categories after exporting. We should leverage similar GPU-accelerated NMS if we combine multiple models.

---

## 🤖 Reinforcement Learning / Autonomous Exploration (AE)

### 1. Recurrent PPO & Supervised Guard Strategy
* **Team B Approach:**
  * **Best Performing RL (PPO + LSTM):** Designed and trained a recurrent **PPO + LSTM** policy (`train_ppolstm.py`) to manage temporal observations and memory of visited states in the partially observable gridworld.
  * **Supervised Imitation for Guards:** Collected human keyboard demonstration trajectories (WASD inputs) for guards (`guard_demos.pkl`) and trained an MLP-based `SupervisedPolicy` classification model (`guard_policy.pt`), rather than using it for the Scout.
  * **No CNN-DQN:** Despite early claims in their README.md overview, no CNN-DQN code or weights exist in their actual repository; they relied on standard BFS grid pathfinding for mapping shortest paths.
* **Our Current Approach:** Stateful Dijkstra/A* pathfinding with rule-based objective prioritization.
* **Comparison & Key Learnings:**
  * **Recurrent Memory (LSTM):** In partially observable settings where scouts and guards move dynamically, adding an LSTM cell (like their `PPOLSTMPolicy` containing a hidden dimension of 128) allows the agent to maintain an internal map memory (such as tracking recently visited cells) without raw frame stacking.
  * **Seeding Behavior with Demonstrations:** Training agents with supervised demonstration data (imitation learning) is an effective way to establish basic navigation capabilities before fine-tuning via reinforcement learning.

---

## 📄 Optical Character Recognition (OCR) — *For Reference*

### 1. Tesseract Preprocessing & Tuning
* **Team B Approach:** Preprocessed images using grayscale conversion, dynamic resizing (downscaling if the max dimension exceeded 960px to preserve speed), and Otsu binarization (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`). They used Tesseract config `--oem 3 --psm 6` (assume single uniform block of text).
* **Our Current Approach:** (Our main focus is ASR, CV, and AE/RL, but understanding layout preprocessing is valuable).
* **Key Learnings:** Otsu's thresholding is highly effective for clean binary segmentation of scanned document text, reducing noise in poor lighting/Gaussian noise.
