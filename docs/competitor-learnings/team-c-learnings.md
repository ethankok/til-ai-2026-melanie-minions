# DSTA BrainHack TIL-AI 2025 — Team C Learnings

This document outlines the advanced network architectures, model training techniques, and inference pipelines from a public TIL-2025 team repository (Team C, Finalists) and compares them to our current **TIL 2026** codebase.

---

## 🎤 Automatic Speech Recognition (ASR)

### 1. Model Compilation & FP16 Quantization
* **Team C Approach:**
  * Shipped `faster-whisper` (`WhisperModel`, `compute_type="float16"`).
  * Experimented with a custom Hugging Face Whisper model using `torch.compile(self.model, mode="reduce-overhead")` and `fp16` quantization.
* **Our Current Approach:** `faster-whisper` (`distil-large-v3`) in `CTranslate2` format on GPU with `compute_type="float16"`.
* **Comparison & Key Learnings:**
  * **PyTorch 2.0 Compilation:** For standard Hugging Face models, `torch.compile(model, mode="reduce-overhead")` compiles the model's computation graph into optimized CUDA kernels, significantly reducing CPU overhead during batch inference.
  * **CTranslate2 Advantage:** `faster-whisper` utilizes the `CTranslate2` custom C++ engine, which is already heavily optimized and usually outperforms compiled PyTorch models in memory usage and throughput. Our current implementation is robust and optimal.

---

## 🤖 Reinforcement Learning / Autonomous Exploration (AE)

### 1. Advanced DQN Architecture
* **Team C Approach:** Designed a highly sophisticated deep Q-network (`dqn_model.py`) incorporating the following features:
  * **Attention Layer:** A sub-network (`nn.Sequential(Linear -> ReLU -> Linear -> Softmax)`) that dynamically weights the 43 engineered input features (35 viewcone elements, 4 direction bits, 1 scout role, 2 location coords, 1 step count) before feature extraction.
  * **Layer Normalization:** Applied `nn.LayerNorm(256)` after linear layers to stabilize inputs during training.
  * **Residual Connections:** Implemented a residual connection (`f1 + f2`) between the first and second feature extraction blocks (`self.feature1` and `self.feature2`) to prevent vanishing gradients.
  * **Dueling Architecture:** Combined a Value stream (estimating state value $V(s)$) and an Advantage stream (estimating advantage $A(s,a)$) via $Q(s,a) = V(s) + A(s,a) - \text{mean}(A(s, \cdot))$ to learn which states are valuable without learning the effect of each action.
  * **Orthogonal Weight Initialization:** Initialized weights using `nn.init.orthogonal_(m.weight, gain=1)` to improve gradient flow.
* **Our Current Approach:** Stateful rule-based planner (`AEManager`) with Dijkstra path planning.
* **Comparison & Key Learnings:**
  * **Value Stream Hybridization:** If we decide to integrate neural networks with our rule-based heuristic, Team C's Dueling DQN architecture is the gold-standard network design.
  * **Attention-Gated Feature Selection:** In high-dimensional visual state spaces, an Attention Layer helps the model filter out irrelevant tiles (like distant walls) and focus on crucial items (like nearby guards or fuel).
  * **Normalization Levers:** Normalizing location (`location / 15.0`), step count (`step / 100.0`), and viewcone integers (`viewcone / 255.0`) to a `[0, 1]` range is critical for stable neural network training and prevents gradient explosion.
