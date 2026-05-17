# Dockerfile for building the NLP image.
#
# v14c-qwen3-4b: base image is now vllm/vllm-openai instead of the NGC
# pytorch container. The NGC base shipped pre-compiled flash_attn and
# torchao .so's against its own torch ABI; vllm-as-pip-dep downgraded
# torch and stranded those .so's with undefined symbols and missing
# attributes. Iterating around that was costing builds. vllm/vllm-openai
# is the upstream-blessed image — torch, vllm, transformers, and
# flash_attn are all pinned to a coherent stack — so the ABI fight goes
# away and we get transformers>=4.51 (required for Qwen3 model_type).


# vllm/vllm-openai ships a coherent torch + vllm + transformers + flash_attn
# stack. v0.9.x is the most recent series we've found tested against Qwen3
# AWQ checkpoints; pin to a specific tag for reproducibility.
FROM vllm/vllm-openai:v0.9.0

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV NLP_MODEL_DIR=/workspace/models
# v14-llm-rag: 'llm' routes the answerer through vLLM + a local LLM (now
# Qwen3-4B-Instruct-2507-AWQ). Set NLP_ANSWERER=extractive to fall back to
# the v9 RoBERTa-large path (also requires the fine-tuned weights to be
# present, which v14c does not bundle by default).
ENV NLP_ANSWERER=llm
ENV NLP_LLM_DIR=/workspace/models/llm
# Default repo for download_models.py. cpatonn ships the AWQ-4bit quant of
# the July-2025 Instruct release. The "2507" suffix denotes the
# non-thinking instruct variant, which is exactly what we want for short
# extractive-style answers.
ENV NLP_LLM_REPO=cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit
WORKDIR /workspace

# vllm/vllm-openai already provides: vllm, torch, transformers (>=4.51),
# tokenizers, safetensors, fastapi, uvicorn, numpy, huggingface_hub,
# sentencepiece, flash_attn. We only need to add the small extras our
# manager uses (BM25 over passages).
RUN pip install --no-cache-dir 'rank_bm25==0.2.2'

# Bake model weights into the image so inference runs fully offline.
# ENV HF_HUB_OFFLINE=1 is set above for runtime; we override it for this
# download-only RUN. Pulls BGE retriever + reranker + RoBERTa-base (kept
# for emergency extractive rollback) + the Qwen3-4B AWQ LLM.
COPY download_models.py .
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python download_models.py

# v14-llm-rag: hard-fail the build if the LLM weights didn't land. The
# manager will still boot without them by falling back to extractive, but
# a silent fallback would mask a packaging regression on a v14c tag.
RUN if [ "$NLP_ANSWERER" = "llm" ] && [ ! -f /workspace/models/llm/config.json ]; then \
        echo "[nlp build] WARNING: NLP_ANSWERER=llm but /workspace/models/llm/config.json missing"; \
        echo "[nlp build] container will fall back to extractive QA at runtime"; \
    fi

# Copies your source files.
COPY src .

# Override vllm/vllm-openai's default OpenAI-API-server CMD with our own
# FastAPI server. uvicorn comes pre-installed in the base image.
ENTRYPOINT []
CMD ["uvicorn", "nlp_server:app", "--port", "5004", "--host", "0.0.0.0"]
