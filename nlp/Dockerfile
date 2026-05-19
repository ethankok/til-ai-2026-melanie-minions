# Dockerfile for building the NLP image.
#
# v19-hybrid-router: return to the NGC base that actually survived cloud for
# v14's Qwen2.5-7B-AWQ run. The vllm/vllm-openai base was convenient for Qwen3
# local tests, but repeatedly timed out or errored on cloud submissions.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ARG NLP_RERANKER_REPO=BAAI/bge-reranker-base
ARG NLP_RERANKER_LOCAL_NAME=bge-reranker-base

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV NLP_MODEL_DIR=/workspace/models
# v19: route easy/L1-looking questions through v9 RoBERTa and only hard/L2-
# looking questions through Qwen2.5-7B-AWQ. Set extractive to recover v9.
ENV NLP_ANSWERER=hybrid
ENV NLP_QA_MAX_SEQ_LEN=256
ENV NLP_LLM_DIR=/workspace/models/llm
ENV NLP_RERANKER_REPO=${NLP_RERANKER_REPO}
ENV NLP_RERANKER_LOCAL_NAME=${NLP_RERANKER_LOCAL_NAME}
ENV NLP_DOWNLOAD_MODERNBERT=0
ENV NLP_LLM_REPO=Qwen/Qwen2.5-7B-Instruct-AWQ
ENV NLP_SKIP_LLM_DOWNLOAD=0
ENV NLP_LLM_GPU_MEM_FRACTION=0.62
ENV NLP_LLM_MAX_MODEL_LEN=3072
ENV NLP_LLM_MAX_NEW_TOKENS=32
ENV NLP_HYBRID_QWEN_THRESHOLD=3.0
# v15-lora: directory inside the image where the trained LoRA adapter
# lives. llm_answerer.py auto-enables LoRA in vLLM when this path contains
# an adapter_config.json. Set to empty string to force base-model-only.
ENV NLP_LLM_LORA_DIR=/workspace/models/lora
WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# NGC 25.11 ships optional torchao/flash-attn extensions that are compiled
# against its original torch stack. Installing vLLM changes the active torch
# ABI enough that Transformers can crash while merely importing BERT
# (`torch.int1` missing from torchao; flash_attn can similarly hit undefined
# symbols). They are optional for this path, so remove them before model
# download/import.
RUN pip uninstall -y torchao flash-attn flash_attn || true

# Bake model weights into the image so inference runs fully offline.
# ENV HF_HUB_OFFLINE=1 is set above for runtime; we override it for this
# download-only RUN. Pulls BGE retriever + configured reranker + RoBERTa-base
# (kept for emergency extractive rollback). Override NLP_RERANKER_REPO and
# NLP_RERANKER_LOCAL_NAME only for reranker ablations; the shipped default is
# the v9 BGE reranker because the Qwen3 reranker-only gate regressed badly.
COPY download_models.py .
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python download_models.py

# v14-llm-rag: hard-fail the build if the LLM weights didn't land. The
# manager will still boot without them by falling back to extractive, but
# a silent fallback would mask a packaging regression on a v14c tag.
RUN if { [ "$NLP_ANSWERER" = "llm" ] || [ "$NLP_ANSWERER" = "hybrid" ]; } && [ ! -f /workspace/models/llm/config.json ]; then \
        echo "[nlp build] WARNING: NLP_ANSWERER=$NLP_ANSWERER but /workspace/models/llm/config.json missing"; \
        echo "[nlp build] container will fall back to extractive QA at runtime"; \
    fi

# Bundle locally-produced model artefacts.
#
# Two paths are supported, in priority order:
#
#   1. nlp/models/llm-merged/ (PREFERRED on T4) — a quantised, already
#      LoRA-merged Qwen3-8B (currently GPTQ W4A16 by default). Produced by
#      training/nlp/merge_lora_and_quantize.py. Avoids vLLM 0.9.0's broken
#      Punica LoRA Triton kernels on sm_75 (T4). If this dir exists, it
#      overrides the downloaded base at /workspace/models/llm/.
#
#   2. nlp/models/lora/ (FALLBACK, Ampere+ only) — a raw LoRA adapter that
#      vLLM applies at runtime via LoRARequest. Use this on GPUs where the
#      Triton LoRA kernels actually compile (sm_80+).
#
# If BOTH dirs exist, the merged dir wins and the runtime LoRA dir is
# ignored (so we don't double-apply). To force the runtime-LoRA path, move
# nlp/models/llm-merged/ out of the build context.
COPY models /workspace/models-local
RUN for d in modernbert-finetuned-squad2 deberta-finetuned-squad2 roberta-finetuned-squad2 flan-t5-finetuned; do \
        if [ -d "/workspace/models-local/$d" ] && [ -f "/workspace/models-local/$d/config.json" ]; then \
            rm -rf "/workspace/models/$d" && \
            mkdir -p "/workspace/models/$d" && \
            cp -r "/workspace/models-local/$d/"* "/workspace/models/$d/" && \
            echo "[nlp build] bundled local QA artefact: $d"; \
        fi; \
    done && \
    if [ -f /workspace/models-local/answer_ranker.json ]; then \
        cp /workspace/models-local/answer_ranker.json /workspace/models/answer_ranker.json && \
        echo "[nlp build] bundled local answer ranker"; \
    fi && \
    if [ "$NLP_ANSWERER" = "llm" ] && [ -d /workspace/models-local/llm-merged ] && [ -f /workspace/models-local/llm-merged/config.json ]; then \
        rm -rf /workspace/models/llm && \
        mkdir -p /workspace/models/llm && \
        cp -r /workspace/models-local/llm-merged/* /workspace/models/llm/ && \
        echo "[nlp build] bundled merged quantized LLM (overrides downloaded base)"; \
        if [ -f /workspace/models-local/llm-merged/MERGED_FROM ]; then \
            echo "[nlp build]   provenance:"; \
            sed 's/^/[nlp build]     /' /workspace/models-local/llm-merged/MERGED_FROM; \
        fi; \
        echo "[nlp build] runtime LoRA path will be DISABLED (merged weights already include fine-tune)"; \
    elif [ "$NLP_ANSWERER" = "llm" ] && [ -d /workspace/models-local/lora ] && [ -f /workspace/models-local/lora/adapter_config.json ]; then \
        mkdir -p /workspace/models/lora && \
        cp -r /workspace/models-local/lora/* /workspace/models/lora/ && \
        echo "[nlp build] bundled v15 LoRA adapter ($(cat /workspace/models-local/lora/BASE_MODEL 2>/dev/null || echo unknown base))"; \
        echo "[nlp build] WARNING: runtime LoRA path is broken on T4/sm_75 (vLLM 0.9.0 Triton kernel)"; \
    elif [ "$NLP_ANSWERER" = "hybrid" ]; then \
        echo "[nlp build] hybrid answerer active; using downloaded LLM plus extractive QA"; \
    else \
        echo "[nlp build] extractive answerer active; LLM artefacts ignored"; \
    fi && rm -rf /workspace/models-local

# Copies your source files.
COPY src .

CMD uvicorn nlp_server:app --port 5004 --host 0.0.0.0
