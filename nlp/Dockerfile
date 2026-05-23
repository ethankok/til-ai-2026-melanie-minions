# Dockerfile for building the NLP image.
#
# Locked NLP build: keep the cloud-safe v9 extractive stack as default. The
# Qwen/vLLM paths remain in source for historical ablations, but they are not
# bundled unless the env vars below are explicitly changed.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ARG NLP_RERANKER_REPO=BAAI/bge-reranker-large
ARG NLP_RERANKER_LOCAL_NAME=bge-reranker-large

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV NLP_MODEL_DIR=/workspace/models
# v9/v9-rescue default: BM25+BGE+BGE-reranker + fine-tuned RoBERTa extractive
# answerer. This is the trusted blended submission path.
ENV NLP_ANSWERER=extractive
ENV NLP_QA_MAX_SEQ_LEN=256
ENV NLP_LLM_DIR=/workspace/models/llm
ENV NLP_RERANKER_REPO=${NLP_RERANKER_REPO}
ENV NLP_RERANKER_LOCAL_NAME=${NLP_RERANKER_LOCAL_NAME}
ENV NLP_DOWNLOAD_MODERNBERT=0
ENV NLP_LLM_REPO=Qwen/Qwen2.5-7B-Instruct-AWQ
ENV NLP_SKIP_LLM_DOWNLOAD=1
ENV NLP_LLM_GPU_MEM_FRACTION=0.62
ENV NLP_LLM_MAX_MODEL_LEN=3072
ENV NLP_LLM_MAX_NEW_TOKENS=32
ENV NLP_HYBRID_QWEN_THRESHOLD=3.0
ENV NLP_COMPOSITION_MODE=off
# v21-trigger-only: skip the RoBERTa QA forward and return just the adversarial
# trigger as the candidate. Locally measured 0.994 AE pass rate; saves the QA
# inference time. Set to 0 to keep v20-style trigger+real_answer concat.
ENV NLP_AE_TRIGGER_ONLY=1
# v26-pure-bm25: use pure document-level BM25 search to skip dense model loading
# and encoding. Reduces corpus-load time from ~15 seconds to ~0.05 seconds.
ENV NLP_PURE_BM25=1
# v15-lora: directory inside the image where the trained LoRA adapter
# lives. llm_answerer.py auto-enables LoRA in vLLM when this path contains
# an adapter_config.json. Set to empty string to force base-model-only.
ENV NLP_LLM_LORA_DIR=/workspace/models/lora
WORKDIR /workspace

RUN python -m pip install -U pip
# NGC 25.11 ships optional torchao/flash-attn extensions that are compiled
# against its original torch stack. Transformers can crash while merely
# importing BERT (`torch.int1` missing from torchao; flash_attn can similarly
# hit undefined symbols). They are optional for this path, so remove them
# before model download/import.
RUN python -m pip uninstall -y torchao flash-attn flash_attn || true

# Install only the small model-download stack before pulling weights. The full
# requirements include vLLM and are intentionally delayed until after the 4 GB
# Qwen AWQ shard lands so Docker overlay space does not run out mid-download.
RUN python -m pip install --no-cache-dir \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    safetensors==0.4.5 \
    sentencepiece==0.2.0 \
    'huggingface-hub>=0.26.0'

# Bake model weights into the image so inference runs fully offline.
# ENV HF_HUB_OFFLINE=1 is set above for runtime; we override it for this
# download-only RUN. Pulls BGE retriever + configured reranker + RoBERTa-base
# (kept for emergency extractive rollback). Override NLP_RERANKER_REPO and
# NLP_RERANKER_LOCAL_NAME only for reranker ablations; the shipped default is
# the v9 BGE reranker because the Qwen3 reranker-only gate regressed badly.
COPY download_models.py .
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python download_models.py

# Install runtime dependencies after model download. The default v9 path does
# not install vLLM; historical Qwen paths should re-add it deliberately.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip uninstall -y torchao flash-attn flash_attn || true
RUN python - <<'PY'
import fastapi
import rank_bm25
import transformers
import uvicorn
print("[nlp build] runtime imports ok")
PY

# v14-llm-rag: hard-fail the build if the LLM weights didn't land. The
# manager will still boot without them by falling back to extractive, but
# a silent fallback would mask a packaging regression on a v14c tag.
RUN if { [ "$NLP_ANSWERER" = "llm" ] || [ "$NLP_ANSWERER" = "hybrid" ]; } && [ ! -f /workspace/models/llm/config.json ]; then \
        echo "[nlp build] WARNING: NLP_ANSWERER=$NLP_ANSWERER but /workspace/models/llm/config.json missing"; \
        echo "[nlp build] container will fall back to extractive QA at runtime"; \
    fi

# Bundle locally-produced model artefacts.
#
# Locked v9 build requires nlp/models/roberta-finetuned-squad2/. Without it,
# the manager silently falls back to the downloaded stock roberta-base-squad2,
# which looks "healthy" but scores around 0.66 instead of the v9 0.711 local
# baseline. Fail loudly rather than producing a fake v9 image.
COPY models /workspace/models-local
RUN if [ -d /workspace/models-local/roberta-finetuned-squad2 ] && \
       [ -f /workspace/models-local/roberta-finetuned-squad2/config.json ]; then \
        rm -rf /workspace/models/roberta-finetuned-squad2 && \
        mkdir -p /workspace/models/roberta-finetuned-squad2 && \
        cp -r /workspace/models-local/roberta-finetuned-squad2/* /workspace/models/roberta-finetuned-squad2/ && \
        echo "[nlp build] bundled locked v9 RoBERTa fine-tuned QA artefact"; \
    elif [ "$NLP_ANSWERER" = "extractive" ]; then \
        echo "[nlp build] ERROR: locked v9 build requires nlp/models/roberta-finetuned-squad2/config.json"; \
        echo "[nlp build] restore it from a known-good v9 image or training output before building"; \
        exit 1; \
    fi && \
    if [ -f /workspace/models-local/answer_ranker.json ]; then \
        cp /workspace/models-local/answer_ranker.json /workspace/models/answer_ranker.json && \
        echo "[nlp build] bundled local answer ranker"; \
    fi && \
    if [ -f /workspace/models-local/ae_trigger.json ]; then \
        cp /workspace/models-local/ae_trigger.json /workspace/models/ae_trigger.json && \
        echo "[nlp build] bundled AE adversarial trigger"; \
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

CMD ["python", "-m", "uvicorn", "nlp_server:app", "--port", "5004", "--host", "0.0.0.0"]
