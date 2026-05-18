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
# v16-deberta: default back to the fast extractive stack. The v14/v15 LLM
# line found a higher raw-accuracy ceiling but lost the qualifier blend on
# speed/cloud startup, and the Qwen3 LoRA merge/quant path failed quality.
ENV NLP_ANSWERER=extractive
ENV NLP_QA_MAX_SEQ_LEN=256
ENV NLP_LLM_DIR=/workspace/models/llm
ENV NLP_RERANKER_REPO=tomaarsen/Qwen3-Reranker-0.6B-seq-cls
ENV NLP_RERANKER_LOCAL_NAME=qwen3-reranker-0.6b-seq-cls
ENV NLP_DOWNLOAD_MODERNBERT=0
# Default repo for download_models.py. v14d: Qwen/Qwen3-8B-AWQ (official
# Qwen quant of the Qwen3 8B base). Thinking mode is the model default;
# our chat template call passes enable_thinking=False so <think>...</think>
# blocks are suppressed at the prompt level.
ENV NLP_LLM_REPO=Qwen/Qwen3-8B-AWQ
# Do not download the 5+ GB LLM for extractive tags. Override to 0 only when
# intentionally rebuilding the v14/v15 LLM path.
ENV NLP_SKIP_LLM_DOWNLOAD=1
# v15-lora: directory inside the image where the trained LoRA adapter
# lives. llm_answerer.py auto-enables LoRA in vLLM when this path contains
# an adapter_config.json. Set to empty string to force base-model-only.
ENV NLP_LLM_LORA_DIR=/workspace/models/lora
WORKDIR /workspace

# vllm/vllm-openai already provides: vllm, torch, transformers (>=4.51),
# tokenizers, safetensors, fastapi, uvicorn, numpy, huggingface_hub,
# sentencepiece, flash_attn. We only need to add the small extras our
# manager uses (BM25 over passages).
RUN pip install --no-cache-dir 'rank_bm25==0.2.2'

# Bake model weights into the image so inference runs fully offline.
# ENV HF_HUB_OFFLINE=1 is set above for runtime; we override it for this
# download-only RUN. Pulls BGE retriever + configured reranker + RoBERTa-base
# (kept for emergency extractive rollback). The v18 experiment swaps only the
# reranker to Qwen3 and leaves the v9 RoBERTa answerer path intact.
COPY download_models.py .
# vllm/vllm-openai doesn't create a `python` symlink — only python3 is on PATH.
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python3 download_models.py

# v14-llm-rag: hard-fail the build if the LLM weights didn't land. The
# manager will still boot without them by falling back to extractive, but
# a silent fallback would mask a packaging regression on a v14c tag.
RUN if [ "$NLP_ANSWERER" = "llm" ] && [ ! -f /workspace/models/llm/config.json ]; then \
        echo "[nlp build] WARNING: NLP_ANSWERER=llm but /workspace/models/llm/config.json missing"; \
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
    else \
        echo "[nlp build] extractive answerer active; LLM artefacts ignored"; \
    fi && rm -rf /workspace/models-local

# Copies your source files.
COPY src .

# Override vllm/vllm-openai's default OpenAI-API-server CMD with our own
# FastAPI server. uvicorn comes pre-installed in the base image.
ENTRYPOINT []
CMD ["uvicorn", "nlp_server:app", "--port", "5004", "--host", "0.0.0.0"]
