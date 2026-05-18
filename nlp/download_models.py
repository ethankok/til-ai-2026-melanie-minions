"""Pre-download model weights into the image so /nlp can run offline.

Run at build time via the Dockerfile. Outputs to /workspace/models, which
matches NLP_MODEL_DIR in nlp_manager.py.

v14-llm-rag adds Qwen2.5-7B-Instruct-AWQ as the answerer. We fetch it via
huggingface_hub.snapshot_download because vLLM expects the AWQ weights/config
on disk, not the transformers cache layout. Set NLP_SKIP_LLM_DOWNLOAD=1 (or
build with a different target) to skip the ~5 GB LLM fetch when iterating on
the retrieval-only path.
"""

import os
from pathlib import Path

from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

OUT = Path("/workspace/models")

SPECS = [
    ("BAAI/bge-small-en-v1.5", AutoModel, "bge-small-en-v1.5"),
    (
        "BAAI/bge-reranker-base",
        AutoModelForSequenceClassification,
        "bge-reranker-base",
    ),
    (
        "deepset/roberta-base-squad2",
        AutoModelForQuestionAnswering,
        "roberta-base-squad2",
    ),
    (
        "kiddothe2b/ModernBERT-base-squad2",
        AutoModelForQuestionAnswering,
        "modernbert-base-squad2",
    ),
]

# v14d-qwen3-8b: upsizes from v14c's 4B to 8B on the same Qwen3 family.
# Direct test of whether v14c's accuracy drop (0.754 → 0.659 local) was 4B
# capacity floor or Qwen3-Instruct paraphrase tendency. Using the official
# Qwen/Qwen3-8B-AWQ (base, with thinking mode default) — apply_chat_template
# is already called with enable_thinking=False so the chat template
# suppresses <think>...</think> blocks. AWQ weights ~5GB; still comfortable
# on T4 16GB with retriever + reranker + KV cache.
LLM_REPO = os.getenv("NLP_LLM_REPO", "Qwen/Qwen3-8B-AWQ")
LLM_LOCAL = "llm"


def _download_retrievers() -> None:
    for repo, model_cls, local_name in SPECS:
        dest = OUT / local_name
        print(f"--> {repo} -> {dest}")
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        mdl = model_cls.from_pretrained(repo, trust_remote_code=True)
        if local_name == "modernbert-base-squad2":
            arch = getattr(mdl.config, "architectures", None)
            print(f"[download_models] ModernBERT QA architectures={arch}")
        tok.save_pretrained(dest)
        mdl.save_pretrained(dest)


def _download_llm() -> None:
    if os.getenv("NLP_SKIP_LLM_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
        print(f"[download_models] skipping LLM download (NLP_SKIP_LLM_DOWNLOAD set)")
        return
    from huggingface_hub import snapshot_download

    dest = OUT / LLM_LOCAL
    print(f"--> {LLM_REPO} -> {dest}")
    # allow_patterns avoids pulling random extras; AWQ weights are .safetensors.
    snapshot_download(
        repo_id=LLM_REPO,
        local_dir=str(dest),
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "tokenizer*",
            "*.model",
            "*.tiktoken",
        ],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _download_retrievers()
    _download_llm()


if __name__ == "__main__":
    main()
