"""Pre-download model weights into the image so /nlp can run offline.

Run at build time via the Dockerfile. Outputs to /workspace/models, which
matches NLP_MODEL_DIR in nlp_manager.py.
"""

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
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for repo, model_cls, local_name in SPECS:
        dest = OUT / local_name
        print(f"--> {repo} -> {dest}")
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = model_cls.from_pretrained(repo)
        tok.save_pretrained(dest)
        mdl.save_pretrained(dest)


if __name__ == "__main__":
    main()
