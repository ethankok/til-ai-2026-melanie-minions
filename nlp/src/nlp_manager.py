"""NLP RAG manager — hybrid retrieval + cross-encoder rerank + extractive QA.

Pipeline at corpus load:
  - Sentence-split each document, slide a 3-sentence window with 1-sentence
    overlap to build passages. Each passage remembers its parent document
    index (used to emit DOC-XXXX IDs).
  - Index every passage in (a) BM25Okapi and (b) a dense BGE encoder.

Pipeline at query:
  - Hybrid score = z(BM25) + z(dense_cos). Take top-K passages.
  - Cross-encoder rerank the top-K to refine ordering.
  - documents: top-3 *unique* parent doc IDs in reranked order.
  - answer: run an extractive QA span model over the top reranked passages,
    pick the highest-confidence span, return ≤64 tokens.

Document ID assignment is positional 1-indexed: the i-th document in the
load list becomes "DOC-{i+1:04d}". The evaluator has no other way to
identify documents (it only sends a list of plain strings), so this is the
only viable scheme.

Models (all bundled into the image; see download_models.py):
  - dense:    BAAI/bge-small-en-v1.5
  - reranker: BAAI/bge-reranker-base
  - qa:       deepset/roberta-base-squad2
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("NLP_MODEL_DIR", "/workspace/models"))
DENSE_DIR = MODEL_DIR / "bge-small-en-v1.5"
RERANKER_DIR = MODEL_DIR / "bge-reranker-base"
QA_DIR = MODEL_DIR / "roberta-base-squad2"

CHUNK_SENTENCES = 3
CHUNK_OVERLAP = 1
TOP_K_RETRIEVE = 30          # passages handed to reranker
TOP_K_RERANK = 10            # passages handed to QA
TOP_DOCS_RETURNED = 3        # eval considers first 3
QA_MAX_ANSWER_TOKENS = 64    # eval truncates beyond this
EMBED_BATCH = 64
RERANK_BATCH = 32
QA_BATCH = 16
DENSE_MAX_LEN = 256
RERANK_MAX_LEN = 256
QA_MAX_SEQ_LEN = 384
QA_DOC_STRIDE = 128

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "was", "were", "what", "when", "where", "which", "who",
        "why", "with",
    }
)
_PRINTABLE = set(string.printable)


def _doc_id(idx0: int) -> str:
    """Positional, 1-indexed, zero-padded to 4 digits."""
    return f"DOC-{idx0 + 1:04d}"


def _bm25_tokenize(text: str) -> list[str]:
    return [
        t for t in (m.lower() for m in _TOKEN_RE.findall(text))
        if len(t) > 1 and t not in _STOPWORDS
    ]


def _clean_answer(text: str) -> str:
    """Strip non-printable chars (matches the eval's preprocessing)."""
    return "".join(c for c in text if c in _PRINTABLE).strip()


def _split_sentences(document: str) -> list[str]:
    sents = [s.strip() for s in _SENTENCE_RE.split(document) if s.strip()]
    if not sents and document.strip():
        sents = [document.strip()]
    return sents


def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    std = arr.std()
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


# ----------------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------------


class NLPManager:
    """Hybrid RAG with extractive QA."""

    def __init__(self) -> None:
        self.loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy loaded on first use to keep startup fast and let load_corpus
        # do the heavy lifting in a worker thread.
        self._models_initialized = False
        self._dense_tok = None
        self._dense_model = None
        self._rerank_tok = None
        self._rerank_model = None
        self._qa_tok = None
        self._qa_model = None

        # Corpus state.
        self.documents: list[str] = []
        self.passages: list[str] = []
        self.passage_doc_idx: list[int] = []
        self.bm25: BM25Okapi | None = None
        self.passage_embeds: torch.Tensor | None = None

    # ------------------------------------------------------------------ models

    def _init_models(self) -> None:
        if self._models_initialized:
            return

        dense_path = str(DENSE_DIR) if DENSE_DIR.exists() else "BAAI/bge-small-en-v1.5"
        rerank_path = (
            str(RERANKER_DIR) if RERANKER_DIR.exists() else "BAAI/bge-reranker-base"
        )
        qa_path = str(QA_DIR) if QA_DIR.exists() else "deepset/roberta-base-squad2"

        from transformers import AutoModel

        self._dense_tok = AutoTokenizer.from_pretrained(dense_path)
        # BGE uses an encoder; AutoModel gives the encoder we need for CLS pooling.
        self._dense_model = AutoModel.from_pretrained(dense_path).to(self.device).eval()

        self._rerank_tok = AutoTokenizer.from_pretrained(rerank_path)
        self._rerank_model = (
            AutoModelForSequenceClassification.from_pretrained(rerank_path)
            .to(self.device)
            .eval()
        )

        self._qa_tok = AutoTokenizer.from_pretrained(qa_path)
        self._qa_model = (
            AutoModelForQuestionAnswering.from_pretrained(qa_path)
            .to(self.device)
            .eval()
        )

        if self.device.type == "cuda":
            # Half precision is a ~2x speedup on these small models with no
            # measurable quality loss in our regime.
            self._dense_model = self._dense_model.half()
            self._rerank_model = self._rerank_model.half()
            self._qa_model = self._qa_model.half()

        self._models_initialized = True

    # ----------------------------------------------------------------- corpus

    def _chunk_document(self, doc: str) -> list[str]:
        sents = _split_sentences(doc)
        if not sents:
            return []
        if len(sents) <= CHUNK_SENTENCES:
            return [" ".join(sents)]
        step = max(1, CHUNK_SENTENCES - CHUNK_OVERLAP)
        chunks: list[str] = []
        for i in range(0, len(sents), step):
            window = sents[i : i + CHUNK_SENTENCES]
            if not window:
                break
            chunks.append(" ".join(window))
            if i + CHUNK_SENTENCES >= len(sents):
                break
        return chunks

    @torch.no_grad()
    def _embed_passages(self, passages: list[str]) -> torch.Tensor:
        # BGE: CLS-pool then L2-normalize. No query prefix here (those go on the
        # query side only).
        out = []
        for start in range(0, len(passages), EMBED_BATCH):
            batch = passages[start : start + EMBED_BATCH]
            enc = self._dense_tok(
                batch,
                padding=True,
                truncation=True,
                max_length=DENSE_MAX_LEN,
                return_tensors="pt",
            ).to(self.device)
            hidden = self._dense_model(**enc).last_hidden_state[:, 0]
            hidden = F.normalize(hidden, p=2, dim=1)
            out.append(hidden.float().cpu())
        if not out:
            return torch.empty(0, 384)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def _embed_query(self, query: str) -> torch.Tensor:
        # BGE recommends a query prefix for English retrieval.
        text = "Represent this sentence for searching relevant passages: " + query
        enc = self._dense_tok(
            [text],
            padding=True,
            truncation=True,
            max_length=DENSE_MAX_LEN,
            return_tensors="pt",
        ).to(self.device)
        hidden = self._dense_model(**enc).last_hidden_state[:, 0]
        hidden = F.normalize(hidden, p=2, dim=1)
        return hidden.float().cpu().squeeze(0)

    def load_corpus(self, documents: list[str]) -> None:
        self._init_models()

        self.documents = list(documents)
        self.passages = []
        self.passage_doc_idx = []
        for doc_idx, doc in enumerate(self.documents):
            for chunk in self._chunk_document(doc):
                self.passages.append(chunk)
                self.passage_doc_idx.append(doc_idx)

        if not self.passages:
            self.passages = [""]
            self.passage_doc_idx = [0] if self.documents else [0]

        tokenized = [_bm25_tokenize(p) for p in self.passages]
        # rank_bm25 expects non-empty token lists; guard against pathological docs.
        tokenized = [toks if toks else ["_empty_"] for toks in tokenized]
        self.bm25 = BM25Okapi(tokenized)

        self.passage_embeds = self._embed_passages(self.passages)
        self.loaded = True

    # ------------------------------------------------------------- retrieval

    def _retrieve(self, question: str, k: int) -> list[int]:
        q_tokens = _bm25_tokenize(question) or [question.lower()]
        bm25_scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float32)

        q_embed = self._embed_query(question)
        dense_scores = (self.passage_embeds @ q_embed).numpy().astype(np.float32)

        hybrid = _zscore(bm25_scores) + _zscore(dense_scores)
        # Top-k indices (argpartition then sort for stable ordering).
        k = min(k, hybrid.shape[0])
        if k <= 0:
            return []
        idx = np.argpartition(-hybrid, k - 1)[:k]
        idx = idx[np.argsort(-hybrid[idx])]
        return idx.tolist()

    @torch.no_grad()
    def _rerank(self, question: str, passage_idxs: list[int]) -> list[int]:
        if not passage_idxs:
            return []
        pairs = [(question, self.passages[i]) for i in passage_idxs]
        scores: list[float] = []
        for start in range(0, len(pairs), RERANK_BATCH):
            batch = pairs[start : start + RERANK_BATCH]
            enc = self._rerank_tok(
                [b[0] for b in batch],
                [b[1] for b in batch],
                padding=True,
                truncation=True,
                max_length=RERANK_MAX_LEN,
                return_tensors="pt",
            ).to(self.device)
            logits = self._rerank_model(**enc).logits.squeeze(-1).float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            scores.extend(logits)
        order = sorted(range(len(passage_idxs)), key=lambda i: -scores[i])
        return [passage_idxs[i] for i in order]

    def _top_doc_ids(self, ranked_passage_idxs: Iterable[int]) -> list[str]:
        seen: list[int] = []
        for pidx in ranked_passage_idxs:
            d = self.passage_doc_idx[pidx]
            if d not in seen:
                seen.append(d)
                if len(seen) >= TOP_DOCS_RETURNED:
                    break
        return [_doc_id(d) for d in seen]

    # ------------------------------------------------------------ extraction

    @torch.no_grad()
    def _extract_answer(self, question: str, passage_idxs: list[int]) -> str:
        if not passage_idxs:
            return ""
        contexts = [self.passages[i] for i in passage_idxs[:TOP_K_RERANK]]

        best_text = ""
        best_score = -1e9
        # Process each context separately so a long context doesn't dominate
        # via shared softmax; pick the highest-scoring answer span overall.
        for ctx in contexts:
            enc = self._qa_tok(
                question,
                ctx,
                max_length=QA_MAX_SEQ_LEN,
                truncation="only_second",
                stride=QA_DOC_STRIDE,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt",
            )
            offset_mapping = enc.pop("offset_mapping")
            enc.pop("overflow_to_sample_mapping", None)
            enc_gpu = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self._qa_model(**enc_gpu)
            start_logits = outputs.start_logits.float().cpu()
            end_logits = outputs.end_logits.float().cpu()

            # Identify context-token positions (sequence_ids == 1) so we never
            # pick a span inside the question.
            for feat in range(start_logits.shape[0]):
                seq_ids = enc.sequence_ids(feat)
                valid = torch.tensor(
                    [1.0 if sid == 1 else 0.0 for sid in seq_ids], dtype=torch.float32
                )
                # Mask out non-context tokens with a large negative number.
                mask = (1.0 - valid) * -1e4
                s = start_logits[feat] + mask
                e = end_logits[feat] + mask

                # Take top-k starts/ends, search the best valid span.
                top_starts = torch.topk(s, k=min(20, s.shape[0])).indices.tolist()
                top_ends = torch.topk(e, k=min(20, e.shape[0])).indices.tolist()
                offsets = offset_mapping[feat].tolist()

                for si in top_starts:
                    for ei in top_ends:
                        if ei < si or ei - si + 1 > QA_MAX_ANSWER_TOKENS:
                            continue
                        if seq_ids[si] != 1 or seq_ids[ei] != 1:
                            continue
                        score = float(s[si] + e[ei])
                        if score <= best_score:
                            continue
                        start_char, _ = offsets[si]
                        _, end_char = offsets[ei]
                        if end_char <= start_char:
                            continue
                        cand = ctx[start_char:end_char].strip()
                        if not cand:
                            continue
                        best_score = score
                        best_text = cand

        return _clean_answer(best_text)

    # ----------------------------------------------------------------- query

    def _answer_one(self, question: str) -> dict:
        if not self.loaded or not self.passages:
            return {"documents": [], "answer": ""}

        retrieved = self._retrieve(question, TOP_K_RETRIEVE)
        reranked = self._rerank(question, retrieved)
        documents = self._top_doc_ids(reranked)
        answer = self._extract_answer(question, reranked)
        return {"documents": documents, "answer": answer}

    def qa_batch(self, questions: list[str]) -> list[dict]:
        return [self._answer_one(q) for q in questions]

    # Compatibility shim — old call site returns a bare string.
    def qa(self, question: str) -> dict:
        return self._answer_one(question)
