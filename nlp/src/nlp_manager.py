"""NLP RAG manager — hybrid retrieval + cross-encoder rerank + extractive QA.

Pipeline at corpus load:
  - Sentence-window chunking across each document (3 sentences with 1-sentence
    overlap). Each chunk carries its parent document index (used to emit
    DOC-XXXX IDs). Tried paragraph-aware chunking in `v5b-no-fallback` and
    it regressed cloud score by 0.027 — reverted in `v5c-no-para`.
  - Index every chunk in (a) BM25Okapi and (b) a dense BGE encoder.

Pipeline at query:
  - Hybrid score = z(BM25) + z(dense_cos). Take top-K passages.
  - Cross-encoder rerank the top-K to refine ordering.
  - documents: top-3 *unique* parent doc IDs in reranked order, then
    backfilled from the un-reranked hybrid list if the reranker concentrated
    on too few parent docs.
  - answer: batched extractive QA across the top reranked passages; return
    the highest-confidence span ≤64 tokens.

Doc IDs are derived in `_parse_doc_payload`. The official cloud format (as
of 14 May 2026) is `{"id": "DOC-XXXX", "document": "..."}` dicts; the
parser also handles a couple of plausible alternatives plus a positional
fallback.

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
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("NLP_MODEL_DIR", "/workspace/models"))
DENSE_DIR = MODEL_DIR / "bge-small-en-v1.5"
RERANKER_DIR = MODEL_DIR / "bge-reranker-base"
# QA model preference, in priority order:
#   1. Fine-tuned generative (seq2seq Flan-T5) — see training/nlp/finetune_genqa.py
#   2. Fine-tuned extractive (RoBERTa-SQuAD2) — see training/nlp/finetune_qa.py
#   3. Stock SQuAD2 weights downloaded by download_models.py
#   4. HF hub fallback (only if container is online)
QA_GEN_FINETUNED_DIR = MODEL_DIR / "flan-t5-finetuned"
QA_EXT_FINETUNED_DIR = MODEL_DIR / "roberta-finetuned-squad2"
QA_BASE_DIR = MODEL_DIR / "roberta-base-squad2"

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
_DOC_ID_RE = re.compile(r"\bDOC-(\d{4})\b")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "was", "were", "what", "when", "where", "which", "who",
        "why", "with",
    }
)
_PRINTABLE = set(string.printable)


def _doc_id_positional(idx0: int) -> str:
    """Positional fallback: 1-indexed, zero-padded to 4 digits."""
    return f"DOC-{idx0 + 1:04d}"


def _parse_doc_payload(doc, idx0: int) -> tuple[str, str]:
    """Return (id, text) for one entry of the corpus-load `documents` list.

    Tries (in order):
      1. Dict shape: {"id": "DOC-XXXX", "document": "..."} (or "text"/"content"/"body").
      2. First non-empty line is exactly `DOC-XXXX` -> strip it off the body.
      3. First 80 chars contain a `DOC-XXXX` token -> use it, leave body intact.
      4. Positional fallback DOC-{idx0+1:04d}.

    The cloud uses #1. The other branches are kept defensive — they cost
    nothing and saved us once already when the eval-server briefly sent the
    wrong format.
    """
    if isinstance(doc, dict):
        text = ""
        for k in ("text", "content", "document", "body"):
            if k in doc:
                text = doc[k]
                break
        did = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
        if did:
            return str(did), str(text)
        # fall through to text-based parsing on `text`
        doc = text

    text = str(doc)
    stripped = text.lstrip()

    # Case 2: first line is "DOC-XXXX" by itself.
    first_line_end = stripped.find("\n")
    first_line = stripped[:first_line_end] if first_line_end >= 0 else stripped
    flmatch = re.fullmatch(r"\s*DOC-(\d{4})\s*", first_line)
    if flmatch:
        body = stripped[first_line_end + 1 :] if first_line_end >= 0 else ""
        return f"DOC-{flmatch.group(1)}", body

    # Case 3: DOC-XXXX appears in the first 80 chars (header, title, etc.).
    head_match = _DOC_ID_RE.search(stripped[:80])
    if head_match:
        return f"DOC-{head_match.group(1)}", text

    # Case 4: positional fallback.
    return _doc_id_positional(idx0), text


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
        self._qa_is_generative = False

        # Corpus state.
        self.documents: list[str] = []
        self.doc_ids: list[str] = []  # i-th entry = ID for the i-th doc
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
        # QA model selection ladder. We detect seq2seq (generative) vs encoder
        # (extractive) from the saved config's is_encoder_decoder flag.
        if QA_GEN_FINETUNED_DIR.exists():
            qa_path = str(QA_GEN_FINETUNED_DIR)
            qa_source = "gen-finetuned"
        elif QA_EXT_FINETUNED_DIR.exists():
            qa_path = str(QA_EXT_FINETUNED_DIR)
            qa_source = "ext-finetuned"
        elif QA_BASE_DIR.exists():
            qa_path = str(QA_BASE_DIR)
            qa_source = "ext-base"
        else:
            qa_path = "deepset/roberta-base-squad2"
            qa_source = "ext-hub"
        qa_cfg = AutoConfig.from_pretrained(qa_path)
        self._qa_is_generative = bool(getattr(qa_cfg, "is_encoder_decoder", False))
        print(
            f"[nlp_manager] QA model: {qa_source} "
            f"({'generative' if self._qa_is_generative else 'extractive'}) "
            f"({qa_path})",
            flush=True,
        )

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
        if self._qa_is_generative:
            self._qa_model = (
                AutoModelForSeq2SeqLM.from_pretrained(qa_path)
                .to(self.device)
                .eval()
            )
        else:
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
        """Sentence-window chunking across the whole document.

        Same behaviour as v4-dict-id. Earlier paragraph-aware variant in
        `v5b-no-fallback` regressed cloud score -0.027 (likely because the
        held-out corpus has different paragraph structure than the local
        novice docs), so we are back on the v4 chunking.
        """
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

    def load_corpus(self, documents: list) -> None:
        self._init_models()

        self.documents = []
        self.doc_ids = []
        seen_ids: set[str] = set()
        for idx, raw in enumerate(documents):
            doc_id, text = _parse_doc_payload(raw, idx)
            # Guard against duplicate IDs: append a positional disambiguator.
            base_id = doc_id
            n = 1
            while doc_id in seen_ids:
                n += 1
                doc_id = f"{base_id}#{n}"
            seen_ids.add(doc_id)
            self.doc_ids.append(doc_id)
            self.documents.append(text)

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

    def _top_doc_ids(
        self,
        ranked_passage_idxs: Iterable[int],
        fallback: Iterable[int] | None = None,
    ) -> list[str]:
        """Collect up to TOP_DOCS_RETURNED unique parent doc IDs.

        Walks `ranked_passage_idxs` first (reranker output). If the reranker
        concentrated on too few parent docs to fill the top-3 slots, backfills
        from `fallback` (the un-reranked hybrid-retrieved list). This protects
        retrieval recall — the new eval gates every case on retrieval, so
        returning fewer than 3 unique doc IDs leaves credit on the table.
        """
        seen: list[int] = []
        for pidx in ranked_passage_idxs:
            d = self.passage_doc_idx[pidx]
            if d not in seen:
                seen.append(d)
                if len(seen) >= TOP_DOCS_RETURNED:
                    return [self.doc_ids[d] for d in seen]
        if fallback is not None:
            for pidx in fallback:
                d = self.passage_doc_idx[pidx]
                if d not in seen:
                    seen.append(d)
                    if len(seen) >= TOP_DOCS_RETURNED:
                        break
        return [self.doc_ids[d] for d in seen]

    # ------------------------------------------------------------ extraction

    @torch.no_grad()
    def _extract_answer(self, question: str, passage_idxs: list[int]) -> str:
        if self._qa_is_generative:
            return self._generate_answer(question, passage_idxs)
        return self._extract_answer_span(question, passage_idxs)

    @torch.no_grad()
    def _generate_answer(self, question: str, passage_idxs: list[int]) -> str:
        if not passage_idxs:
            return ""
        # For generative QA, condition on the top reranked chunk only — adding
        # multiple chunks risks the model paraphrasing across irrelevant context
        # and failing the AE 0.9 threshold. Keep it tight and verbatim-ish.
        top_chunk = self.passages[passage_idxs[0]]
        prompt = f"question: {question.strip()} context: {top_chunk.strip()}"
        enc = self._qa_tok(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out = self._qa_model.generate(
            **enc,
            max_new_tokens=QA_MAX_ANSWER_TOKENS,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        text = self._qa_tok.decode(out[0], skip_special_tokens=True).strip()
        return _clean_answer(text)

    @torch.no_grad()
    def _extract_answer_span(self, question: str, passage_idxs: list[int]) -> str:
        if not passage_idxs:
            return ""
        contexts = [self.passages[i] for i in passage_idxs[:TOP_K_RERANK]]

        # Tokenize all (question, context) pairs at once. Long contexts overflow
        # into multiple windows; `overflow_to_sample_mapping` tells us which
        # context each output feature came from.
        enc = self._qa_tok(
            [question] * len(contexts),
            contexts,
            max_length=QA_MAX_SEQ_LEN,
            truncation="only_second",
            stride=QA_DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        offset_mapping = enc.pop("offset_mapping")
        sample_mapping = enc.pop("overflow_to_sample_mapping")
        num_features = enc["input_ids"].shape[0]
        if num_features == 0:
            return ""

        # Batched forward over features. With QA_BATCH=16 and TOP_K_RERANK=10
        # contexts, most queries fit in a single forward pass — replacing the
        # previous N sequential forwards.
        start_logits_chunks: list[torch.Tensor] = []
        end_logits_chunks: list[torch.Tensor] = []
        tensor_inputs = {k: v for k, v in enc.items() if isinstance(v, torch.Tensor)}
        for s_idx in range(0, num_features, QA_BATCH):
            e_idx = min(s_idx + QA_BATCH, num_features)
            batch_inputs = {
                k: v[s_idx:e_idx].to(self.device) for k, v in tensor_inputs.items()
            }
            out = self._qa_model(**batch_inputs)
            start_logits_chunks.append(out.start_logits.float().cpu())
            end_logits_chunks.append(out.end_logits.float().cpu())
        start_logits = torch.cat(start_logits_chunks, dim=0)
        end_logits = torch.cat(end_logits_chunks, dim=0)

        best_text = ""
        best_score = -1e9
        for feat in range(num_features):
            ctx_idx = int(sample_mapping[feat])
            ctx = contexts[ctx_idx]
            seq_ids = enc.sequence_ids(feat)
            valid = torch.tensor(
                [1.0 if sid == 1 else 0.0 for sid in seq_ids], dtype=torch.float32
            )
            # Mask out non-context tokens with a large negative number so any
            # span chosen comes from the context, never the question.
            mask = (1.0 - valid) * -1e4
            s = start_logits[feat] + mask
            e = end_logits[feat] + mask

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
        documents = self._top_doc_ids(reranked, fallback=retrieved)
        answer = self._extract_answer(question, reranked)
        return {"documents": documents, "answer": answer}

    def qa_batch(self, questions: list[str]) -> list[dict]:
        return [self._answer_one(q) for q in questions]

    # Compatibility shim — old call site returns a bare string.
    def qa(self, question: str) -> dict:
        return self._answer_one(question)
