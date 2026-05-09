"""Baseline NLP RAG manager.

This lightweight baseline stores the corpus, does simple lexical retrieval, and
returns the most relevant sentence/span. It is not strong, but it is fast,
offline, deterministic, and schema-valid.
"""

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class NLPManager:
    """Simple lexical-retrieval QA baseline."""

    def __init__(self):
        self.loaded = False
        self.documents: list[str] = []
        self.passages: list[str] = []
        self.passage_tokens: list[Counter[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        return [
            tok.lower()
            for tok in _TOKEN_RE.findall(text)
            if tok.lower() not in _STOPWORDS and len(tok) > 1
        ]

    def _split_passages(self, document: str) -> list[str]:
        # Sentence chunks preserve concise answers. Fall back to paragraph/text if
        # punctuation is sparse.
        sentences = [s.strip() for s in _SENTENCE_RE.split(document) if s.strip()]
        if not sentences and document.strip():
            sentences = [document.strip()]
        return sentences

    def load_corpus(self, documents: list[str]) -> None:
        """Loads the corpus of documents for RAG QA."""
        self.documents = documents
        self.passages = []
        for doc in documents:
            self.passages.extend(self._split_passages(doc))

        self.passage_tokens = [Counter(self._tokenize(p)) for p in self.passages]
        self.loaded = True

    def qa(self, question: str) -> str:
        """Answer a question using lexical retrieval over the loaded corpus."""
        if not self.loaded or not self.passages:
            return ""

        q_tokens = Counter(self._tokenize(question))
        if not q_tokens:
            return self.passages[0][:300]

        best_idx = 0
        best_score = -1.0
        for idx, p_tokens in enumerate(self.passage_tokens):
            overlap = sum(min(count, p_tokens.get(tok, 0)) for tok, count in q_tokens.items())
            # Tiny length normalization so long passages don't always win.
            score = overlap / max(sum(p_tokens.values()) ** 0.5, 1.0)
            if score > best_score:
                best_score = score
                best_idx = idx

        answer = self.passages[best_idx].strip()
        return answer[:500]
