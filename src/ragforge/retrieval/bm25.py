"""
BM25 Full-Text Retriever
=========================
A standalone ``Retriever`` implementation using BM25 (Okapi BM25).

Features:
    - Pure BM25 full-text search with jieba tokenization
    - No external model dependency (jieba is the only requirement)
    - Satisfies the ``Retriever`` protocol
    - Fully independent — inject whatever tokenization you want
"""

from __future__ import annotations

from typing import Callable

import jieba
from rank_bm25 import BM25Okapi

# Default tokenizer: jieba Chinese word segmentation
_default_tokenizer: Callable[[str], list[str]] = lambda text: list(
    jieba.cut(text.strip())
)


class BM25Retriever:
    """BM25 full-text retriever with jieba tokenization.

    Args:
        tokenizer:
            Optional custom tokenizer function ``str -> list[str]``.
            Defaults to jieba-based tokenization.

    Example::

        from RAGForge import BM25Retriever

        retriever = BM25Retriever()
        results = retriever.retrieve("苹果手机", ["iPhone 15 价格", "华为手机"])
        # [(document, rank), ...]
    """

    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._tokenizer = tokenizer or _default_tokenizer

    def retrieve(
        self, query: str, documents: list[str]
    ) -> list[tuple[str, int]]:
        """Rank documents by BM25 score.

        Args:
            query: Search query.
            documents: Candidate documents.

        Returns:
            List of ``(document, rank)`` pairs sorted by descending BM25 score.
        """
        tokenized_docs = [self._tokenizer(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = self._tokenizer(query)
        scores = bm25.get_scores(tokenized_query)

        # Sort by absolute BM25 score descending
        ranked = sorted(
            zip(documents, scores), key=lambda x: abs(x[1]), reverse=True
        )
        return [(doc, idx + 1) for idx, (doc, _) in enumerate(ranked)]

    def retrieve_with_scores(
        self, query: str, documents: list[str]
    ) -> list[tuple[str, int, float]]:
        """Rank documents by BM25 score, including raw scores.

        Args:
            query: Search query.
            documents: Candidate documents.

        Returns:
            List of ``(document, rank, bm25_score)`` triples sorted
            by descending BM25 score.
        """
        tokenized_docs = [self._tokenizer(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = self._tokenizer(query)
        scores = bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(documents, scores), key=lambda x: abs(x[1]), reverse=True
        )
        return [(doc, idx + 1, score) for idx, (doc, score) in enumerate(ranked)]
