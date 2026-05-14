"""RAGForge Protocol Definitions
===================================
Define abstract interfaces for all pluggable components.
Any component can be replaced by implementing the corresponding protocol.

Design Principle:
    - Each protocol defines a minimal, focused interface
    - Protocols use structural subtyping (duck typing with type hints)
    - No inheritance required — just implement the methods
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding models.

    Any class that implements ``embed()`` and ``embed_batch()``
    satisfies this protocol and can be plugged into a VectorRetriever.
    """

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a dense vector.

        Args:
            text: Input text string.

        Returns:
            1-D numpy array of shape ``(dim,)``.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts.

        Args:
            texts: List of input text strings.

        Returns:
            List of 1-D numpy arrays.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """Protocol for document retrievers.

    Each retriever produces a ranked list of ``(document, rank)`` pairs
    where *rank* starts from 1 (best = 1).
    """

    def retrieve(
        self, query: str, documents: list[str]
    ) -> list[tuple[str, int]]:
        """Retrieve and rank documents for a given query.

        Args:
            query: Search query string.
            documents: Candidate documents to rank.

        Returns:
            List of ``(document_text, rank)`` sorted by relevance
            (rank 1 = most relevant).
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for cross-encoder reranking models.

    A reranker takes a query + candidate documents and returns
    fine-grained relevance scores.
    """

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Rerank documents given a query.

        Args:
            query: Search query string.
            documents: Candidate documents to rerank.

        Returns:
            List of raw logit scores, one per document,
            in the same order as *documents*.
        """
        ...


@runtime_checkable
class FusionStrategy(Protocol):
    """Protocol for multi-list result fusion.

    A fusion strategy merges multiple ranked lists (e.g., BM25 + vector)
    into a single ranked list with fused scores.
    """

    def fuse(
        self, ranked_lists: list[list[tuple[str, int]]]
    ) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists into a single scored ranking.

        Args:
            ranked_lists: Each element is a list of ``(document, rank)``
                pairs from one retriever.

        Returns:
            List of ``(document, fused_score)`` sorted descending by score.
        """
        ...


@runtime_checkable
class Judge(Protocol):
    """Protocol for relevance judges (LLM-as-Judge).

    A judge evaluates whether retrieved documents are relevant to a query.
    """

    def judge(
        self, query: str, documents: list[str]
    ) -> list[dict]:
        """Judge relevance of each document to the query.

        Args:
            query: Search query string.
            documents: Retrieved documents to judge.

        Returns:
            List of dicts, one per document, each containing:
            ``{"document": str, "relevant": bool, "score": float, "reason": str}``
        """
        ...


@runtime_checkable
class QueryTransform(Protocol):
    """Protocol for query transformation / understanding.

    A query transform takes a raw query and produces an improved version
    or multiple sub-queries for better retrieval.
    """

    def transform(self, query: str) -> str | list[str]:
        """Transform a query for better retrieval.

        Args:
            query: Raw user query.

        Returns:
            A single improved query string, or a list of sub-query strings.
        """
        ...
