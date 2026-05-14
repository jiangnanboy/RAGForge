"""
Hybrid Retriever
================
Composes BM25 + Vector retrieval in parallel.

Features:
    - Runs BM25 and vector retrieval concurrently (thread pool)
    - Agnostic to the specific ``Embedder`` and ``Retriever`` used
    - Returns two independent ranked lists for downstream fusion
    - Each sub-retriever is independently replaceable
"""

from __future__ import annotations

import concurrent.futures

from ..protocols import Embedder, Retriever
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.vector import VectorRetriever


class HybridRetriever:
    """Parallel BM25 + Vector hybrid retriever.

    Args:
        embedder:
            Any ``Embedder`` for the vector retrieval branch.
        bm25_retriever:
            Optional BM25 retriever. Defaults to a standard
            ``BM25Retriever`` with jieba tokenization.
        vector_retriever:
            Optional vector retriever. Defaults to a
            ``VectorRetriever`` using *embedder*.

    Example::

        from RAGForge import HybridRetriever, FastembedEmbedder, ModelConfig

        embedder = FastembedEmbedder(ModelConfig(embedding_model_path="..."))
        retriever = HybridRetriever(embedder=embedder)
        bm25_results, vector_results = retriever.retrieve_dual("query", docs)
    """

    def __init__(
        self,
        embedder: Embedder,
        bm25_retriever: Retriever | None = None,
        vector_retriever: Retriever | None = None,
    ) -> None:
        self._bm25 = bm25_retriever or BM25Retriever()
        self._vector = vector_retriever or VectorRetriever(embedder)

    def retrieve(
        self, query: str, documents: list[str]
    ) -> list[tuple[str, int]]:
        """Fallback single-list retrieve (uses vector only).

        Prefer ``retrieve_dual()`` to get both ranked lists.
        """
        return self._vector.retrieve(query, documents)

    def retrieve_dual(
        self, query: str, documents: list[str]
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Run BM25 and vector retrieval in parallel.

        Args:
            query: Search query.
            documents: Candidate documents.

        Returns:
            A 2-tuple of ``(bm25_ranked, vector_ranked)`` where each
            is a list of ``(document, rank)`` pairs.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_bm25 = pool.submit(self._bm25.retrieve, query, documents)
            fut_vec = pool.submit(self._vector.retrieve, query, documents)
            return fut_bm25.result(), fut_vec.result()

    def retrieve_dual_with_scores(
        self, query: str, documents: list[str]
    ) -> tuple[list[tuple[str, int, float]], list[tuple[str, int, float]]]:
        """Run BM25 and vector retrieval in parallel, including raw scores.

        Each sub-retriever must implement ``retrieve_with_scores()``.

        Args:
            query: Search query.
            documents: Candidate documents.

        Returns:
            A 2-tuple of ``(bm25_scored, vector_scored)`` where each
            is a list of ``(document, rank, score)`` triples.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_bm25 = pool.submit(
                self._bm25.retrieve_with_scores, query, documents
            )
            fut_vec = pool.submit(
                self._vector.retrieve_with_scores, query, documents
            )
            return fut_bm25.result(), fut_vec.result()
