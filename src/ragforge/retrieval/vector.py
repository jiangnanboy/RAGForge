"""
Vector Semantic Retriever
==========================
A standalone ``Retriever`` implementation using cosine similarity.

Features:
    - Dense-vector cosine similarity search
    - Accepts any ``Embedder`` implementation (dependency injection)
    - Supports pre-computed document embeddings for efficiency
    - Satisfies the ``Retriever`` protocol
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..protocols import Embedder


class VectorRetriever:
    """Cosine-similarity vector retriever.

    Args:
        embedder:
            Any object satisfying the ``Embedder`` protocol
            (e.g., ``FastembedEmbedder``).

    Example::

        from RAGForge import VectorRetriever, FastembedEmbedder, ModelConfig

        embedder = FastembedEmbedder(ModelConfig(embedding_model_path="..."))
        retriever = VectorRetriever(embedder)
        results = retriever.retrieve("query", ["doc1", "doc2"])
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        documents: list[str],
        doc_embeddings: list[np.ndarray] | None = None,
    ) -> list[tuple[str, int]]:
        """Rank documents by cosine similarity to the query embedding.

        Args:
            query: Search query.
            documents: Candidate documents.
            doc_embeddings:
                Pre-computed embeddings for *documents*.
                If ``None``, embeddings are computed on-the-fly.

        Returns:
            List of ``(document, rank)`` pairs sorted by descending
            cosine similarity.
        """
        query_emb = self._embedder.embed(query).reshape(1, -1)

        if doc_embeddings is None:
            doc_embeddings = self._embedder.embed_batch(documents)

        doc_emb_matrix = np.array(doc_embeddings)
        sim_scores = cosine_similarity(query_emb, doc_emb_matrix)[0]

        ranked = sorted(
            zip(documents, sim_scores), key=lambda x: x[1], reverse=True
        )
        return [(doc, idx + 1) for idx, (doc, _) in enumerate(ranked)]

    def retrieve_with_scores(
        self,
        query: str,
        documents: list[str],
        doc_embeddings: list[np.ndarray] | None = None,
    ) -> list[tuple[str, int, float]]:
        """Rank documents by cosine similarity, including raw scores.

        Args:
            query: Search query.
            documents: Candidate documents.
            doc_embeddings:
                Pre-computed embeddings for *documents*.

        Returns:
            List of ``(document, rank, cosine_similarity)`` triples
            sorted by descending similarity.
        """
        query_emb = self._embedder.embed(query).reshape(1, -1)

        if doc_embeddings is None:
            doc_embeddings = self._embedder.embed_batch(documents)

        doc_emb_matrix = np.array(doc_embeddings)
        sim_scores = cosine_similarity(query_emb, doc_emb_matrix)[0]

        ranked = sorted(
            zip(documents, sim_scores), key=lambda x: x[1], reverse=True
        )
        return [(doc, idx + 1, score) for idx, (doc, score) in enumerate(ranked)]
