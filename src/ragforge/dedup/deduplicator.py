"""
Document Deduplicator
=====================
Identifies and removes near-duplicate documents using embedding similarity.

Features:
    - Embedding-based near-duplicate detection
    - Configurable similarity threshold
    - Returns deduplicated document list
    - Optionally returns duplicate clusters for inspection
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..protocols import Embedder


class Deduplicator:
    """Document deduplicator using embedding similarity.

    Args:
        embedder:
            Any ``Embedder`` for computing document embeddings.
        threshold:
            Cosine similarity threshold above which documents
            are considered duplicates (default 0.95).

    Example::

        from ragforge import Deduplicator, FastembedEmbedder, ModelConfig

        dedup = Deduplicator(
            embedder=FastembedEmbedder(ModelConfig()),
            threshold=0.95,
        )

        documents = ["iPhone 价格", "iPhone价格", "华为手机报价"]
        unique = dedup.deduplicate(documents)
        print(f"{len(documents)} → {len(unique)} unique")
    """

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.95,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold

    def deduplicate(
        self,
        documents: list[str],
    ) -> list[str]:
        """Remove near-duplicate documents.

        Keeps the first occurrence of each duplicate cluster.

        Args:
            documents: List of document strings.

        Returns:
            Deduplicated list preserving original order.
        """
        if not documents:
            return []

        embeddings = self._embedder.embed_batch(documents)
        emb_matrix = np.array(embeddings)
        sim_matrix = cosine_similarity(emb_matrix)

        keep: list[bool] = [True] * len(documents)
        for i in range(len(documents)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(documents)):
                if not keep[j]:
                    continue
                if sim_matrix[i][j] >= self._threshold:
                    keep[j] = False  # j is a duplicate of i

        return [doc for doc, k in zip(documents, keep) if k]

    def find_clusters(
        self,
        documents: list[str],
    ) -> list[list[int]]:
        """Find groups of near-duplicate document indices.

        Args:
            documents: List of document strings.

        Returns:
            List of clusters, where each cluster is a list of
            document indices. Singletons (non-duplicates) are
            not included in the output.
        """
        if not documents:
            return []

        embeddings = self._embedder.embed_batch(documents)
        emb_matrix = np.array(embeddings)
        sim_matrix = cosine_similarity(emb_matrix)

        visited: set[int] = set()
        clusters: list[list[int]] = []

        for i in range(len(documents)):
            if i in visited:
                continue
            cluster = [i]
            for j in range(i + 1, len(documents)):
                if j not in visited and sim_matrix[i][j] >= self._threshold:
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) > 1:
                clusters.append(sorted(cluster))
            visited.add(i)

        return clusters
