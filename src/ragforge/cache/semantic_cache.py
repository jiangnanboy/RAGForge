"""
Semantic Cache
==============
Cache search results keyed by query embedding similarity.

Features:
    - Embeds queries and finds cached entries by cosine similarity
    - Configurable similarity threshold
    - TTL-based expiration
    - Thread-safe (basic)
    - Uses any ``Embedder`` implementation
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..protocols import Embedder

class _CacheEntry:
    """Internal cache entry."""

    __slots__ = ("query", "query_embedding", "results", "timestamp")

    def __init__(
        self,
        query: str,
        query_embedding: np.ndarray,
        results: list[tuple[str, float]],
    ) -> None:
        self.query = query
        self.query_embedding = query_embedding
        self.results = results
        self.timestamp = time.time()


class SemanticCache:
    """Semantic cache for RAG search results.

    Caches pipeline results keyed by query embedding. When a new query
    is similar enough to a cached one, returns the cached results instead
    of re-running the pipeline.

    Args:
        embedder:
            Any ``Embedder`` for computing query embeddings.
        similarity_threshold:
            Minimum cosine similarity to consider a cache hit (default 0.95).
        ttl_seconds:
            Time-to-live for cache entries in seconds (default 3600).
            Set to 0 for no expiration.
        max_size:
            Maximum number of entries to keep in cache (default 1000).

    Example::

        from ragforge import SemanticCache, FastembedEmbedder, ModelConfig

        cache = SemanticCache(
            embedder=FastembedEmbedder(ModelConfig()),
            similarity_threshold=0.95,
        )

        # First call: runs pipeline, stores result
        result = cache.get_or_search(
            "苹果手机价格",
            search_fn=lambda q, d: pipeline.search(q, d),
            documents=docs,
        )

        # Similar query: returns cached result instantly
        result = cache.get_or_search(
            "苹果手机多少钱",
            search_fn=lambda q, d: pipeline.search(q, d),
            documents=docs,
        )
    """

    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.95,
        ttl_seconds: float = 3600,
        max_size: int = 1000,
    ) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._entries: list[_CacheEntry] = []
        self._hits = 0
        self._misses = 0

    def get(
        self, query: str
    ) -> list[tuple[str, float]] | None:
        """Look up a cached result for the query.

        Args:
            query: Search query.

        Returns:
            Cached results if a hit is found (similarity > threshold),
            ``None`` otherwise.
        """
        now = time.time()
        query_emb = self._embedder.embed(query).reshape(1, -1)

        for entry in self._entries:
            # Check TTL
            if self._ttl > 0 and (now - entry.timestamp) > self._ttl:
                continue
            # Check similarity
            sim = cosine_similarity(query_emb, entry.query_embedding.reshape(1, -1))[0][0]
            if sim >= self._threshold:
                self._hits += 1
                return entry.results

        self._misses += 1
        return None

    def put(
        self,
        query: str,
        results: list[tuple[str, float]],
    ) -> None:
        """Store a search result in the cache.

        Args:
            query: The query that produced these results.
            results: Pipeline output to cache.
        """
        query_emb = self._embedder.embed(query)

        # Evict oldest if at capacity
        if len(self._entries) >= self._max_size:
            self._entries.pop(0)

        self._entries.append(_CacheEntry(query, query_emb, results))

    def get_or_search(
        self,
        query: str,
        search_fn: object,
        documents: list[str],
    ) -> list[tuple[str, float]]:
        """Get cached result or execute search function.

        Args:
            query: Search query.
            search_fn:
                Callable ``search_fn(query, documents) -> results``.
            documents: Candidate documents.

        Returns:
            Either cached or fresh search results.
        """
        cached = self.get(query)
        if cached is not None:
            return cached

        results = search_fn(query, documents)
        self.put(query, results)
        return results

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        count = len(self._entries)
        self._entries.clear()
        self._hits = 0
        self._misses = 0
        return count

    @property
    def stats(self) -> dict[str, int | float]:
        """Cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }
