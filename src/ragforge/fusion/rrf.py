"""
Reciprocal Rank Fusion (RRF)
=============================
A standalone ``FusionStrategy`` implementation.

Features:
    - Classic RRF scoring: ``1 / (k + rank + 1)``
    - Configurable RRF constant *k* and query weight
    - Top-position bonus for rank 1 and ranks 2-3
    - Fully independent — all parameters injected via ``PipelineConfig``
"""

from __future__ import annotations

from ..config import PipelineConfig


class RRFFusion:
    """Reciprocal Rank Fusion with top-position bonuses.

    Args:
        config: A ``PipelineConfig`` with RRF hyper-parameters.

    Example::

        from RAGForge import RRFFusion, PipelineConfig

        fusion = RRFFusion(PipelineConfig(rrf_k=60, query_weight=2.0))
        ranked_lists = [
            [("doc_a", 1), ("doc_b", 2)],  # BM25 results
            [("doc_b", 1), ("doc_c", 2)],  # Vector results
        ]
        fused = fusion.fuse(ranked_lists)
        # [("doc_b", 0.0234), ("doc_a", 0.0164), ("doc_c", 0.0081)]
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def fuse(
        self, ranked_lists: list[list[tuple[str, int]]]
    ) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists using RRF + top-position bonuses.

        Args:
            ranked_lists: Lists of ``(document, rank)`` from different
                retrievers.

        Returns:
            List of ``(document, fused_score)`` sorted descending.
        """
        k = self._config.rrf_k
        weight = self._config.query_weight
        bonus_r1 = self._config.bonus_rank1
        bonus_r2_3 = self._config.bonus_rank2_3

        # 1) Accumulate RRF scores
        doc_scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for doc, rank in ranked:
                rrf_score = 1 / (k + rank + 1) * weight
                doc_scores[doc] = doc_scores.get(doc, 0.0) + rrf_score

        # 2) Sort and apply top-position bonuses
        sorted_pairs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )
        for pos, (doc, _) in enumerate(sorted_pairs, start=1):
            if pos == 1:
                doc_scores[doc] += bonus_r1
            elif 2 <= pos <= 3:
                doc_scores[doc] += bonus_r2_3

        # 3) Return top-K
        fused = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_k = self._config.top_k_recall
        return fused[:top_k]

    def fuse_with_details(
        self, ranked_lists: list[list[tuple[str, int]]]
    ) -> tuple[list[tuple[str, float]], list[dict]]:
        """Fuse with per-document calculation details for X-ray tracing.

        Args:
            ranked_lists: Lists of ``(document, rank)`` from different retrievers.

        Returns:
            A 2-tuple of:
            - List of ``(document, fused_score)`` sorted descending.
            - List of detail dicts (one per document) for X-ray display,
              each containing: ``doc``, ``rrf_base``, ``bonus``,
              ``bonus_type``, ``final_score``, ``rank``.
        """
        k = self._config.rrf_k
        weight = self._config.query_weight
        bonus_r1 = self._config.bonus_rank1
        bonus_r2_3 = self._config.bonus_rank2_3

        # 1) Accumulate RRF base scores
        doc_scores: dict[str, float] = {}
        doc_details: dict[str, dict] = {}
        for ranked in ranked_lists:
            for doc, rank in ranked:
                rrf_score = 1 / (k + rank + 1) * weight
                doc_scores[doc] = doc_scores.get(doc, 0.0) + rrf_score

        # 2) Sort and apply top-position bonuses
        sorted_pairs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )
        for pos, (doc, _) in enumerate(sorted_pairs, start=1):
            bonus = 0.0
            bonus_type = ""
            if pos == 1:
                bonus = bonus_r1
                bonus_type = "bonus_rank1"
            elif 2 <= pos <= 3:
                bonus = bonus_r2_3
                bonus_type = "bonus_rank2_3"
            doc_scores[doc] += bonus
            doc_details[doc] = {
                "doc": doc,
                "rrf_base": round(doc_scores[doc] - bonus, 6),
                "bonus": round(bonus, 4),
                "bonus_type": bonus_type,
                "final_score": round(doc_scores[doc], 6),
                "rank": pos,
            }

        # 3) Return top-K
        fused = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_k = self._config.top_k_recall
        results = fused[:top_k]
        details_list = [doc_details[doc] for doc, _ in results if doc in doc_details]
        return results, details_list
