"""
Position-Aware Blend Fusion
============================
Blends RRF retrieval scores with reranker scores using
position-dependent weights.

Features:
    - Different weight ratios for top-1-3, top-4-10, and top-11+ ranks
    - Top positions trust retrieval more; lower positions trust reranker
    - All weights configurable via ``PipelineConfig``
    - Fully independent composable component
"""

from __future__ import annotations

from ..config import PipelineConfig


class PositionAwareBlend:
    """Position-aware weighted blend of retrieval + reranker scores.

    Args:
        config: A ``PipelineConfig`` with ``blend_weights``.

    Example::

        from RAGForge import PositionAwareBlend, PipelineConfig

        blend = PositionAwareBlend(PipelineConfig())
        rrf_results = [("doc_a", 0.05), ("doc_b", 0.03)]
        rerank_scores = {"doc_a": 0.92, "doc_b": 0.78}
        blended = blend.blend(rrf_results, rerank_scores)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def blend(
        self,
        rrf_results: list[tuple[str, float]],
        rerank_scores: dict[str, float],
    ) -> list[tuple[str, float]]:
        """Blend RRF and reranker scores with position-aware weights.

        Args:
            rrf_results: List of ``(document, rrf_score)`` from fusion.
            rerank_scores: Dict of ``{document: reranker_score}``.

        Returns:
            List of ``(document, blended_score)`` sorted descending.
        """
        weights = self._config.blend_weights
        blended: list[tuple[str, float]] = []

        for pos, (doc, rrf_score) in enumerate(rrf_results, start=1):
            rr_score = rerank_scores.get(doc, 0.0)

            # Select weight bucket based on position
            if 1 <= pos <= 3:
                w = weights["top1-3"]
            elif 4 <= pos <= 10:
                w = weights["top4-10"]
            else:
                w = weights["top11+"]

            final_score = rrf_score * w["retrieval"] + rr_score * w["reranker"]
            blended.append((doc, final_score))

        return sorted(blended, key=lambda x: x[1], reverse=True)

    def blend_with_details(
        self,
        rrf_results: list[tuple[str, float]],
        rerank_scores: dict[str, float],
    ) -> tuple[list[tuple[str, float]], list[dict]]:
        """Blend with per-document calculation details for X-ray tracing.

        Args:
            rrf_results: List of ``(document, rrf_score)`` from fusion.
            rerank_scores: Dict of ``{document: reranker_score}``.

        Returns:
            A 2-tuple of:
            - List of ``(document, blended_score)`` sorted descending.
            - List of detail dicts (one per document) for X-ray display,
              each containing: ``doc``, ``rrf_score``, ``rerank_score``,
              ``w_retrieval``, ``w_reranker``, ``bucket``, ``final_score``.
        """
        weights = self._config.blend_weights
        blended: list[tuple[str, float]] = []
        details_list: list[dict] = []

        for pos, (doc, rrf_score) in enumerate(rrf_results, start=1):
            rr_score = rerank_scores.get(doc, 0.0)

            if 1 <= pos <= 3:
                w = weights["top1-3"]
                bucket = "top1-3"
            elif 4 <= pos <= 10:
                w = weights["top4-10"]
                bucket = "top4-10"
            else:
                w = weights["top11+"]
                bucket = "top11+"

            final_score = rrf_score * w["retrieval"] + rr_score * w["reranker"]
            blended.append((doc, final_score))
            details_list.append({
                "doc": doc,
                "rrf_score": round(rrf_score, 6),
                "rerank_score": round(rr_score, 4),
                "w_retrieval": w["retrieval"],
                "w_reranker": w["reranker"],
                "bucket": bucket,
                "final_score": round(final_score, 6),
                "rank": pos,
            })

        return sorted(blended, key=lambda x: x[1], reverse=True), details_list

    def fuse(
        self, ranked_lists: list[list[tuple[str, int]]]
    ) -> list[tuple[str, float]]:
        """FusionStrategy protocol — delegates to ``blend()``.

        Note: This method is a compatibility shim. For blending with
        reranker scores, prefer calling ``blend()`` directly.
        """
        # Fallback: treat single list as RRF results, no reranker scores
        if ranked_lists:
            return [
                (doc, 1.0 / (rank + 1))
                for doc, rank in sorted(ranked_lists[0], key=lambda x: x[1])
            ]
        return []
