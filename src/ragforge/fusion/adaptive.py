"""
Adaptive Fusion
===============
Automatically learns optimal fusion parameters from feedback data.

Features:
    - Grid search over RRF k, query weight, bonus values
    - Position-aware blend weight optimization
    - Learns from (query, relevant_documents) pairs
    - Returns the best-performing ``PipelineConfig``
    - No ML framework dependency — pure Python
"""

from __future__ import annotations

from ..config import PipelineConfig
from ..protocols import FusionStrategy


class AdaptiveFusion(FusionStrategy):
    """Learns optimal fusion parameters from feedback.

    The learned parameters are stored in ``best_config`` and can be
    passed to ``RRFFusion`` and ``PositionAwareBlend``.

    Example::

        from ragforge import AdaptiveFusion

        # Provide training signal: queries with known relevant docs
        feedback = [
            ("苹果手机价格", {"iPhone 15 官方售价 5999元"}),
            ("Python教程", {"Python入门指南"}),
        ]

        fusion = AdaptiveFusion.from_feedback(feedback)
        print(fusion.best_config)
        # PipelineConfig(rrf_k=42, query_weight=1.8, ...)

        # Use it like a normal FusionStrategy
        fused = fusion.fuse([bm25_results, vector_results])
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self.best_config: PipelineConfig = PipelineConfig()

    def fuse(
        self, ranked_lists: list[list[tuple[str, int]]]
    ) -> list[tuple[str, float]]:
        """Fuse using current best-known parameters.

        Falls back to default RRF if ``from_feedback()`` has not
        been called yet.
        """
        k = self.best_config.rrf_k
        weight = self.best_config.query_weight
        bonus_r1 = self.best_config.bonus_rank1
        bonus_r2_3 = self.best_config.bonus_rank2_3

        doc_scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for doc, rank in ranked:
                rrf_score = 1 / (k + rank + 1) * weight
                doc_scores[doc] = doc_scores.get(doc, 0.0) + rrf_score

        sorted_pairs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        for pos, (doc, _) in enumerate(sorted_pairs, start=1):
            if pos == 1:
                doc_scores[doc] += bonus_r1
            elif 2 <= pos <= 3:
                doc_scores[doc] += bonus_r2_3

        fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return fused[: self.best_config.top_k_recall]

    @classmethod
    def from_feedback(
        cls,
        feedback: list[tuple[str, set[str] | list[str]]],
        ranked_lists_fn: object | None = None,
    ) -> AdaptiveFusion:
        """Learn optimal parameters from (query, relevant_docs) feedback.

        Args:
            feedback:
                List of ``(query, relevant_documents)`` tuples.
            ranked_lists_fn:
                Optional callable ``(query) -> list[list[tuple[str, int]]]``
                that returns ranked lists for the query (e.g., from your
                retriever). If ``None``, uses a simple simulation.

        Returns:
            An ``AdaptiveFusion`` instance with ``best_config`` set.
        """
        instance = cls()

        # Define search grid
        k_values = [30, 42, 60, 80, 100]
        weight_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        bonus_r1_values = [0.0, 0.03, 0.05, 0.08]
        bonus_r2_3_values = [0.0, 0.01, 0.02, 0.04]

        best_score = -1.0
        best_params: dict = {}

        for k in k_values:
            for w in weight_values:
                for br1 in bonus_r1_values:
                    for br23 in bonus_r2_3_values:
                        total_recall = 0.0
                        for query, relevant_set in feedback:
                            relevant_set = set(relevant_set)

                            if ranked_lists_fn is not None:
                                ranked_lists = ranked_lists_fn(query)
                            else:
                                # Simulation: create synthetic ranked lists
                                ranked_lists = cls._simulate_ranked(
                                    query, relevant_set
                                )

                            # Apply fusion with current params
                            doc_scores: dict[str, float] = {}
                            for ranked in ranked_lists:
                                for doc, rank in ranked:
                                    s = 1 / (k + rank + 1) * w
                                    doc_scores[doc] = doc_scores.get(doc, 0.0) + s

                            sorted_pairs = sorted(
                                doc_scores.items(), key=lambda x: x[1], reverse=True
                            )
                            for pos, (doc, _) in enumerate(sorted_pairs, start=1):
                                if pos == 1:
                                    doc_scores[doc] += br1
                                elif 2 <= pos <= 3:
                                    doc_scores[doc] += br23

                            final = sorted(
                                doc_scores.items(), key=lambda x: x[1], reverse=True
                            )[:10]

                            # Recall@10
                            found = sum(1 for d, _ in final if d in relevant_set)
                            recall = found / len(relevant_set) if relevant_set else 0
                            total_recall += recall

                        avg_recall = total_recall / len(feedback) if feedback else 0
                        if avg_recall > best_score:
                            best_score = avg_recall
                            best_params = {
                                "rrf_k": k,
                                "query_weight": w,
                                "bonus_rank1": br1,
                                "bonus_rank2_3": br23,
                            }

        instance.best_config = PipelineConfig(**best_params)
        return instance

    @staticmethod
    def _simulate_ranked(
        query: str, relevant_docs: set[str]
    ) -> list[list[tuple[str, int]]]:
        """Create synthetic ranked lists for simulation."""
        all_docs = list(relevant_docs) + [
            f"irrelevant document {i}" for i in range(5)
        ]
        # Simulate BM25: relevant docs tend to rank higher
        import random
        random.seed(hash(query))
        docs_copy = list(all_docs)
        random.shuffle(docs_copy)
        # Bias: put relevant docs earlier
        docs_copy.sort(key=lambda d: (0 if d in relevant_docs else 1, random.random()))
        list1 = [(d, i + 1) for i, d in enumerate(docs_copy)]
        # Simulate vector: different ordering
        random.shuffle(docs_copy)
        docs_copy.sort(key=lambda d: (0 if d in relevant_docs else 1, random.random()))
        list2 = [(d, i + 1) for i, d in enumerate(docs_copy)]
        return [list1, list2]
