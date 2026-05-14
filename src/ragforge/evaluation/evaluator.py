"""
RAG Evaluator
=============
Evaluation framework for comparing RAG pipeline configurations.

Features:
    - NDCG, Recall, Precision, MRR metrics
    - LLM-as-Judge based relevance assessment (no ground truth needed)
    - A/B comparison of two pipeline configurations
    - Latency profiling
    - Aggregate reporting
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..type_utils import EvalMetrics
from ..evaluation.judge import LLMJudge


def _dcg(relevance: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    scores = relevance[:k]
    if not scores:
        return 0.0
    return sum(s / np.log2(i + 2) for i, s in enumerate(scores))


def _ndcg(predicted: list[float], ideal: list[float], k: int) -> float:
    """Normalized DCG at k."""
    dcg_val = _dcg(predicted, k)
    idcg_val = _dcg(sorted(ideal, reverse=True), k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


@dataclass
class _QueryEval:
    """Internal: evaluation result for a single query."""
    query: str
    ndcg: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    latency_ms: float = 0.0
    judged_relevant: list[bool] = field(default_factory=list)


class Evaluator:
    """RAG pipeline evaluator.

    Args:
        judge: A ``LLMJudge`` instance for relevance assessment.

    Example::

        from ragforge import Evaluator, LLMJudge, LLMConfig

        judge = LLMJudge(LLMConfig(api_key="sk-..."))
        evaluator = Evaluator(judge=judge)

        queries = ["苹果手机多少钱", "Python怎么学"]
        relevant_docs = {
            "苹果手机多少钱": {"iPhone 15 价格 5999元", "苹果官网定价"},
            "Python怎么学": {"Python入门教程", "Python最佳实践"},
        }

        metrics = evaluator.evaluate(
            pipeline=my_pipeline,
            queries=queries,
            ground_truth=relevant_docs,
            top_k=5,
        )
        print(f"NDCG@5: {metrics.ndcg:.3f}")
    """

    def __init__(self, judge: LLMJudge) -> None:
        self._judge = judge

    def evaluate(
        self,
        pipeline: object,
        queries: list[str],
        ground_truth: dict[str, set[str]] | dict[str, list[str]],
        top_k: int = 5,
    ) -> EvalMetrics:
        """Evaluate a pipeline against ground truth.

        Args:
            pipeline: Any object with a ``search(query, documents)`` method.
            queries: List of test queries.
            ground_truth:
                Dict mapping query -> set/list of relevant document texts.
            top_k: Number of top results to consider.

        Returns:
            An ``EvalMetrics`` instance with aggregated scores.
        """
        evals: list[_QueryEval] = []

        for query in queries:
            relevant_set = set(ground_truth.get(query, []))

            # Run pipeline
            start = time.perf_counter()
            try:
                results = pipeline.search(query, list(relevant_set))
            except Exception:
                results = []
            latency = (time.perf_counter() - start) * 1000

            retrieved_docs = [doc for doc, _ in results[:top_k]]

            # Judge relevance via LLM
            judged = self._judge.judge(query, retrieved_docs)
            judge_scores = [j["score"] for j in judged]
            judge_binary = [j["relevant"] for j in judged]

            # Calculate metrics
            ideal_scores = [1.0] * len(relevant_set)

            # NDCG
            ndcg = _ndcg(judge_scores, ideal_scores, top_k)

            # Recall: fraction of ground truth found in top-k
            found = sum(1 for d in retrieved_docs if d in relevant_set)
            recall = found / len(relevant_set) if relevant_set else 0.0

            # Precision: fraction of retrieved that are relevant
            precision = (
                sum(1 for b in judge_binary if b) / len(judge_binary)
                if judge_binary else 0.0
            )

            # MRR
            mrr = 0.0
            for i, b in enumerate(judge_binary):
                if b:
                    mrr = 1.0 / (i + 1)
                    break

            evals.append(_QueryEval(
                query=query,
                ndcg=ndcg,
                recall=recall,
                precision=precision,
                mrr=mrr,
                latency_ms=latency,
            ))

        return EvalMetrics(
            ndcg=float(np.mean([e.ndcg for e in evals])),
            recall=float(np.mean([e.recall for e in evals])),
            precision=float(np.mean([e.precision for e in evals])),
            mrr=float(np.mean([e.mrr for e in evals])),
            avg_latency_ms=float(np.mean([e.latency_ms for e in evals])),
            num_queries=len(evals),
        )

    def compare(
        self,
        pipeline_a: object,
        pipeline_b: object,
        queries: list[str],
        ground_truth: dict[str, set[str]] | dict[str, list[str]],
        top_k: int = 5,
    ) -> dict:
        """Compare two pipelines side by side.

        Args:
            pipeline_a: First pipeline to evaluate.
            pipeline_b: Second pipeline to evaluate.
            queries: List of test queries.
            ground_truth: Dict mapping query -> relevant documents.
            top_k: Number of top results to consider.

        Returns:
            Dict with ``pipeline_a``, ``pipeline_b`` metrics
            and a formatted ``report`` string.
        """
        metrics_a = self.evaluate(pipeline_a, queries, ground_truth, top_k)
        metrics_b = self.evaluate(pipeline_b, queries, ground_truth, top_k)

        # Build comparison report
        lines = [
            "Pipeline Comparison Report",
            "=" * 70,
            f"{'Metric':<20}{'Pipeline A':>12}{'Pipeline B':>12}",
            "-" * 44,
            f"{'NDCG@' + str(top_k):<20}{metrics_a.ndcg:>12.4f}{metrics_b.ndcg:>12.4f}",
            f"{'Recall@' + str(top_k):<20}{metrics_a.recall:>12.4f}{metrics_b.recall:>12.4f}",
            f"{'Precision@' + str(top_k):<20}{metrics_a.precision:>12.4f}{metrics_b.precision:>12.4f}",
            f"{'MRR':<20}{metrics_a.mrr:>12.4f}{metrics_b.mrr:>12.4f}",
            f"{'Avg Latency (ms)':<20}{metrics_a.avg_latency_ms:>12.1f}{metrics_b.avg_latency_ms:>12.1f}",
            "-" * 44,
            f"{'Queries evaluated':<20}{metrics_a.num_queries:>12}{metrics_b.num_queries:>12}",
        ]

        return {
            "pipeline_a": metrics_a,
            "pipeline_b": metrics_b,
            "report": "\n".join(lines),
        }
