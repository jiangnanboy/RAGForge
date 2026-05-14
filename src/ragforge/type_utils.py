"""
RAGForge Shared Types
===========================
Lightweight data containers used across the library.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A single retrieval result with document text and relevance score."""

    document: str
    score: float
    rank: int

    @property
    def pair(self) -> tuple[str, int]:
        """Return as ``(document, rank)`` — compatible with fusion input."""
        return (self.document, self.rank)


@dataclass(slots=True)
class StepTrace:
    """Trace record for a single pipeline step."""

    name: str
    duration_ms: float
    input_summary: str = ""
    output_summary: str = ""
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"StepTrace(name={self.name!r}, duration_ms={self.duration_ms:.1f})"
        )


@dataclass(slots=True)
class PipelineTrace:
    """Complete trace of one pipeline execution."""

    query: str = ""
    total_duration_ms: float = 0.0
    steps: list[StepTrace] = field(default_factory=list)
    final_results: list[tuple[str, float]] = field(default_factory=list)

    def add_step(self, step: StepTrace) -> None:
        self.steps.append(step)

    @property
    def formatted(self) -> str:
        """Return a human-readable trace table."""
        lines = [
            f"Query: {self.query}",
            f"Total: {self.total_duration_ms:.1f}ms",
            "",
            f"{'Step':<22}{'Time':>8}  {'Output'}",
            "-" * 80,
        ]
        for s in self.steps:
            pct = s.duration_ms / self.total_duration_ms * 100 if self.total_duration_ms else 0
            lines.append(f"{s.name:<22}{s.duration_ms:>7.1f}ms  {s.output_summary}")
        if self.final_results:
            lines.append("")
            lines.append("Final Results:")
            for i, (doc, score) in enumerate(self.final_results, 1):
                lines.append(f"  Top{i}  {score:.4f}  {doc[:60]}{'...' if len(doc) > 60 else ''}")
        return "\n".join(lines)

    @property
    def xray(self) -> str:
        """Return detailed X-ray trace with per-document calculations.

        Shows internal scores for each pipeline step:
        - BM25 scores per document
        - Vector cosine similarity per document
        - RRF fusion calculation (base score + bonus → final)
        - Rerank logit → sigmoid → relevance level
        - Blend weight calculation (rrf_score × w_r + rerank_score × w_rr = final)
        """
        lines = [
            f"Query: {self.query}",
            f"Total: {self.total_duration_ms:.1f}ms",
            "",
        ]

        for s in self.steps:
            xray = s.details.get("xray")
            if not xray:
                # No detailed data — show summary only
                lines.append(f"┌─ {s.name} {'─' * max(1, 50 - len(s.name))}")
                lines.append(f"│  {s.output_summary}")
                lines.append(f"└{'─' * 60}")
                lines.append("")
                continue

            name = s.name
            separator = "─" * max(1, 50 - len(name))

            # Retrieval step: BM25 + Vector sub-sections
            if "retrieval" in xray:
                retrieval_data = xray["retrieval"]

                # BM25 sub-section
                if "bm25" in retrieval_data:
                    lines.append(f"┌─ BM25 Retrieval {separator}")
                    for item in retrieval_data["bm25"]:
                        doc = item["doc"]
                        score = item["score"]
                        rank = item["rank"]
                        lines.append(
                            f'│  "{doc:<28}" BM25={score:>7.1f}  rank={rank}'
                        )
                    lines.append(f"└{'─' * 60}")
                    lines.append("")

                # Vector sub-section
                if "vector" in retrieval_data:
                    lines.append(f"┌─ Vector Retrieval {separator}")
                    for item in retrieval_data["vector"]:
                        doc = item["doc"]
                        score = item["score"]
                        rank = item["rank"]
                        lines.append(
                            f'│  "{doc:<28}" cos_sim={score:.4f}  rank={rank}'
                        )
                    lines.append(f"└{'─' * 60}")
                    lines.append("")

            # RRF Fusion details
            if "fusion" in xray:
                lines.append(f"┌─ RRF Fusion {separator}")
                for item in xray["fusion"]:
                    doc = item["doc"]
                    rrf_base = item["rrf_base"]
                    bonus = item["bonus"]
                    bonus_type = item["bonus_type"]
                    final = item["final_score"]
                    if bonus > 0:
                        lines.append(
                            f'│  "{doc:<28}" rrf={rrf_base:.4f}  '
                            f'+{bonus_type}={bonus:.2f}  → {final:.4f}'
                        )
                    else:
                        lines.append(
                            f'│  "{doc:<28}" rrf={rrf_base:.4f}  → {final:.4f}'
                        )
                lines.append(f"└{'─' * 60}")
                lines.append("")

            # Rerank details
            if "rerank" in xray:
                lines.append(f"┌─ Rerank {separator}")
                for item in xray["rerank"]:
                    doc = item["doc"]
                    logit = item["logit"]
                    sigmoid_val = item["sigmoid"]
                    level = item["level"]
                    lines.append(
                        f'│  "{doc:<28}" logit={logit:>6.1f}  '
                        f'sigmoid={sigmoid_val:.3f}  "{level}"'
                    )
                lines.append(f"└{'─' * 60}")
                lines.append("")

            # Blend details
            if "blend" in xray:
                lines.append(f"┌─ Blend {separator}")
                for item in xray["blend"]:
                    doc = item["doc"]
                    rrf_s = item["rrf_score"]
                    rr_s = item["rerank_score"]
                    w_r = item["w_retrieval"]
                    w_rr = item["w_reranker"]
                    final = item["final_score"]
                    lines.append(
                        f'│  "{doc:<28}" '
                        f'{rrf_s:.4f}×{w_r} + {rr_s:.3f}×{w_rr} = {final:.4f}'
                    )
                lines.append(f"└{'─' * 60}")
                lines.append("")

        # Final Results
        if self.final_results:
            lines.append(f"{'═' * 60}")
            lines.append("  FINAL RESULTS:")
            for i, (doc, score) in enumerate(self.final_results, 1):
                marker = f"← FINAL #{i}"
                lines.append(f"  #{i:<3} {score:.4f}  {doc[:50]}  {marker}")

        return "\n".join(lines)


@dataclass(slots=True)
class JudgeResult:
    """Result of judging a single document."""

    document: str
    relevant: bool
    score: float
    reason: str


@dataclass(slots=True)
class EvalMetrics:
    """Aggregated evaluation metrics."""

    ndcg: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    num_queries: int = 0
