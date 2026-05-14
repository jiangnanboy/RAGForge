"""
Pipeline Tracer
================
Records detailed trace information for each pipeline step.

Features:
    - Per-step timing (milliseconds)
    - Input/output summaries
    - Structured detail dicts (scores, ranks, etc.)
    - Human-readable formatted output
    - Returns trace alongside results via ``SearchPipeline.search(trace=True)``
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from ..type_utils import StepTrace, PipelineTrace


class Tracer:
    """Records pipeline execution traces.

    Usage inside pipeline::

        tracer = Tracer()
        with tracer.step("BM25 Retrieval") as s:
            results = bm25.retrieve(query, docs)
            s.output_summary = f"{len(results)} docs ranked"
            s.details = {"top1": results[0][0] if results else None}
    """

    def __init__(self) -> None:
        self._steps: list[StepTrace] = []
        self._current: StepTrace | None = None

    @contextmanager
    def step(self, name: str) -> Generator[StepTrace, None, None]:
        """Context manager that times a pipeline step.

        Args:
            name: Human-readable step name.

        Yields:
            A ``StepTrace`` that can be annotated inside the ``with`` block.
        """
        step = StepTrace(name=name, duration_ms=0.0)
        self._current = step
        start = time.perf_counter()
        try:
            yield step
        finally:
            step.duration_ms = (time.perf_counter() - start) * 1000
            self._steps.append(step)
            self._current = None

    def build_trace(
        self,
        query: str,
        final_results: list[tuple[str, float]],
    ) -> PipelineTrace:
        """Build a complete ``PipelineTrace`` from recorded steps.

        Args:
            query: The original search query.
            final_results: Final ranked results.

        Returns:
            A ``PipelineTrace`` with all steps and timing.
        """
        total = sum(s.duration_ms for s in self._steps)
        trace = PipelineTrace(
            query=query,
            total_duration_ms=total,
            steps=list(self._steps),
            final_results=final_results,
        )
        self._steps.clear()
        return trace
