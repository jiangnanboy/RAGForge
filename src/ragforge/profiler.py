"""
Pipeline Profiler
==================
Lightweight latency profiling for pipeline steps.

Features:
    - Per-step timing with percentage breakdown
    - Integrated into ``SearchPipeline`` via ``profile=True``
    - Human-readable formatted output
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


class PipelineProfiler:
    """Records per-step latency for a pipeline execution.

    Usage inside pipeline::

        profiler = PipelineProfiler()
        with profiler.profile("BM25") as s:
            bm25_results = bm25.retrieve(query, docs)

        print(profiler.report)
    """

    def __init__(self) -> None:
        self._records: list[tuple[str, float]] = []

    @contextmanager
    def profile(self, name: str) -> Generator[_ProfileSlot, None, None]:
        """Context manager that times a code block.

        Args:
            name: Step name.

        Yields:
            A slot object (currently unused, reserved for future metadata).
        """
        start = time.perf_counter()
        yield _ProfileSlot()
        duration = (time.perf_counter() - start) * 1000
        self._records.append((name, duration))

    @property
    def total_ms(self) -> float:
        return sum(d for _, d in self._records)

    @property
    def report(self) -> str:
        """Formatted profiling report."""
        total = self.total_ms
        lines = [
            f"{'Step':<24}{'Time (ms)':>10}{'% Total':>8}",
            "-" * 44,
        ]
        for name, duration in self._records:
            pct = duration / total * 100 if total > 0 else 0
            lines.append(f"{name:<24}{duration:>10.1f}{pct:>7.1f}%")
        lines.append("-" * 44)
        lines.append(f"{'TOTAL':<24}{total:>10.1f}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._records.clear()


class _ProfileSlot:
    """Placeholder slot for profile context."""
    pass
