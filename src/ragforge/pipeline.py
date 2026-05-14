"""
RAGForge Pipeline
======================
Composable search pipeline that ties together retriever, fusion, reranker,
tracing, caching, query understanding, and profiling.

Design Philosophy:
    - **Composition over inheritance**: pipeline assembles from components
    - **Each step is optional**: use only what you need
    - **No hidden global state**: everything is explicit
    - **Observable**: built-in tracing and profiling

Example — Full Pipeline with Tracing::

    from RAGForge import (
        SearchPipeline, FastembedEmbedder, FastembedReranker,
        HybridRetriever, RRFFusion, PositionAwareBlend,
        ModelConfig, PipelineConfig,
    )

    model_cfg = ModelConfig(
        embedding_model_path="/path/to/embedding",
        rerank_model_path="/path/to/reranker",
    )
    pipe_cfg = PipelineConfig()

    pipeline = SearchPipeline(
        embedder=FastembedEmbedder(model_cfg),
        reranker=FastembedReranker(model_cfg),
        fusion=RRFFusion(pipe_cfg),
        blend=PositionAwareBlend(pipe_cfg),
    )

    # With tracing
    result, trace = pipeline.search("query", documents, trace=True)
    print(trace.formatted)

    # With profiling
    result, profile = pipeline.search("query", documents, profile=True)
    print(profile.report)

Example — BM25 Only::

    from RAGForge import SearchPipeline, BM25Retriever

    pipeline = SearchPipeline(retriever=BM25Retriever())
    results = pipeline.search("query", documents)
"""

from __future__ import annotations

from typing import Any

from .config import PipelineConfig, QueryTransformStrategy
from .protocols import Embedder, Retriever, Reranker, FusionStrategy, QueryTransform
from .fusion import RRFFusion
from .retrieval import BM25Retriever
from .tracing.trace import Tracer
from .profiler import PipelineProfiler

from concurrent.futures import ThreadPoolExecutor


class SearchPipeline:
    """Composable search pipeline.

    Assemble from independent components — use only what you need.

    Args:
        embedder:
            Optional ``Embedder`` for vector retrieval and embeddings.
        retriever:
            Optional ``Retriever``. Defaults to ``BM25Retriever``.
        fusion:
            Optional ``FusionStrategy``.
            Defaults to ``RRFFusion`` with default config.
        reranker:
            Optional ``Reranker`` for cross-encoder reranking.
        blend:
            Optional ``PositionAwareBlend`` for final score blending.
        query_transform:
            Optional ``QueryTransform`` for query preprocessing
            (e.g., ``QueryPlanner``).
        config:
            Optional ``PipelineConfig`` for algorithm hyper-parameters.
            Controls fusion behavior, blend weights, and
            ``query_transform_strategy`` (REPLACE vs RETRIEVE_AND_FUSE).
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        retriever: Retriever | None = None,
        fusion: FusionStrategy | None = None,
        reranker: Reranker | None = None,
        blend: Any | None = None,
        query_transform: QueryTransform | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever or BM25Retriever()
        self._fusion = fusion or RRFFusion(config)
        self._reranker = reranker
        self._blend = blend
        self._query_transform = query_transform
        self._config = config or PipelineConfig()

    def search(
        self,
        query: str,
        documents: list[str],
        trace: bool = False,
        profile: bool = False,
    ) -> list[tuple[str, float]] | tuple[list[tuple[str, float]], Any]:
        """Execute the search pipeline.

        Pipeline steps (executed in order, optional steps are skipped):

        1. **Query Transform** — if ``query_transform`` is set
        2. **Retrieve** — BM25 + Vector (hybrid) or single retriever.
           If ``query_transform_strategy`` is ``RETRIEVE_AND_FUSE``,
           retrieval runs for BOTH the original query and the transformed
           query, then fuses results via RRF (Multi-Query Retrieval).
        3. **Fuse** — RRF fusion (if hybrid retrieval or multi-query)
        4. **Rerank** — Cross-encoder reranking (if reranker set)
        5. **Blend** — Position-aware score blending (if blend + reranker)

        Args:
            query: Search query.
            documents: Candidate documents.
            trace: If ``True``, returns ``(results, PipelineTrace)``.
            profile: If ``True``, returns ``(results, PipelineProfiler)``.

        Returns:
            If ``trace=False`` and ``profile=False``:
                List of ``(document, score)`` pairs.
            If ``trace=True``:
                Tuple of ``(results, PipelineTrace)``.
            If ``profile=True``:
                Tuple of ``(results, PipelineProfiler)``.
        """
        tracer = Tracer() if trace else None
        profiler = PipelineProfiler() if profile else None

        # Step 0: Query Transform (optional)
        effective_query = query
        transform_result = None
        if self._query_transform:
            if tracer:
                with tracer.step("Query Transform") as s:
                    transform_result = self._query_transform.transform(query)
                    s.input_summary = query[:60]
                    strategy = self._config.query_transform_strategy
                    s.details["strategy"] = strategy.value
                    if isinstance(transform_result, list):
                        effective_query = " | ".join(transform_result)
                        s.output_summary = (
                            f"Decomposed into {len(transform_result)} sub-queries "
                            f"(strategy: {strategy.value})"
                        )
                        s.details["sub_queries"] = transform_result
                    else:
                        effective_query = transform_result
                        s.output_summary = (
                            f"Rewritten: {transform_result[:60]} "
                            f"(strategy: {strategy.value})"
                        )
            else:
                transform_result = self._query_transform.transform(query)
                if isinstance(transform_result, list):
                    effective_query = " | ".join(transform_result)
                else:
                    effective_query = transform_result

        # Step 1: Retrieve
        strategy = self._config.query_transform_strategy
        use_multi_query = (
            transform_result is not None
            and strategy == QueryTransformStrategy.RETRIEVE_AND_FUSE
        )

        if use_multi_query:
            # Build query list: original + all transformed queries
            if isinstance(transform_result, list):
                queries_to_search = [query] + transform_result
            else:
                queries_to_search = [query, transform_result]

            if tracer:
                with tracer.step("Multi-Query Retrieval") as s:
                    fused = self._retrieve_multi(
                        queries_to_search, documents, tracer, profiler,
                    )
                    s.output_summary = (
                        f"{len(queries_to_search)} queries → "
                        f"{len(fused)} fused candidates"
                    )
                    s.details["queries"] = [
                        q[:40] for q in queries_to_search
                    ]
                    if fused:
                        s.details["top1"] = fused[0][0][:80]
            elif profiler:
                with profiler.profile("Multi-Query Retrieval"):
                    fused = self._retrieve_multi(
                        queries_to_search, documents, tracer=None, profiler=profiler,
                    )
            else:
                fused = self._retrieve_multi(
                    queries_to_search, documents, tracer=None, profiler=None,
                )
        else:
            if tracer:
                with tracer.step("Retrieval") as s:
                    fused = self._retrieve(
                        effective_query, documents, collect_xray=True,
                    )
                    s.output_summary = f"{len(fused)} candidates"
                    if fused:
                        s.details["top1"] = fused[0][0][:80]
                    # Populate X-ray data from last _retrieve call
                    if self._last_xray:
                        s.details["xray"] = self._last_xray
            elif profiler:
                with profiler.profile("Retrieval"):
                    fused = self._retrieve(effective_query, documents)
            else:
                fused = self._retrieve(effective_query, documents)

        # Step 2: Rerank (optional)
        rerank_norm = None
        if self._reranker:
            docs = [d[0] for d in fused]
            if tracer:
                with tracer.step("Rerank") as s:
                    rerank_norm, rel_levels, rerank_xray = self._do_rerank(
                        effective_query, docs, collect_xray=True,
                    )
                    s.output_summary = f"{len(docs)} docs reranked"
                    if rerank_norm and docs:
                        top_score = max(rerank_norm.values())
                        s.details["max_score"] = round(top_score, 4)
                    if rerank_xray:
                        s.details["xray"] = rerank_xray
            elif profiler:
                with profiler.profile("Rerank"):
                    rerank_norm, rel_levels = self._do_rerank(effective_query, docs)
            else:
                rerank_norm, rel_levels = self._do_rerank(effective_query, docs)

        # Step 3: Blend (optional)
        if self._blend and rerank_norm is not None:
            if tracer:
                with tracer.step("Blend") as s:
                    if hasattr(self._blend, "blend_with_details"):
                        final, blend_details = self._blend.blend_with_details(
                            fused, rerank_norm,
                        )
                        s.details["xray"] = {"blend": blend_details}
                    else:
                        final = self._blend.blend(fused, rerank_norm)
                    s.output_summary = f"{len(final)} results blended"
            elif profiler:
                with profiler.profile("Blend"):
                    final = self._blend.blend(fused, rerank_norm)
            else:
                final = self._blend.blend(fused, rerank_norm)
        else:
            final = fused

        # Build trace if requested
        if tracer:
            built_trace = tracer.build_trace(query, final)
            return final, built_trace

        if profiler:
            return final, profiler

        return final

    def _retrieve(
        self, query: str, documents: list[str],
        collect_xray: bool = False,
    ) -> list[tuple[str, float]]:
        """Internal: run retrieval + optional fusion.

        Args:
            query: Search query.
            documents: Candidate documents.
            collect_xray:
                If ``True``, populates ``self._last_xray`` with
                per-document score details for X-ray tracing.
        """
        self._last_xray: dict | None = None

        if hasattr(self._retriever, "retrieve_dual") and self._fusion:
            bm25_ranked, vector_ranked = self._retriever.retrieve_dual(
                query, documents
            )

            if collect_xray and hasattr(self._retriever, "retrieve_dual_with_scores"):
                bm25_scored, vector_scored = self._retriever.retrieve_dual_with_scores(
                    query, documents
                )

            # Fusion — with or without details
            if collect_xray and hasattr(self._fusion, "fuse_with_details"):
                fused, fusion_details = self._fusion.fuse_with_details(
                    [bm25_ranked, vector_ranked]
                )
            else:
                fused = self._fusion.fuse([bm25_ranked, vector_ranked])
                fusion_details = None

            # Build X-ray data
            if collect_xray and hasattr(self._retriever, "retrieve_dual_with_scores"):
                xray: dict = {"retrieval": {}}
                xray["retrieval"]["bm25"] = [
                    {"doc": doc, "score": round(score, 2), "rank": rank}
                    for doc, rank, score in bm25_scored
                ]
                xray["retrieval"]["vector"] = [
                    {"doc": doc, "score": round(score, 4), "rank": rank}
                    for doc, rank, score in vector_scored
                ]
                if fusion_details:
                    xray["fusion"] = fusion_details
                self._last_xray = xray
        else:
            single_ranked = self._retriever.retrieve(query, documents)
            fused = [(doc, 1.0 / rank) for doc, rank in single_ranked]

        return fused

    def _retrieve_multi(
        self,
        queries: list[str],
        documents: list[str],
        tracer: Any = None,
        profiler: Any = None,
    ) -> list[tuple[str, float]]:
        """Internal: retrieve with multiple queries and fuse results.

        Implements the **Multi-Query Retrieval** pattern:
        1. Run retrieval for each query (in parallel when possible)
        2. Re-rank each result set to ``(doc, rank)`` format
        3. Fuse all ranked lists using RRF

        Args:
            queries: List of query strings (original + transformed).
            documents: Candidate documents to search against.
            tracer: Optional Tracer for per-query tracing.
            profiler: Optional PipelineProfiler (already timing at outer level).

        Returns:
            Fused list of ``(document, score)`` pairs sorted descending.
        """
        # 1) Retrieve for each query — parallel via ThreadPoolExecutor
        per_query_results: list[list[tuple[str, float]]] = []

        def _run(q: str) -> list[tuple[str, float]]:
            if profiler:
                with profiler.profile(f"Retrieve: {q[:20]}"):
                    return self._retrieve(q, documents)
            return self._retrieve(q, documents)

        if len(queries) > 1:
            with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
                per_query_results = list(pool.map(_run, queries))
        else:
            per_query_results = [_run(queries[0])]

        # 2) Convert scored results → ranked lists for RRF fusion
        ranked_lists: list[list[tuple[str, int]]] = []
        for results in per_query_results:
            ranked = [
                (doc, rank) for rank, (doc, _score) in enumerate(results, 1)
            ]
            ranked_lists.append(ranked)

        # 3) Fuse all ranked lists
        if self._fusion:
            return self._fusion.fuse(ranked_lists)

        # Fallback: use RRF with default config
        fallback_fusion = RRFFusion(self._config)
        return fallback_fusion.fuse(ranked_lists)

    def _do_rerank(
        self, query: str, docs: list[str],
        collect_xray: bool = False,
    ) -> tuple:
        """Internal: run reranker if available.

        Args:
            query: Search query.
            docs: Documents to rerank.
            collect_xray:
                If ``True``, returns a 3-tuple with X-ray detail dict
                as the third element.

        Returns:
            If ``collect_xray=False``: ``(rerank_norm, rel_levels)``
            If ``collect_xray=True``: ``(rerank_norm, rel_levels, xray)``
        """
        if hasattr(self._reranker, "rerank_normalized"):
            rerank_norm, rel_levels = self._reranker.rerank_normalized(query, docs)

            if collect_xray and hasattr(self._reranker, "rerank"):
                raw_logits = self._reranker.rerank(query, docs)
                xray_list = []
                for doc, logit, norm_score, level in zip(
                    docs, raw_logits, rerank_norm.values(), rel_levels.values(),
                ):
                    xray_list.append({
                        "doc": doc,
                        "logit": round(logit, 2),
                        "sigmoid": round(norm_score, 4),
                        "level": level,
                    })
                return rerank_norm, rel_levels, {"rerank": xray_list}

            return rerank_norm, rel_levels
        return None, None
