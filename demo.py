"""
RAGForge — Usage Examples
===============================
Demonstrates all features: retrieval, reranking, query understanding,
evaluation, adaptive fusion, semantic cache, deduplication, tracing, profiling.
"""
from ragforge.pipeline import SearchPipeline
from ragforge.models import FastembedEmbedder, FastembedReranker
from ragforge.retrieval import BM25Retriever, VectorRetriever, HybridRetriever
from ragforge.fusion import RRFFusion, PositionAwareBlend, AdaptiveFusion
from ragforge.llm import LLMClient
from ragforge.query import QueryPlanner
from ragforge.evaluation import LLMJudge, Evaluator
from ragforge.cache import SemanticCache
from ragforge.dedup import Deduplicator
from ragforge.config import ModelConfig, PipelineConfig, LLMConfig, QueryTransformStrategy

import os
from dotenv import load_dotenv

load_dotenv()

deepseek_api = os.environ['DEEPSEEK_API']
deepseek_url = os.environ['DEEPSEEK_URL']
deepseek_model = os.environ['DEEPSEEK_MODEL']

# ================================================================
# Example 1: Full Pipeline with Tracing
# ================================================================
def example_pipeline_with_tracing():
    """Full pipeline with detailed step-by-step tracing."""
    model_cfg = ModelConfig(
        embedding_model_path=None,
        rerank_model_path=None,
    )
    pipe_cfg = PipelineConfig()

    pipeline = SearchPipeline(
        embedder=FastembedEmbedder(model_cfg),
        reranker=FastembedReranker(model_cfg),
        retriever=HybridRetriever(embedder=FastembedEmbedder(model_cfg)),
        fusion=RRFFusion(pipe_cfg),
        blend=PositionAwareBlend(pipe_cfg),
    )

    query = "苹果手机多少钱"
    documents = ["iPhone 15售价多少", "苹果手机官方定价", "华为手机报价"]

    # trace=True returns (results, PipelineTrace)
    results, trace = pipeline.search(query, documents, trace=True)
    print(trace.formatted)
    print()
    print(trace.xray)  # Detailed X-ray: BM25 scores, Vector scores, RRF, Rerank, Blend


# ================================================================
# Example 2: Pipeline with Profiling
# ================================================================
def example_pipeline_with_profiling():
    """Full pipeline with latency profiling."""
    model_cfg = ModelConfig(
        embedding_model_path=None,
        rerank_model_path=None,
    )

    pipeline = SearchPipeline(
        embedder=FastembedEmbedder(model_cfg),
        reranker=FastembedReranker(model_cfg),
    )

    # profile=True returns (results, PipelineProfiler)
    results, profiler = pipeline.search("query", documents, profile=True)
    print(profiler.report)


# ================================================================
# Example 3: Query Understanding
# ================================================================
def example_query_understanding():
    """Query rewrite, decomposition, HyDE, and expansion."""
    llm_cfg = LLMConfig(api_key=deepseek_api)
    planner = QueryPlanner(llm_cfg)

    # Rewrite colloquial query
    rewritten = planner.rewrite("苹果手机多少钱")
    print(f"Rewritten: {rewritten}")

    # Decompose complex query
    sub_queries = planner.decompose("对比iPhone和华为的拍照效果")
    print(f"Sub-queries: {sub_queries}")

    # Generate hypothetical document
    hypothetical = planner.hyde("什么是RAG检索增强生成")
    print(f"HyDE: {hypothetical[:100]}...")

    # Expand with synonyms
    expansions = planner.expand("深度学习")
    print(f"Expansions: {expansions}")


# ================================================================
# Example 4: Pipeline with Query Transform
# ================================================================
def example_pipeline_with_query_transform():
    """Pipeline with automatic query rewriting."""
    llm_cfg = LLMConfig(api_key=deepseek_api)
    model_cfg = ModelConfig(embedding_model_path=None)

    pipeline = SearchPipeline(
        embedder=FastembedEmbedder(model_cfg),
        query_transform=QueryPlanner(llm_cfg),  # Auto-rewrite queries
    )

    results = pipeline.search("苹果手机多少钱", documents)
    print(results)


# ================================================================
# Example 4b: Multi-Query Retrieval (Original + Rewritten)
# ================================================================
def example_multi_query_retrieval():
    """Retrieve with BOTH original and rewritten queries, then fuse.

    This is the **Multi-Query Retrieval** pattern:
    1. Query Transform rewrites "苹果手机多少钱" → "苹果iPhone价格"
    2. Retrieval runs for BOTH queries in parallel
    3. RRF fuses the two result sets into one

    Benefit: higher recall — the original query catches keyword matches
    that the rewritten query might miss, and vice versa.
    """
    llm_cfg = LLMConfig(api_key=deepseek_api)
    model_cfg = ModelConfig(
        embedding_model_path=None,
        rerank_model_path=None,
    )

    # Enable RETRIEVE_AND_FUSE in pipeline config
    pipe_cfg = PipelineConfig(
        query_transform_strategy=QueryTransformStrategy.RETRIEVE_AND_FUSE,
    )

    pipeline = SearchPipeline(
        embedder=FastembedEmbedder(model_cfg),
        reranker=FastembedReranker(model_cfg),
        retriever=HybridRetriever(embedder=FastembedEmbedder(model_cfg)),
        fusion=RRFFusion(pipe_cfg),
        blend=PositionAwareBlend(pipe_cfg),
        query_transform=QueryPlanner(llm_cfg),
        config=pipe_cfg,
    )

    # Trace to see both retrieval paths
    results, trace = pipeline.search("苹果手机多少钱", documents, trace=True)
    print(trace.formatted)
    print(trace.xray)


    # Works with decompose too — original + all sub-queries
    # planner = QueryPlanner(llm_cfg)
    # pipeline_decompose = SearchPipeline(
    #     query_transform=_DecomposingTransform(planner),
    #     config=PipelineConfig(
    #         query_transform_strategy=QueryTransformStrategy.RETRIEVE_AND_FUSE,
    #     ),
    # )

    # "对比iPhone和华为拍照" → decompose into 2 sub-queries
    # → 3 retrievals (original + 2 sub-queries), all fused


class _DecomposingTransform:
    """Helper: wraps QueryPlanner to use decompose() instead of rewrite().

    Demonstrates that ANY callable satisfying the QueryTransform protocol
    can be used with RETRIEVE_AND_FUSE.
    """
    def __init__(self, planner):
        self._planner = planner

    def transform(self, query: str) -> list[str]:
        return self._planner.decompose(query)


# ================================================================
# Example 5: Evaluation (LLM-as-Judge)
# ================================================================
def example_evaluation():
    """Evaluate pipeline quality with LLM-as-Judge."""
    llm_cfg = LLMConfig(api_key=deepseek_api)
    model_cfg = ModelConfig(
        embedding_model_path=None,
        rerank_model_path=None,
    )

    judge = LLMJudge(llm_cfg)
    evaluator = Evaluator(judge=judge)

    pipeline = SearchPipeline(
        embedder=FastembedEmbedder(model_cfg),
        reranker=FastembedReranker(model_cfg),
    )

    queries = ["苹果手机多少钱", "Python怎么入门"]
    ground_truth = {
        "苹果手机多少钱": {"iPhone 15 官方售价 5999元起", "苹果官网定价查询"},
        "Python怎么入门": {"Python零基础入门教程", "Python最佳学习路线"},
    }

    metrics = evaluator.evaluate(pipeline, queries, ground_truth, top_k=5)
    print(f"NDCG@5: {metrics.ndcg:.3f}")
    print(f"Recall@5: {metrics.recall:.3f}")
    print(f"MRR: {metrics.mrr:.3f}")


# ================================================================
# Example 6: A/B Pipeline Comparison
# ================================================================
def example_ab_comparison():
    """Compare two pipeline configurations side by side."""
    llm_cfg = LLMConfig(api_key=deepseek_api)
    model_cfg = ModelConfig(
        embedding_model_path=None,
        rerank_model_path=None,
    )

    judge = LLMJudge(llm_cfg)
    evaluator = Evaluator(judge=judge)

    # Pipeline A: Hybrid + Rerank + Blend
    pipe_a = SearchPipeline(
        embedder=FastembedEmbedder(model_cfg),
        reranker=FastembedReranker(model_cfg),
    )

    # Pipeline B: BM25 Only
    pipe_b = SearchPipeline(retriever=BM25Retriever())

    queries = ["苹果手机多少钱", "Python怎么入门"]
    ground_truth = {"苹果手机多少钱": {"iPhone 15 官方售价 5999元起", "苹果官网定价查询"}, "Python怎么入门": {"Python零基础入门教程", "Python最佳学习路线"}}

    comparison = evaluator.compare(pipe_a, pipe_b, queries, ground_truth)
    print(comparison["report"])


# ================================================================
# Example 7: Adaptive Fusion (Auto-tuned Parameters)
# ================================================================
def example_adaptive_fusion():
    """Automatically learn optimal fusion parameters from feedback."""
    feedback = [
        ("苹果手机价格", {"iPhone 15 官方售价 5999元", "苹果官网定价"}),
        ("Python教程", {"Python入门指南", "Python最佳实践"}),
    ]

    # Learn optimal parameters
    fusion = AdaptiveFusion.from_feedback(feedback)
    print(f"Best config: rrf_k={fusion.best_config.rrf_k}, "
          f"weight={fusion.best_config.query_weight}")

    # Use as a normal FusionStrategy
    # pipeline = SearchPipeline(fusion=fusion)
    # results = pipeline.search("query", documents)


# ================================================================
# Example 8: Semantic Cache
# ================================================================
def example_semantic_cache():
    """Cache search results to avoid redundant computation."""
    model_cfg = ModelConfig(embedding_model_path=None)

    pipeline = SearchPipeline(
        embedder=FastembedEmbedder(model_cfg),
    )

    cache = SemanticCache(
        embedder=FastembedEmbedder(model_cfg),
        similarity_threshold=0.95,
    )

    query = "苹果手机多少钱"

    # First call: runs pipeline, caches result
    result1 = cache.get_or_search(
        query, pipeline.search, documents
    )

    # Similar query: returns cached result (0ms)
    result2 = cache.get_or_search(
        "苹果手机多少钱", pipeline.search, documents
    )

    print(cache.stats)  # {'entries': 1, 'hits': 1, 'misses': 0, ...}


# ================================================================
# Example 9: Document Deduplication
# ================================================================
def example_dedup():
    """Remove near-duplicate documents before retrieval."""
    model_cfg = ModelConfig(embedding_model_path=None)

    dedup = Deduplicator(
        embedder=FastembedEmbedder(model_cfg),
        threshold=0.95,
    )

    documents = [
        "iPhone 15 官方售价 5999元",
        "iPhone 15官方售价5999元",  # near-duplicate
        "苹果手机价格查询",
        "华为手机最新报价",
    ]

    unique = dedup.deduplicate(documents)
    print(f"{len(documents)} → {len(unique)} unique documents")

    # Find duplicate clusters
    clusters = dedup.find_clusters(documents)
    print(f"Found {len(clusters)} duplicate clusters")


# ================================================================
# Example 10: BM25 Only (Simplest Possible Usage)
# ================================================================
def example_bm25_only():
    """No models needed — pure keyword search."""
    pipeline = SearchPipeline(retriever=BM25Retriever())
    results = pipeline.search("苹果手机", ["iPhone 15 价格", "华为手机"])
    for doc, score in results:
        print(f"  {score:.4f} → {doc}")

# ================================================================
# Example 11: BM25 Only (Simplest Possible Usage)
# ================================================================
def example_bm25_only():
    """No models needed — pure keyword search."""
    pipeline = SearchPipeline(retriever=BM25Retriever())
    results = pipeline.search("苹果手机", ["iPhone 15 价格", "华为手机"])
    for doc, score in results:
        print(f"  {score:.4f} → {doc}")

# ================================================================
# Example 12: Vector-Only Retrieval (custom embedder)
# ================================================================
def example_vector_only():
    """
    Vector retrieval with any Embedder implementation.
    You could swap FastembedEmbedder for OpenAI, Cohere, etc.
    """
    embedder = FastembedEmbedder(
        ModelConfig(embedding_model_path=None)
    )
    retriever = VectorRetriever(embedder)

    results = retriever.retrieve("How much is an iPhone?", [
        "iPhone 15 Price", "Apple Phone Official Price",
        "Latest Huawei Phone Quote", "Xiaomi Phone Price Inquiry",
    ])
    for doc, rank in results:
        print(f"  Rank {rank}: {doc}")


# ================================================================
# Example 13: Hybrid Retrieval + RRF Fusion (no reranker)
# ================================================================
def example_hybrid_with_fusion():
    """
    Parallel BM25 + Vector search, fused with RRF — skip reranking.
    """
    embedder = FastembedEmbedder(ModelConfig(
        embedding_model_path=None
    ))
    hybrid = HybridRetriever(embedder=embedder)
    fusion = RRFFusion(PipelineConfig(rrf_k=60, top_k_recall=10))

    query = "苹果手机多少钱"
    documents = ["iPhone 15售价", "苹果官方定价", "华为手机报价"]

    bm25_results, vector_results = hybrid.retrieve_dual(query, documents)
    fused = fusion.fuse([bm25_results, vector_results])

    print("Fused results:")
    for doc, score in fused:
        print(f"  {score:.4f} → {doc}")


# ================================================================
# Example 14: Full Pipeline (one-liner assembly)
# ================================================================
def example_full_pipeline():
    """
    Complete pipeline: Hybrid Retrieve → RRF Fuse → Rerank → Blend.
    All components are injected — swap any part freely.
    """
    model_cfg = ModelConfig(
        embedding_model_path=None, # 上传路径
        rerank_model_path=None, # 上传路径
    )
    pipe_cfg = PipelineConfig()

    pipeline = SearchPipeline(
        # embedder=FastembedEmbedder(model_cfg),
        reranker=FastembedReranker(model_cfg),
        retriever=HybridRetriever(embedder=FastembedEmbedder(model_cfg)),
        fusion=RRFFusion(pipe_cfg),
        blend=PositionAwareBlend(pipe_cfg),
    )

    query = "苹果手机多少钱"
    documents = ["iPhone 15售价多少", "苹果手机官方定价", "华为手机报价"]

    results = pipeline.search(query, documents)
    for doc, score in results:
        print(f"  {score:.4f} → {doc}")


# ================================================================
# Example 15: Custom Retriever (implement the protocol)
# ================================================================
def example_custom_retriever():
    """
    Implement your own retriever by satisfying the Retriever protocol.
    No inheritance needed — just implement the ``retrieve()`` method.
    """

    class TFIDFRetriever:
        """A custom retriever using TF-IDF instead of BM25."""

        def retrieve(self, query: str, documents: list[str]) -> list[tuple[str, int]]:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)
            query_vec = vectorizer.transform([query])
            scores = (tfidf_matrix * query_vec.T).toarray().flatten()

            ranked = sorted(
                zip(documents, scores), key=lambda x: x[1], reverse=True
            )
            return [(doc, idx + 1) for idx, (doc, _) in enumerate(ranked)]

    pipeline = SearchPipeline(
        retriever=TFIDFRetriever(),
        fusion=RRFFusion(PipelineConfig()),
    )
    results = pipeline.search("query", ["doc1", "doc2"])
    print(results)

documents = [
    "iPhone 15售价多少", "苹果手机官方定价", "华为手机报价",
    "小米手机价格", "二手苹果手机值多少钱",
]

if __name__ == "__main__":
    print("Run individual examples:")
    # print("  example_pipeline_with_tracing()")
    example_pipeline_with_tracing()

    # print("  example_pipeline_with_profiling()")
    # example_pipeline_with_profiling()

    # print("  example_query_understanding()")
    # example_query_understanding()

    # print("  example_pipeline_with_query_transform()")
    # example_pipeline_with_query_transform()

    print("  example_multi_query_retrieval()  ← NEW: original + rewritten fusion")
    example_multi_query_retrieval()

    # print("  example_evaluation()")
    # example_evaluation()

    # print("  example_ab_comparison()")
    # example_ab_comparison()

    # print("  example_adaptive_fusion()")
    # example_adaptive_fusion()

    # print("  example_semantic_cache()")
    # example_semantic_cache()

    # print("  example_dedup()")
    # example_dedup()

    # print("  example_bm25_only()")
    # example_bm25_only()

    # print(" example_vector_only()")
    # example_vector_only()
