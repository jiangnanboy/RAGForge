<div align="center">
  <img src="logo.png" alt="Image" width="80%" />
</div>

<h1 align="center">RAGForge</h1>

<div align="center">

<p align="center">
  <strong>模块化、可组合的 RAG 检索库 —— 像锻造零件一样构建你的检索管道</strong>
</p>

README：[中文](README.md) | [English](README_EN.md)

</div>

<p align="center">
  <strong>纯 CPU，一个库搞定 RAG 检索管道。</strong>
</p>

<p align="center">
  Embedding 和 Rerank 全部 ONNX CPU 推理，无需 GPU。<br>
  查询理解可选接入 DeepSeek API，后续将支持本地小模型。<br>
  <strong>6 个 Protocol 接口 · 10+ 可组合组件 · X-Ray 级追踪</strong>
</p>

<p align="center">
  <a href="#为什么选-ragforge">为什么选</a> ·
  <a href="#30-秒上手">快速上手</a> ·
  <a href="#x-ray-追踪">X-Ray</a> ·
  <a href="#功能全景">功能</a> ·
  <a href="#architecture">架构</a> ·
  <a href="#installation">安装</a>
</p>

---

## 💡 为什么选 RAGForge

做一个 RAG 检索，你需要什么？

- 一个 Embedding 模型把文本变成向量
- 一个 Reranker 对候选文档精排
- BM25 + 向量的混合检索
- RRF 融合两路结果
- Query 改写、分解、HyDE 等查询理解
- 一套评估体系来调参

LangChain 和 LlamaIndex 都能做，但它们是**全栈框架**——你引入它们，就引入了几百个你可能永远用不到的依赖。更关键的是，它们都是**黑盒**：你不知道 BM25 给了每个文档多少分，不知道 RRF 融合后排名为什么变了，不知道 Rerank 的 logit 怎么映射到最终分数。

**RAGForge 的定位：只做检索管道这一段，做到极致，而且完全透明。**

| | LangChain | LlamaIndex | **RAGForge** |
|---|:---:|:---:|:---:|
| **定位** | 全栈 LLM 框架 | 全栈 RAG 框架 | **专注检索管道** |
| **CPU 推理** | 依赖外部库 | 依赖外部库 | ✅ **Embedding + Rerank 全 CPU** |
| **GPU 依赖** | 常需要 | 常需要 | ✅ **完全不需要** |
| **混合检索** | ✅ 但黑盒 | ✅ 但黑盒 | ✅ 每步可追踪 |
| **RRF 融合** | 简单实现 | 未内置 | ✅ Bonus + 自适应 |
| **X-Ray 调试** | ❌ | ❌ | ✅ **独有** |
| **Query 理解** | 基础 | 基础 | ✅ Multi-Query + HyDE |
| **LLM 评估** | 通用 | 通用 | ✅ 专为检索设计 |
| **组件可替换** | 受限 | 受限 | ✅ Protocol 接口 |
| **向量库绑定** | ❌ | ❌ | ✅ 不关心 |

> **一句话**：如果你只需要 BM25，用 3 行代码；如果你需要 Hybrid + Rerank + Multi-Query + X-Ray + A/B 评估，也能一行不漏。全看你组装什么。**而且全程 CPU，不需要一块 GPU。**

---

## ⚡ 30 秒上手

安装：pip install ragforge-sdk

### 最简：纯关键词搜索

```python
from ragforge import SearchPipeline, BM25Retriever

pipeline = SearchPipeline(retriever=BM25Retriever())
results = pipeline.search("苹果手机", ["iPhone 15 价格", "华为手机"])
for doc, score in results:
    print(f"  {score:.4f} → {doc}")
```

不需要 Embedding，不需要向量库，不需要 GPU。`jieba` + `rank_bm25` 就够了。

### 完整：Hybrid + Rerank + Blend + X-Ray

```python
from ragforge import (
    SearchPipeline, FastembedEmbedder, FastembedReranker,
    ModelConfig, PipelineConfig,
)

model_cfg = ModelConfig(
    embedding_model_path="/path/to/multilingual-e5-small-onnx",
    rerank_model_path="/path/to/bge-reranker-v2-m3-ONNX",
)

pipeline = SearchPipeline(
    embedder=FastembedEmbedder(model_cfg),
    reranker=FastembedReranker(model_cfg),
)

query = "苹果手机多少钱"
documents = ["iPhone 15售价多少", "苹果手机官方定价", "华为手机报价"]

results, trace = pipeline.search(query, documents, trace=True)
print(trace.formatted)
```

> **全部 CPU 推理**。Embedding 模型 ~400MB，Reranker 模型 ~500MB（int8 量化），首次自动下载，之后缓存到 `~/.cache/RAGForge/`。不需要 CUDA，不需要 GPU 驱动。

### 组件独立使用

```python
from ragforge import BM25Retriever, VectorRetriever, FastembedEmbedder

# 独立的 BM25
bm25 = BM25Retriever()
results = bm25.retrieve("查询", ["文档1", "文档2"])

# 独立的向量检索
embedder = FastembedEmbedder(ModelConfig())
vector = VectorRetriever(embedder)
results = vector.retrieve("查询", ["文档1", "文档2"])
```

---

## 🔬 X-Ray 追踪

这是 RAGForge 的**杀手级功能**。

大多数 RAG 工具只告诉你最终结果。RAGForge 的 X-Ray 追踪展示**每一步的内部计算**——每个文档在每个阶段的精确分数。

```python
results, trace = pipeline.search(query, documents, trace=True)
print(trace.xray)
```

```
Step                      Time  Output
--------------------------------------------------------------------------------
Retrieval              1917.6ms  3 candidates
Rerank                 1834.1ms  3 docs reranked
Blend                     0.0ms  3 results blended

Final Results:
  Top1  0.3087  苹果手机官方定价
  Top2  0.2821  iPhone 15售价多少
  Top3  0.0640  华为手机报价

Query: 苹果手机多少钱
Total: 3751.8ms

┌─ BM25 Retrieval ─────────────────────────────────────────
│  "苹果手机官方定价                    " BM25=    0.6  rank=1
│  "iPhone 15售价多少               " BM25=    0.5  rank=2
│  "华为手机报价                      " BM25=    0.1  rank=3
└────────────────────────────────────────────────────────────

┌─ Vector Retrieval ─────────────────────────────────────────
│  "苹果手机官方定价                    " cos_sim=0.9609  rank=1
│  "iPhone 15售价多少               " cos_sim=0.9362  rank=2
│  "华为手机报价                      " cos_sim=0.9250  rank=3
└────────────────────────────────────────────────────────────

┌─ RRF Fusion ─────────────────────────────────────────
│  "苹果手机官方定价                    " rrf=0.0645  +bonus_rank1=0.05  → 0.1145
│  "iPhone 15售价多少               " rrf=0.0635  +bonus_rank2_3=0.02  → 0.0835
│  "华为手机报价                      " rrf=0.0625  +bonus_rank2_3=0.02  → 0.0825
└────────────────────────────────────────────────────────────

┌─ Rerank ────────────────────────────────────────────
│  "苹果手机官方定价                    " logit=   2.1  sigmoid=0.891  "Highly relevant"
│  "iPhone 15售价多少               " logit=   2.0  sigmoid=0.878  "Highly relevant"
│  "华为手机报价                      " logit=  -4.7  sigmoid=0.009  "Low relevance"
└────────────────────────────────────────────────────────────

┌─ Blend ─────────────────────────────────────────────
│  "苹果手机官方定价                    " 0.1145×0.75 + 0.891×0.25 = 0.3087
│  "iPhone 15售价多少               " 0.0835×0.75 + 0.878×0.25 = 0.2821
│  "华为手机报价                      " 0.0825×0.75 + 0.009×0.25 = 0.0640
└────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  FINAL RESULTS:
  #1   0.3087  苹果手机官方定价  ← FINAL #1
  #2   0.2821  iPhone 15售价多少  ← FINAL #2
  #3   0.0640  华为手机报价  ← FINAL #3

```

> 一眼看到每个文档为什么排在这个位置——是 BM25 拉上去的？还是 Rerank？权重怎么分配的？调参再也不用猜。

---

## 🗺️ 功能全景

### 🔎 检索核心

| 功能 | 说明 | 推理 |
|------|------|------|
| **Hybrid Retrieval** | BM25 + Dense Vector 并行检索 | CPU |
| **RRF Fusion** | Reciprocal Rank Fusion + Top-Position Bonus | CPU |
| **Cross-Encoder Reranking** | ONNX Reranker (int8 量化) + Sigmoid 归一化 | CPU |
| **Position-Aware Blending** | 按排名区间动态分配检索分 / 重排分权重 | CPU |
| **Adaptive Fusion** | 从反馈中自动学习最优 RRF 参数 | CPU |

### 🧠 查询理解

| 功能 | 说明 | 推理 |
|------|------|------|
| **Query Rewrite** | 口语化查询 → 检索友好查询 | DeepSeek API / 未来本地小模型 |
| **Query Decomposition** | 多意图查询 → 2-4 个子查询 | DeepSeek API / 未来本地小模型 |
| **HyDE** | 生成假设答案文档提升向量召回 | DeepSeek API / 未来本地小模型 |
| **Query Expansion** | 同义词 / 相关词扩展 | DeepSeek API / 未来本地小模型 |
| **Multi-Query Retrieval** | 原始 + 改写查询分别检索 → RRF 融合 | CPU + API |

> **设计哲学**：检索管道的核心计算（Embedding、BM25、Rerank、Fusion）全部本地 CPU 完成。查询理解（Query Rewrite 等）目前通过 DeepSeek API 实现，后续将支持接入本地小模型（如 Qwen2.5-1.5B、DeepSeek-R1-Distill），实现完全离线部署。

### 🔭 可观测性

| 功能 | 说明 |
|------|------|
| **X-Ray Trace** | 每个文档在每一步的精确分数（**独有**） |
| **Pipeline Trace** | 步骤级计时 + 摘要 |
| **Latency Profiler** | 每步耗时 + 占比报告 |

### 📊 评估

| 功能 | 说明 |
|------|------|
| **LLM-as-Judge** | DeepSeek 自动判断相关性 |
| **Metrics** | NDCG / Recall / Precision / MRR |
| **A/B Comparison** | 两个管道配置并排对比 |

### 🛠 工程能力

| 功能 | 说明 |
|------|------|
| **Semantic Cache** | 基于向量相似度的结果缓存 |
| **Document Dedup** | 基于向量相似度的文档去重 |
| **Protocol-Based** | 6 个接口，任何组件都可替换 |
| **Zero Global State** | 所有依赖显式注入 |
| **Lazy Loading** | 模型首次使用时加载，自动缓存 |

---

## 📖 详细用法
详见demo.py

<details>
<summary><b>🔧 查询理解（DeepSeek API）</b></summary>

```python
from ragforge import QueryPlanner, LLMConfig

llm_cfg = LLMConfig(api_key="sk-your-deepseek-key")
planner = QueryPlanner(llm_cfg)

# 改写：口语 → 检索友好
rewritten = planner.rewrite("苹果手机多少钱")
# → "苹果 iPhone 系列手机 价格 报价 官方售价"

# 分解：多意图 → 子查询
sub_queries = planner.decompose("对比iPhone和华为的拍照效果")
# → ["iPhone 拍照效果评测", "华为手机 拍照效果评测", "iPhone vs 华为 拍照对比"]

# HyDE：生成假设答案
hypothetical = planner.hyde("什么是RAG")
# → "RAG（Retrieval-Augmented Generation）是一种结合了检索..."

# 扩展：同义词
expansions = planner.expand("深度学习")
# → ["neural network", "machine learning", ...]
```

</details>

<details>
<summary><b>🔄 Multi-Query Retrieval（原始 + 改写分别检索）</b></summary>

```python
from ragforge import (
    SearchPipeline, QueryPlanner, RRFFusion, PositionAwareBlend,
    FastembedEmbedder, FastembedReranker,
    ModelConfig, PipelineConfig, LLMConfig, QueryTransformStrategy,
)

pipe_cfg = PipelineConfig(
    query_transform_strategy=QueryTransformStrategy.RETRIEVE_AND_FUSE,
)

pipeline = SearchPipeline(
    embedder=FastembedEmbedder(model_cfg),
    reranker=FastembedReranker(model_cfg),
    fusion=RRFFusion(pipe_cfg),
    blend=PositionAwareBlend(pipe_cfg),
    query_transform=QueryPlanner(LLMConfig(api_key="sk-...")),
    config=pipe_cfg,
)

# 原始查询 + 改写查询 → 并行检索 → RRF 融合
results, trace = pipeline.search("苹果手机多少钱", documents, trace=True)
```

流程：
```
query ──► [改写] ─┬──► [检索: 原始查询]   ──┐
                   └──► [检索: 改写查询]   ──┼──► [RRF 融合] ──► [Rerank] ──► 结果
```

> 💡 也支持 `decompose()`——原始 + N 个子查询 → (N+1) 路并行检索，全部融合。

</details>

<details>
<summary><b>📊 评估 & A/B 对比</b></summary>

```python
from ragforge import Evaluator, LLMJudge, LLMConfig

judge = LLMJudge(LLMConfig(api_key="sk-..."))
evaluator = Evaluator(judge=judge)

# 评估管道质量
metrics = evaluator.evaluate(pipeline, queries, ground_truth, top_k=5)
print(f"NDCG@5: {metrics.ndcg:.3f}, Recall@5: {metrics.recall:.3f}, MRR: {metrics.mrr:.3f}")

# A/B 对比两个管道
comparison = evaluator.compare(
    pipeline_a=hybrid_pipeline,
    pipeline_b=bm25_only_pipeline,
    queries=queries,
    ground_truth=ground_truth,
)
print(comparison["report"])
```

```
Pipeline Comparison Report
======================================================================
Metric                Pipeline A  Pipeline B
--------------------------------------------
NDCG@5                    0.8200      0.6100
Recall@5                  0.9000      0.7000
Precision@5               0.7200      0.5200
MRR                       0.9500      0.7500
Avg Latency (ms)         320.0       12.0
```

</details>

<details>
<summary><b>⚡ 自适应融合（从反馈中学习）</b></summary>

```python
from ragforge import AdaptiveFusion

feedback = [
    ("苹果手机价格", {"iPhone 15 官方售价 5999元", "苹果官网定价"}),
    ("Python教程", {"Python入门指南", "Python最佳实践"}),
]

fusion = AdaptiveFusion.from_feedback(feedback)
print(f"Best: k={fusion.best_config.rrf_k}, weight={fusion.best_config.query_weight}")

pipeline = SearchPipeline(fusion=fusion)
```

</details>

<details>
<summary><b>📦 语义缓存 & 文档去重</b></summary>

```python
# 语义缓存：相似查询直接返回缓存结果
from ragforge import SemanticCache
cache = SemanticCache(embedder=embedder, similarity_threshold=0.95)
result1 = cache.get_or_search("苹果手机价格", pipeline.search, documents)
result2 = cache.get_or_search("苹果手机多少钱", pipeline.search, documents)  # 缓存命中
print(cache.stats)  # {'entries': 1, 'hits': 1, 'misses': 0, 'hit_rate': 1.0}

# 文档去重：基于向量相似度
from ragforge import Deduplicator
dedup = Deduplicator(embedder=embedder, threshold=0.95)
unique = dedup.deduplicate(documents)
print(f"{len(documents)} → {len(unique)} unique documents")
```

</details>

<details>
<summary><b>🎨 自定义组件（Protocol 接口）</b></summary>

```python
# 只需实现接口方法，无需继承任何类
class MyRetriever:
    def retrieve(self, query: str, documents: list[str]) -> list[tuple[str, int]]:
        # 你的检索逻辑
        return [(doc, rank) for rank, doc in enumerate(sorted_docs)]

# 换用 OpenAI Embedding
class OpenAIEmbedder:
    def __init__(self, api_key): ...
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...

# 直接插入管道
pipeline = SearchPipeline(retriever=MyRetriever())
```

</details>

---

## 🏛 Architecture

### Pipeline Flow

```
query ──► [Query Transform] ──► [Retriever] ──► [Fusion] ──► [Rerank] ──► [Blend] ──► results
              optional           always       if hybrid    optional    optional
```

Multi-Query 模式：

```
query ──► [Transform] ─┬──► [检索: 原始查询]  ──┐
                      ├──► [检索: 改写查询]  ──┼──► [RRF 融合] ──► [Rerank] ──► [Blend] ──► results
                      └──► [检索: 子查询3]   ──┘
```

### Protocol Interfaces

```python
class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...

class Retriever(Protocol):
    def retrieve(self, query: str, documents: list[str]) -> list[tuple[str, int]]: ...

class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...

class FusionStrategy(Protocol):
    def fuse(self, ranked_lists: list[list[tuple[str, int]]]) -> list[tuple[str, float]]: ...

class QueryTransform(Protocol):
    def transform(self, query: str) -> str | list[str]: ...

class Judge(Protocol):
    def judge(self, query: str, documents: list[str]) -> list[dict]: ...
```

### Project Structure

```
ragforge/
├── protocols.py                 # 6 个 Protocol 接口定义
├── type_utils.py                # 共享数据类型 + X-Ray 格式化
├── pipeline.py                  # SearchPipeline（组装所有组件）
│
├── config/
│   ├── model_config.py          # 模型路径配置
│   ├── pipeline_config.py       # 算法超参数 + QueryTransformStrategy
│   └── llm_config.py            # DeepSeek LLM 配置
│
├── models/                      # ONNX CPU 推理模型
│   ├── embedding.py             # FastembedEmbedder (ONNX)
│   └── reranker.py              # FastembedReranker (ONNX, int8)
│
├── retrieval/
│   ├── bm25.py                  # BM25Retriever (jieba)
│   ├── vector.py                # VectorRetriever (cosine)
│   └── hybrid.py                # HybridRetriever (parallel)
│
├── fusion/
│   ├── rrf.py                   # RRFFusion (bonus + adaptive)
│   ├── blend.py                 # PositionAwareBlend
│   └── adaptive.py              # AdaptiveFusion (auto-tuned)
│
├── llm/
│   └── llm_client.py              # LLMClient (OpenAI-compatible)
│
├── query/
│   └── planner.py               # QueryPlanner (rewrite/decompose/hyde/expand)
│
├── evaluation/
│   ├── judge.py                 # LLMJudge (LLM-as-Judge)
│   └── evaluator.py             # Evaluator (NDCG/Recall/MRR + A/B)
│
├── cache/                       # SemanticCache
├── dedup/                       # Deduplicator
├── tracing/                     # Tracer
└── profiler.py                  # PipelineProfiler
```

---

## ⚙️ Configuration

### ModelConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_model_path` | `str \| None` | `None` | Embedding 模型路径（None 则自动下载） |
| `embedding_dim` | `int` | `384` | 向量维度 |
| `rerank_model_path` | `str \| None` | `None` | Reranker 模型路径（None 则自动下载） |

### PipelineConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rrf_k` | `int` | `60` | RRF 常数 |
| `top_k_recall` | `int` | `30` | 融合后保留的候选数 |
| `query_weight` | `float` | `2.0` | RRF 分数权重 |
| `bonus_rank1` | `float` | `0.05` | 第 1 名加分 |
| `bonus_rank2_3` | `float` | `0.02` | 第 2-3 名加分 |
| `query_transform_strategy` | `str` | `"replace"` | `"replace"` 或 `"retrieve_and_fuse"` |
| `blend_weights` | `dict` | 见源码 | 按排名区间的融合权重 |

### LLMConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | `str` | `""` | DeepSeek API Key |
| `base_url` | `str` | `"https://api.deepseek.com"` | API 地址 |
| `model` | `str` | `"deepseek-v4-flash"` | 模型名称 |

---

## API Reference

### 检索组件

| 类 | 接口 | 说明 | 推理 |
|---|------|------|------|
| `FastembedEmbedder` | `Embedder` | ONNX Embedding (multilingual-e5-small) | CPU |
| `FastembedReranker` | `Reranker` | ONNX Cross-Encoder (bge-reranker-v2-m3, int8) | CPU |
| `BM25Retriever` | `Retriever` | BM25 全文检索 (jieba 分词) | CPU |
| `VectorRetriever` | `Retriever` | 余弦相似度向量检索 | CPU |
| `HybridRetriever` | `Retriever` | BM25 + Vector 并行 | CPU |
| `RRFFusion` | `FusionStrategy` | RRF + Top-Position Bonus | CPU |
| `PositionAwareBlend` | `FusionStrategy` | 位置感知权重融合 | CPU |
| `AdaptiveFusion` | `FusionStrategy` | 反馈驱动的参数自学习 | CPU |
| `SearchPipeline` | — | 可组合管道 | CPU |

### 查询 & 评估

| 类 | 接口 | 说明 | 推理 |
|---|------|------|------|
| `QueryPlanner` | `QueryTransform` | 改写 / 分解 / HyDE / 扩展 | API (DeepSeek) |
| `LLMJudge` | `Judge` | LLM 相关性判断 | API (DeepSeek) |
| `Evaluator` | — | NDCG / Recall / MRR + A/B | API + CPU |

### 工程组件

| 类 | 说明 |
|---|------|
| `SemanticCache` | 向量相似度缓存 |
| `Deduplicator` | 近似文档去重 |
| `Tracer` | 步骤级追踪 |
| `PipelineProfiler` | 延迟分析 |

---

## 📦 Installation

### 核心依赖（检索功能，全 CPU）

```bash
pip install fastembed rank_bm25 jieba scikit-learn numpy requests
```

### LLM 功能（Query Understanding、Evaluation）——可选

```bash
pip install openai  # DeepSeek 兼容 OpenAI API
```

> **不装 openai 也完全没问题。** BM25、Embedding、Rerank、RRF、Blend 全部独立运行。只有当你需要 Query Rewrite、HyDE、LLM 评估时才需要接入 LLM。

### 默认模型

| 组件 | 模型 | 大小     | 量化 | 推理 |
|------|------|--------|------|------|
| Embedding | multilingual-e5-small-onnx | ~400MB | FP16 | **CPU** |
| Reranker | bge-reranker-v2-m3-ONNX-int8 | ~500MB | int8 | **CPU** |

来源：[ModelScope](https://modelscope.cn/profile/jiangnanboy)

自动下载到 `~/.cache/RAGForge/`，首次使用时懒加载。

---

## License

Apache2.0 License - see the [LICENSE](LICENSE) file for details.

