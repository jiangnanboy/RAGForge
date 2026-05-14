"""
Query Planner
=============
Query understanding layer using DeepSeek LLM.

Features:
    - **Rewrite**: Transform colloquial queries into retrieval-friendly form
    - **Decompose**: Split complex multi-intent queries into sub-queries
    - **HyDE**: Generate a hypothetical answer document for better vector recall
    - **Expand**: Generate synonym/related term expansions

All features use DeepSeek via prompt engineering — no fine-tuning required.
"""

from __future__ import annotations

from ..config import LLMConfig
from ..llm.llm_client import LLMClient

# english
_SYSTEM_REWRITE = """You are a search query rewriter. Rewrite the user's query into a more effective search query that would work well for information retrieval. Keep it concise and keyword-rich. Output ONLY the rewritten query, nothing else."""

_SYSTEM_DECOMPOSE = """You are a query decomposition assistant. Break down the user's complex query into 2-4 simpler sub-queries, each targeting one aspect. Output a JSON array of strings, e.g. ["sub-query 1", "sub-query 2"]. Output ONLY the JSON array."""

_SYSTEM_HYDE = """You are a helpful assistant. Given a user's question, generate a hypothetical answer document that would be relevant to answering the question. Write it as a factual, concise paragraph. Output ONLY the hypothetical document text."""

_SYSTEM_EXPAND = """You are a search query expander. Given the user's search query, generate 3-5 related search terms or synonyms that could improve retrieval. Output a JSON array of strings. Output ONLY the JSON array."""

# chinese
_SYSTEM_REWRITE = """你是一名搜索查询改写员。将用户的查询改写成一个对信息检索更有效的搜索查询。保持简洁且富含关键词。仅输出改写后的查询，无其他内容。"""

_SYSTEM_DECOMPOSE = """你是一名查询分解助手。将用户的复杂查询分解为 2 到 4 个更简单的子查询，每个子查询针对一个方面。以 JSON 数组形式输出字符串，例如 ["子查询 1", "子查询 2"]。仅输出 JSON 数组。"""

_SYSTEM_HYDE = """你是一个乐于助人的助手。根据用户的问题，生成一个可能与回答该问题相关的假设性答案文档。将其写成一段事实性的、简洁的段落。仅输出假设性文档文本。"""

_SYSTEM_EXPAND = """你是一个搜索查询扩展器。给定用户的搜索查询，生成 3 到 5 个相关的搜索词或同义词，以提高检索效果。输出一个字符串的 JSON 数组。仅输出 JSON 数组。"""


class QueryPlanner:
    """Query understanding with DeepSeek LLM.

    Args:
        config: An ``LLMConfig`` with DeepSeek API settings.
            Alternatively, pass a ``LLMClient`` directly.

    Example::

        from ragforge import QueryPlanner, LLMConfig

        planner = QueryPlanner(LLMConfig(api_key="sk-..."))

        # Rewrite
        rewritten = planner.rewrite("苹果手机多少钱")

        # Decompose
        sub_queries = planner.decompose("对比iPhone和华为的拍照效果")

        # HyDE
        hypothetical = planner.hyde("什么是RAG检索增强生成")

        # Expand
        expansions = planner.expand("深度学习")
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: LLMClient | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        elif config is not None:
            self._client = LLMClient(config)
        else:
            raise ValueError("Either config or client must be provided")

    def rewrite(self, query: str) -> str:
        """Rewrite a query for better retrieval.

        Args:
            query: Raw user query.

        Returns:
            Rewritten query string.
        """
        return self._client.chat(
            prompt=query,
            system=_SYSTEM_REWRITE,
            temperature=0.0,
        ).strip()

    def decompose(self, query: str) -> list[str]:
        """Decompose a complex query into sub-queries.

        Args:
            query: Complex multi-intent query.

        Returns:
            List of sub-query strings.
        """
        result = self._client.chat_json(
            prompt=query,
            system=_SYSTEM_DECOMPOSE,
            temperature=0.1,
        )
        if isinstance(result, list):
            return [str(item) for item in result]
        return [str(result)]

    def hyde(self, query: str) -> str:
        """Generate a Hypothetical Document Embedding (HyDE).

        Creates a hypothetical answer that can be used as a proxy
        for vector retrieval when the original query is too short
        or ambiguous.

        Args:
            query: User question.

        Returns:
            Hypothetical answer document text.
        """
        return self._client.chat(
            prompt=query,
            system=_SYSTEM_HYDE,
            temperature=0.3,
        ).strip()

    def expand(self, query: str) -> list[str]:
        """Generate synonym/related term expansions.

        Args:
            query: Original search query.

        Returns:
            List of expanded query variants.
        """
        result = self._client.chat_json(
            prompt=query,
            system=_SYSTEM_EXPAND,
            temperature=0.3,
        )
        if isinstance(result, list):
            return [str(item) for item in result]
        return [str(result)]

    def transform(self, query: str) -> str | list[str]:
        """QueryTransform protocol — default to rewrite.

        Args:
            query: Raw user query.

        Returns:
            Rewritten query string.
        """
        return self.rewrite(query)
