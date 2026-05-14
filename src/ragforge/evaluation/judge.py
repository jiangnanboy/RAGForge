"""
LLM Judge (LLM-as-Judge)
==============================
Use LLM to evaluate whether retrieved documents are relevant to a query.

Features:
    - Binary relevance judgment (relevant / not relevant)
    - Relevance score (0-1)
    - Human-readable reasoning
    - Batch evaluation for efficiency
"""

from __future__ import annotations

from ..config import LLMConfig
from ..llm.llm_client import LLMClient
from ..protocols import Judge
from ..type_utils import JudgeResult

# english
_SYSTEM_JUDGE = """You are a relevance judge for a search system. Given a query and a retrieved document, determine if the document is relevant to answering the query.

Respond in JSON format:
{"relevant": true/false, "score": 0.0-1.0, "reason": "brief explanation"}

Scoring guide:
- 0.9-1.0: Highly relevant — directly answers the query
- 0.7-0.8: Mostly relevant — useful but incomplete
- 0.4-0.6: Partially relevant — tangentially related
- 0.0-0.3: Not relevant — does not address the query

Output ONLY the JSON object, no other text."""

# chinese
_SYSTEM_JUDGE = """
你是一个搜索系统的相关性评判员。给定一个查询和一篇检索到的文档，判断该文档对于回答查询是否相关。

以 JSON 格式回复：
{"relevant": true/false, "score": 0.0-1.0, "reason": "简要解释"}

评分指南：
0.9 - 1.0：高度相关 —— 直接回答查询
0.7 - 0.8：大部分相关 —— 有用但不完整
0.4 - 0.6：部分相关 —— 间接相关
0.0 - 0.3：不相关 —— 未涉及查询内容

仅输出 JSON 对象，不输出其他文本。
"""

class LLMJudge(Judge):
    """LLM-as-Judge using DeepSeek.

    Args:
        config: An ``LLMConfig`` with DeepSeek API settings.
            Alternatively, pass a ``LLMClient`` directly.

    Example::

        from ragforge import LLMJudeg, LLMConfig

        judge = LLMJudge(LLMConfig(api_key="sk-..."))
        results = judge.judge("苹果手机价格", [
            "iPhone 15 官方售价 5999 元起",
            "华为手机最新报价",
        ])
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

    def judge(
        self, query: str, documents: list[str]
    ) -> list[dict]:
        """Judge relevance of each document to the query.

        Args:
            query: Search query.
            documents: Retrieved documents to evaluate.

        Returns:
            List of dicts with keys:
            ``document``, ``relevant``, ``score``, ``reason``.
        """
        results: list[dict] = []
        for doc in documents:
            prompt = f"Query: {query}\n\nDocument: {doc}"
            try:
                data = self._client.chat_json(
                    prompt=prompt,
                    system=_SYSTEM_JUDGE,
                    temperature=0.0,
                )
                results.append({
                    "document": doc,
                    "relevant": bool(data.get("relevant", False)),
                    "score": float(data.get("score", 0.0)),
                    "reason": str(data.get("reason", "")),
                })
            except Exception as e:
                results.append({
                    "document": doc,
                    "relevant": False,
                    "score": 0.0,
                    "reason": f"Judge error: {e}",
                })
        return results

    def judge_single(
        self, query: str, document: str
    ) -> JudgeResult:
        """Judge a single document, returning a typed ``JudgeResult``.

        Args:
            query: Search query.
            document: A single retrieved document.

        Returns:
            A ``JudgeResult`` instance.
        """
        results = self.judge(query, [document])
        r = results[0]
        return JudgeResult(
            document=r["document"],
            relevant=r["relevant"],
            score=r["score"],
            reason=r["reason"],
        )
