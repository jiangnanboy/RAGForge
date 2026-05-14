"""
Fastembed Reranker
==================
A standalone ``Reranker`` implementation powered by `fastembed`.

Features:
    - Lazy model loading
    - Auto-download from ModelScope if no local path is configured
    - Sigmoid normalization & relevance-level classification built-in
    - Fully independent — no global state
    - Satisfies the ``Reranker`` protocol
"""

from __future__ import annotations

import os

from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.common.model_description import ModelSource

from ..config import ModelConfig
from ..utils import download_model_if_missing, sigmoid, get_relevance_level


class FastembedReranker:
    """Cross-encoder reranker using fastembed ONNX runtime.

    Args:
        config: A ``ModelConfig`` instance with reranker model settings.

    Example::

        from RAGForge import FastembedReranker, ModelConfig

        reranker = FastembedReranker(ModelConfig(
            rerank_model_path="/path/to/reranker",
        ))

        # Raw logit scores
        scores = reranker.rerank("query", ["doc1", "doc2"])

        # Normalized scores + relevance labels
        norm, levels = reranker.rerank_normalized("query", ["doc1", "doc2"])
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: TextCrossEncoder | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Rerank documents, returning raw logit scores.

        Args:
            query: Search query.
            documents: Candidate documents.

        Returns:
            List of raw scores (one per document, same order).
        """
        model = self._load_model()
        pairs = [(query, doc) for doc in documents]
        return list(model.rerank_pairs(pairs))

    def rerank_normalized(
        self, query: str, documents: list[str]
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Rerank with sigmoid normalization and relevance labels.

        Args:
            query: Search query.
            documents: Candidate documents.

        Returns:
            A 2-tuple of:
            - ``{document: normalized_score}`` (float in [0, 1])
            - ``{document: relevance_level}`` (human-readable string)
        """
        raw_scores = self.rerank(query, documents)
        norm_scores = [sigmoid(s) for s in raw_scores]
        rel_levels = [get_relevance_level(s) for s in norm_scores]
        return dict(zip(documents, norm_scores)), dict(zip(documents, rel_levels))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> TextCrossEncoder:
        """Lazy-load the reranker model (called once, then cached)."""
        if self._model is not None:
            return self._model

        model_path = self._config.rerank_model_path

        if model_path is None:
            model_path = self._auto_download()

        print('rerank model: {}'.format(model_path))
        TextCrossEncoder.add_custom_model(
            model=model_path,
            model_file=self._config.rerank_onnx_file,
            sources=ModelSource(url=model_path),
        )
        self._model = TextCrossEncoder(
            model_name=model_path,
            specific_model_path=model_path,
        )
        return self._model

    def _auto_download(self) -> str:
        """Auto-download the default reranker model from ModelScope."""
        home = os.path.expanduser("~")
        model_dir = os.path.join(
            home, ".cache", "RAGForge", "bge-reranker-v2-m3-ONNX-int8"
        )
        base_url = (
            "https://modelscope.cn/models/jiangnanboy/"
            "bge-reranker-v2-m3-ONNX-int8/resolve/master"
        )
        required_files = {
            "model_int8.onnx": f"{base_url}/model_int8.onnx",
            "config.json": f"{base_url}/config.json",
            "configuration.json": f"{base_url}/configuration.json",
            "tokenizer.json": f"{base_url}/tokenizer.json",
            "special_tokens_map.json": f"{base_url}/special_tokens_map.json",
            "tokenizer_config.json": f"{base_url}/tokenizer_config.json",
            "quantize_config.json": f"{base_url}/quantize_config.json",
        }

        for filename, url in required_files.items():
            download_model_if_missing(os.path.join(model_dir, filename), url)

        return model_dir
