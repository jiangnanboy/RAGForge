"""
Fastembed Embedder
==================
A standalone ``Embedder`` implementation powered by `fastembed`.

Features:
    - Lazy model loading (loaded on first ``embed()`` call)
    - Auto-download from ModelScope if no local path is configured
    - Fully independent — no global state, no singleton
    - Satisfies the ``Embedder`` protocol
"""

from __future__ import annotations

import os

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from ..config import ModelConfig
from ..utils import download_model_if_missing


class FastembedEmbedder:
    """Embedding model wrapper using fastembed ONNX runtime.

    Args:
        config: A ``ModelConfig`` instance with embedding model settings.

    Example::

        from RAGForge import FastembedEmbedder, ModelConfig

        embedder = FastembedEmbedder(ModelConfig(
            embedding_model_path="/path/to/model",
            embedding_dim=384,
        ))
        vec = embedder.embed("Hello, world!")
        vecs = embedder.embed_batch(["Hello", "World"])
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: TextEmbedding | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a dense vector.

        Args:
            text: Input text.

        Returns:
            1-D ``np.ndarray`` of shape ``(embedding_dim,)``.
        """
        model = self._load_model()
        return np.array(list(model.embed(text))[0])

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts.

        Args:
            texts: List of input strings.

        Returns:
            List of 1-D ``np.ndarray`` vectors.
        """
        model = self._load_model()
        return [np.array(emb) for emb in model.embed(texts)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> TextEmbedding:
        """Lazy-load the embedding model (called once, then cached)."""
        if self._model is not None:
            return self._model

        model_path = self._config.embedding_model_path

        if model_path is None:
            model_path = self._auto_download()

        # Register & instantiate via fastembed's custom model API
        # Ignore "already registered" — can happen under multi-threaded loading
        try:
            TextEmbedding.add_custom_model(
                model=model_path,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(url=model_path),
                dim=self._config.embedding_dim,
                model_file=self._config.embedding_onnx_file,
            )
        except ValueError:
            pass
        self._model = TextEmbedding(
            model_name=model_path,
            specific_model_path=model_path,
        )
        return self._model

    def _auto_download(self) -> str:
        """Auto-download the default embedding model from ModelScope."""
        home = os.path.expanduser("~")
        model_dir = os.path.join(
            home, ".cache", "RAGForge", "multilingual-e5-small-onnx"
        )
        base_url = (
            "https://modelscope.cn/models/jiangnanboy/"
            "multilingual-e5-small-onnx/resolve/master"
        )
        required_files = {
            "model.onnx": f"{base_url}/model.onnx",
            "config.json": f"{base_url}/config.json",
            "tokenizer.json": f"{base_url}/tokenizer.json",
            "special_tokens_map.json": f"{base_url}/special_tokens_map.json",
            "tokenizer_config.json": f"{base_url}/tokenizer_config.json",
            "sentencepiece.bpe.model": f"{base_url}/sentencepiece.bpe.model",
        }

        for filename, url in required_files.items():
            download_model_if_missing(os.path.join(model_dir, filename), url)

        return model_dir
