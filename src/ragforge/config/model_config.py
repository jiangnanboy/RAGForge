"""
Model Configuration
===================
Dataclass holding model paths and loading parameters.

Separation of Concerns:
    - *ModelConfig* → model paths, file names, dimensions
    - *PipelineConfig* → algorithm hyper-parameters

This makes it easy to swap models without touching pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for embedding and reranker models.

    Attributes:
        embedding_model_path:
            Local directory containing the embedding ONNX model files.
            If ``None``, the model will be auto-downloaded on first use.
        embedding_onnx_file:
            Name of the embedding ONNX weight file (default ``model.onnx``).
        embedding_dim:
            Dimensionality of the embedding vectors.
        rerank_model_path:
            Local directory containing the reranker ONNX model files.
            If ``None``, the model will be auto-downloaded on first use.
        rerank_onnx_file:
            Name of the reranker ONNX weight file
            (default ``model_int8.onnx``).
    """

    # --- Embedding model ---
    embedding_model_path: str | None = None
    embedding_onnx_file: str = "model.onnx"
    embedding_dim: int = 384

    # --- Reranker model ---
    rerank_model_path: str | None = None
    rerank_onnx_file: str = "model_int8.onnx"
