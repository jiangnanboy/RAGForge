"""
Pipeline Configuration
======================
Dataclass holding algorithm hyper-parameters for the search pipeline.

Separation of Concerns:
    - *ModelConfig* → model paths, file names, dimensions
    - *PipelineConfig* → algorithm hyper-parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QueryTransformStrategy(str, Enum):
    """Strategy for how query transform results are used in the pipeline.

    Attributes:
        REPLACE:
            Replace the original query with the transformed version.
            This is the default and original behavior.
        RETRIEVE_AND_FUSE:
            Retrieve with both the original query AND the transformed
            query, then fuse all result sets together. This implements
            the **Multi-Query Retrieval** pattern for higher recall.

            - If ``transform()`` returns a single string, two retrievals
              are performed (original + rewritten) and results are fused.
            - If ``transform()`` returns a list of sub-queries, N+1
              retrievals are performed (original + each sub-query) and
              all results are fused via RRF.
    """

    REPLACE = "replace"
    RETRIEVE_AND_FUSE = "retrieve_and_fuse"


@dataclass
class PipelineConfig:
    """Hyper-parameters for the retrieval & fusion pipeline.

    Attributes:
        rrf_k:
            Reciprocal Rank Fusion constant. Higher values flatten
            score differences across ranks.
        top_k_recall:
            Number of candidates to keep after fusion (before reranking).
        query_weight:
            Multiplier applied to RRF base scores.
        bonus_rank1:
            Score bonus for the top-1 document after RRF fusion.
        bonus_rank2_3:
            Score bonus for documents ranked 2-3 after RRF fusion.
        blend_weights:
            Position-aware blending weights mapping rank buckets
            to ``(retrieval_weight, reranker_weight)`` dicts.
        query_transform_strategy:
            How to apply query transformation results.
            ``REPLACE`` (default): use rewritten query directly.
            ``RETRIEVE_AND_FUSE``: retrieve with both original and
            rewritten queries, then fuse results.
    """

    rrf_k: int = 60
    top_k_recall: int = 30
    query_weight: float = 2.0
    bonus_rank1: float = 0.05
    bonus_rank2_3: float = 0.02
    query_transform_strategy: QueryTransformStrategy = QueryTransformStrategy.REPLACE
    blend_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "top1-3": {"retrieval": 0.75, "reranker": 0.25},
            "top4-10": {"retrieval": 0.60, "reranker": 0.40},
            "top11+": {"retrieval": 0.40, "reranker": 0.60},
        }
    )
