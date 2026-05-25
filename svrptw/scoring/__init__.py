"""Unified scoring API.

Combines operational cost (from solver), structural quality_index
(from svrptw.metrics.quality), and optional VLM visual_score into a
single weighted ``UnifiedScore``. Used by the construction-trial
bench and by the live council loop to pick a winner across multiple
candidates that disagree on cheap cost vs structural visual.
"""
from __future__ import annotations

from .sub_api import (
    UnifiedScore,
    compare,
    score,
    score_batch,
    DEFAULT_WEIGHTS,
)

__all__ = [
    "UnifiedScore",
    "score",
    "compare",
    "score_batch",
    "DEFAULT_WEIGHTS",
]
