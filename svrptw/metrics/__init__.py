"""Comprehensive solution-quality metrics suite.

Per user direction (iter 5f): build a metrics suite covering clustering
quality, utilization balancing, geometric efficiency, time-window slack,
and route shape. The scorer sub-agent consumes these to produce a
unified per-route + per-solution scorecard, which combines with VLM
visual scores into a single objective function.
"""
from __future__ import annotations

from .quality import (
    SolutionQualityScore,
    score_solution,
    score_routes,
)

__all__ = ["SolutionQualityScore", "score_solution", "score_routes"]
