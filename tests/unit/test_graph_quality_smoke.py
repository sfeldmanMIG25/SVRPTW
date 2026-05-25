"""Phase G2 smoke tests for the graph-aware quality metric.

Cheap-by-design: synthetic Solomon-style instances exercise the
non-geographic fallback path so the test runs offline. The geographic
path is exercised opportunistically via a cached OSM-Manhattan instance
when the cache is present.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.metrics.graph_quality import (
    GraphSolutionQualityScore,
    score_solution_graph,
)
from svrptw.solvers.classical.fast_construct import solve as fc_solve
from svrptw.solvers.common import Solution


def test_score_solution_graph_returns_finite():
    inst = generate(N=20, seed=42)
    sol = fc_solve(inst, Settings(), budget_seconds=0.5, seed=0)
    qs = score_solution_graph(inst, sol)
    assert isinstance(qs, GraphSolutionQualityScore)
    assert math.isfinite(qs.quality_index_graph)
    assert 0.0 <= qs.quality_index_graph <= 1.0


def test_non_geographic_falls_back_to_euclidean():
    """Solomon-style synthetic coords (~[0,100]) trip the
    non-geographic guard. The score should still come out finite via
    the Euclidean fallback path.
    """
    inst = generate(N=20, seed=7)
    sol = fc_solve(inst, Settings(), budget_seconds=0.5, seed=0)
    qs = score_solution_graph(inst, sol)
    # Fallback was used.
    assert qs.fallback_reason in {"non_geographic", "no_node_ids",
                                  "graph_unavailable"}
    # Quality index still in [0, 1].
    assert 0.0 <= qs.quality_index_graph <= 1.0


def test_empty_solution_does_not_crash():
    inst = generate(N=10, seed=11)
    # Build a trivially empty solution (all routes empty).
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[],
        solver="empty",
        wall_clock_seconds=0.0,
        budget_seconds=0.0,
        feasible=False,
    )
    qs = score_solution_graph(inst, sol)
    assert math.isfinite(qs.quality_index_graph)
    assert qs.n_routes == 0


def test_compute_time_at_n200_warm_path():
    """Warm-path timing on a geographic OSM instance (the path the
    metric is actually optimized for). After warm-up, repeat calls
    should complete well under 10ms wall (the Euclidean N=200 metric
    is ~25-100ms; the graph-aware path is roughly an order of
    magnitude faster on the warm steady state).

    Spec target: <5ms. Test asserts <10ms to absorb pytest+OS jitter
    while still flagging an order-of-magnitude regression.

    Skipped when the OSM-Manhattan-N200 instance + cached graph are
    not available (offline / fresh checkout).
    """
    inst_path = Path("instances/v1/OSM-Manhattan-N200-I000.json")
    if not inst_path.exists():
        pytest.skip("OSM-Manhattan-N200-I000 instance not present")
    cache_dir = Path("webui/cache/networks")
    if not any(cache_dir.glob("drive__*.graphml")):
        pytest.skip("OSM driving graph not cached; skipping warm-path timing")
    from svrptw.io import load_instance
    inst = load_instance(str(inst_path))
    sol = fc_solve(inst, Settings(), budget_seconds=1.0, seed=0)
    # Warm-up: build pair table + load BC + Louvain.
    score_solution_graph(inst, sol)
    score_solution_graph(inst, sol)
    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        qs = score_solution_graph(inst, sol)
        times.append((time.perf_counter() - t0) * 1000)
    assert math.isfinite(qs.quality_index_graph)
    assert min(times) < 10.0, (
        f"min compute time {min(times):.2f}ms exceeds 10ms "
        f"(avg {sum(times)/len(times):.2f}ms over {len(times)} calls)"
    )
