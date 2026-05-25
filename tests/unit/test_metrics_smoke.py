"""Smoke tests for ``svrptw.metrics.score_solution``.

The metrics module is the foundation for the unified scoring API; this
test guarantees it returns finite, in-range values and that visibly
different solutions (well-clustered vs all-in-one-route) actually
produce different ``quality_index`` values.
"""
from __future__ import annotations

import math

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.metrics import SolutionQualityScore, score_solution
from svrptw.solvers.classical import greedy
from svrptw.solvers.common.solution import Route, Solution


def _all_in_one_solution(inst) -> Solution:
    """Pathological solution: every customer on a single route, in id
    order. Used as the 'visibly worse' baseline for difference checks."""
    cids = [c.id for c in inst.customers]
    return Solution(
        instance_id=inst.instance_id,
        routes=[Route(customers=cids)],
        solver="degenerate_one_route",
        wall_clock_seconds=0.0,
        budget_seconds=0.0,
        feasible=False,
    )


def test_score_solution_returns_finite_fields():
    inst = generate(N=20, seed=11)
    sol = greedy.solve(inst, Settings())
    qs = score_solution(inst, sol)
    assert isinstance(qs, SolutionQualityScore)
    # Numeric scalars must be finite (no NaN / inf leaking from divides).
    for name in (
        "intra_inter_ratio", "silhouette_like", "load_util_cv",
        "stops_per_route_cv", "load_gini", "mean_detour_ratio",
        "mean_slack_min", "tight_stops_frac", "wait_to_service_ratio",
        "quality_index",
    ):
        v = getattr(qs, name)
        assert math.isfinite(v), f"{name}={v!r} is not finite"
    # Counts non-negative.
    assert qs.n_routes >= 0
    assert qs.n_customers_served >= 0
    assert qs.n_unrouted >= 0
    assert qs.convex_hull_overlap_count >= 0
    assert qs.inter_route_crossings >= 0


def test_quality_index_in_unit_range():
    inst = generate(N=30, seed=7)
    sol = greedy.solve(inst, Settings())
    qs = score_solution(inst, sol)
    assert 0.0 <= qs.quality_index <= 1.0, (
        f"quality_index={qs.quality_index} out of [0, 1]"
    )


def test_empty_solution_does_not_crash():
    inst = generate(N=20, seed=3)
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[],
        solver="empty",
        wall_clock_seconds=0.0,
        budget_seconds=0.0,
        feasible=False,
    )
    qs = score_solution(inst, sol)
    assert qs.n_routes == 0
    assert qs.n_unrouted == inst.num_customers
    assert math.isfinite(qs.quality_index)


def test_visibly_different_solutions_have_different_quality_index():
    """Compare a real (multi-route) solver output to the all-in-one
    degenerate solution. They MUST NOT score identically — that would
    mean the metrics suite is collapsing legitimately distinct
    solutions to the same scalar."""
    inst = generate(N=30, seed=5)
    good = greedy.solve(inst, Settings())
    bad = _all_in_one_solution(inst)
    qg = score_solution(inst, good)
    qb = score_solution(inst, bad)
    assert qg.quality_index != qb.quality_index, (
        f"degenerate vs greedy collapsed to same quality_index "
        f"({qg.quality_index})"
    )
