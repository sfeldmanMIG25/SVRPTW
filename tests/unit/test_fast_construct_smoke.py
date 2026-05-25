"""Phase E2 smoke tests for fast_construct + portfolio_pyvrp_warm.construction.

These tests are intentionally cheap (synthetic N=20-30 instances) so they
fit the unit-test budget. They cover:
  1. fast_construct.solve returns a feasible solution with > 0 routes.
  2. Wall budget honored (within 30% slack).
  3. solve_auto(construction="fast_construct") produces a finite cost.
  4. solve_auto(construction="pyvrp", seed=fixed) is reproducible across
     two consecutive calls (backward-compat).
"""
from __future__ import annotations

import math

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.metrics import score_solution
from svrptw.solvers.classical import fast_construct as fc
from svrptw.solvers.classical import portfolio as pm
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto


def test_fast_construct_returns_feasible():
    """fast_construct.solve produces a feasible solution with at least
    one non-empty route on a small synthetic instance."""
    inst = generate(N=20, seed=42)
    sol = fc.solve(inst, Settings(), budget_seconds=1.0, seed=0)
    assert sol.solver == "fast_construct"
    assert sol.metrics["operational_cost"] < float("inf")
    # At least one route should carry customers.
    n_used = int(sol.metrics["num_vehicles_used"])
    assert n_used > 0, f"fast_construct returned 0 used routes: {sol.routes}"
    assert sol.feasible, f"fast_construct returned infeasible: {sol.metrics}"


def test_fast_construct_honors_budget():
    """Wall time should not exceed budget * 1.3 (generous slack for the
    light LS phase to finish its current op)."""
    inst = generate(N=30, seed=7)
    budget = 1.0
    sol = fc.solve(inst, Settings(), budget_seconds=budget, seed=0)
    assert sol.wall_clock_seconds <= budget * 1.3, (
        f"wall {sol.wall_clock_seconds:.3f}s > {budget*1.3:.3f}s budget"
    )


def test_solve_auto_with_fast_construct_finite_cost():
    """solve_auto(construction='fast_construct') must produce a finite
    operational_cost on a small instance. N=100 trips solve_auto into
    the warm path (warmstart_low default = 100)."""
    inst = generate(N=100, seed=11)
    sol = solve_auto(inst, Settings(), budget_seconds=8.0,
                     construction="fast_construct", seed=0)
    cost = float(sol.metrics["operational_cost"])
    assert math.isfinite(cost), f"non-finite cost: {cost}"
    assert sol.metrics["num_vehicles_used"] > 0


def test_solve_auto_pyvrp_backward_compat():
    """Default construction='pyvrp' with a fixed seed dispatches through
    the pyvrp warm path. Bit-identity across two runs is wall-clock
    dependent (bandit refinement is non-deterministic across wall
    drift), so we assert the path is reachable and the costs are
    finite + within a relative tolerance — i.e. the construction kwarg
    didn't break the existing dispatch.
    Companion to test_portfolio_basin_jump.* which has the same caveat.
    """
    inst = generate(N=100, seed=23)
    s = Settings()
    a = solve_auto(inst, s, budget_seconds=6.0,
                   construction="pyvrp", seed=42)
    b = solve_auto(inst, s, budget_seconds=6.0,
                   construction="pyvrp", seed=42)
    for sol in (a, b):
        assert math.isfinite(sol.metrics["operational_cost"])
        assert sol.solver == "portfolio_pyvrp_warm"
    # Costs should be in the same neighborhood (within 25% — generous
    # to absorb the bandit's wall-clock variance).
    rel = abs(a.metrics["operational_cost"] - b.metrics["operational_cost"]) \
        / max(a.metrics["operational_cost"], 1e-6)
    assert rel < 0.25, (
        f"pyvrp construction dispatch broken? a={a.metrics['operational_cost']:.2f} "
        f"b={b.metrics['operational_cost']:.2f} rel={rel:.3f}"
    )


def test_fast_construct_polish_reduces_crossings():
    """iter-5g — fast_construct(polish=True) should produce strictly
    fewer `inter_route_crossings` than the same call with polish=False
    on a non-trivial instance. Compares via `score_solution()`.

    The polish step (cost-driven 2-opt* + crossing-aware swap) is the
    main quality lever vs the cost-only multi-start picker.
    """
    inst = generate(N=100, seed=13)
    s = Settings()
    sol_polish = fc.solve(inst, s, budget_seconds=2.0, seed=0, polish=True)
    sol_nopolish = fc.solve(inst, s, budget_seconds=2.0, seed=0, polish=False)
    cross_polish = score_solution(inst, sol_polish).inter_route_crossings
    cross_nopolish = score_solution(inst, sol_nopolish).inter_route_crossings
    assert cross_polish < cross_nopolish, (
        f"polish failed to reduce crossings: "
        f"polish={cross_polish} vs no-polish={cross_nopolish}"
    )


def test_fast_construct_n200_wall_under_3s():
    """iter-5g — fast_construct.solve(budget=2.0) must wall under 3s
    even at N=200. Catches regressions in the regret-k / sampled-regret
    fallback for large instances.
    """
    inst = generate(N=200, seed=29)
    s = Settings()
    sol = fc.solve(inst, s, budget_seconds=2.0, seed=0)
    assert sol.wall_clock_seconds < 3.0, (
        f"N=200 wall {sol.wall_clock_seconds:.2f}s >= 3.0s budget"
    )


def test_portfolio_quality_weight_finite_and_q_at_least_baseline():
    """iter-5g — pm.solve(quality_weight=1.0) must (a) accept the
    kwarg, (b) produce a finite-cost solution, and (c) yield a
    quality_index NO LOWER than the same call with quality_weight=0.0
    when seed identical.

    Generous tolerance: the bandit's reward shaping is a soft signal,
    so we allow a 1e-6 floor below baseline (numerical noise from
    accept-path tiebreaks). This guards against the kwarg accidentally
    *worsening* solutions.
    """
    inst = generate(N=50, seed=37)
    s = Settings()
    a = pm.solve(inst, s, budget_seconds=2.0, seed=11, quality_weight=0.0)
    b = pm.solve(inst, s, budget_seconds=2.0, seed=11, quality_weight=1.0)
    assert math.isfinite(a.metrics["operational_cost"])
    assert math.isfinite(b.metrics["operational_cost"])
    qa = score_solution(inst, a).quality_index
    qb = score_solution(inst, b).quality_index
    # Quality should not regress (1e-6 numerical floor for tiebreaks).
    assert qb >= qa - 1e-6, (
        f"quality_weight=1.0 hurt quality: q(qw=1)={qb:.6f} < q(qw=0)={qa:.6f}"
    )
