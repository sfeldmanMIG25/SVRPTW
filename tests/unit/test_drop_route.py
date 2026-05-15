"""SPEC-7-OPS-DESTROY-01 — drop_route invariants."""
from __future__ import annotations

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.solvers.classical import greedy as greedy_mod
from svrptw.solvers.common.local_search_destroy import drop_route


def test_drop_route_preserves_or_returns_input():
    """drop_route must either improve cost or return the input verbatim."""
    inst = generate(N=20, seed=0)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    base_cost = sol.metrics["operational_cost"]

    cand = drop_route(inst, sol, s, max_seconds=0.5, rng_seed=42)
    assert cand.metrics["operational_cost"] <= base_cost + 1e-6


def test_drop_route_noop_when_one_route():
    """Single-route solution can't drop; return unchanged."""
    inst = generate(N=10, seed=1)
    s = Settings()
    # Construct a degenerate single-route solution.
    from svrptw.solvers.common.solution import Route, Solution, evaluate
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[Route(customers=[c.id for c in inst.customers[:5]])],
        solver="hand",
        wall_clock_seconds=0.0,
        budget_seconds=0.0,
        feasible=True,
        metrics={},
    )
    sol.metrics = evaluate(inst, sol, s)
    out = drop_route(inst, sol, s)
    assert out is sol


def test_drop_route_respects_capacity_on_reinsertion():
    """After drop_route, no resulting route may exceed vehicle capacity."""
    inst = generate(N=30, seed=2)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    cand = drop_route(inst, sol, s, max_seconds=1.0, rng_seed=7)
    # capacity_overload field is added by SPEC-0-EVAL-01; must be 0.
    assert cand.metrics.get("capacity_overload", 0.0) == 0.0


# ---- destroy_island tests --------------------------------------------------

from svrptw.solvers.common.local_search_destroy import destroy_island


def test_destroy_island_no_regret():
    """destroy_island must either improve or return the input verbatim."""
    inst = generate(N=30, seed=10)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    base = sol.metrics["operational_cost"]
    cand = destroy_island(inst, sol, s, max_seconds=1.0, rng_seed=1, island_size=5)
    assert cand.metrics["operational_cost"] <= base + 1e-6


def test_destroy_island_capacity_safe():
    """Reinsertion path must respect capacity."""
    inst = generate(N=40, seed=11)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    cand = destroy_island(inst, sol, s, max_seconds=1.0, rng_seed=2, island_size=6)
    assert cand.metrics.get("capacity_overload", 0.0) == 0.0


def test_destroy_island_noop_when_tiny_instance():
    """Too few customers to form an island → return input."""
    inst = generate(N=5, seed=12)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    out = destroy_island(inst, sol, s, max_seconds=0.2, rng_seed=3, island_size=8)
    # input had 5 customers; island_size=8 → can't form → return unchanged
    assert out is sol


# ---- drop_leg tests --------------------------------------------------------

from svrptw.solvers.common.local_search_destroy import drop_leg


def test_drop_leg_no_regret():
    inst = generate(N=30, seed=20)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    base = sol.metrics["operational_cost"]
    cand = drop_leg(inst, sol, s, max_seconds=0.5, rng_seed=4)
    assert cand.metrics["operational_cost"] <= base + 1e-6


def test_drop_leg_capacity_safe():
    inst = generate(N=40, seed=21)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    cand = drop_leg(inst, sol, s, max_seconds=0.5, rng_seed=5)
    assert cand.metrics.get("capacity_overload", 0.0) == 0.0


def test_drop_leg_noop_when_no_route_has_two_customers():
    """If every route has < 2 customers, drop_leg can't form a leg."""
    inst = generate(N=10, seed=22)
    s = Settings()
    from svrptw.solvers.common.solution import Route, Solution, evaluate
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[Route(customers=[c.id]) for c in inst.customers[:3]],
        solver="hand",
        wall_clock_seconds=0.0,
        budget_seconds=0.0,
        feasible=True,
        metrics={},
    )
    sol.metrics = evaluate(inst, sol, s)
    out = drop_leg(inst, sol, s, max_seconds=0.2)
    assert out is sol
