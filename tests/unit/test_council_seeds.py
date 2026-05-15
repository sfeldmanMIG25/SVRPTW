"""SPEC-8-COUNCIL-01 — cold-start seed operator tests.

These operators are exemplars the agent council reads. Each must:
  - conform to the (solution, context) signature
  - preserve feasibility (no capacity overload, no missed deliveries
    introduced unless they were already present)
  - never return a strictly worse-cost result (no-regret)
  - be idempotent on already-tight solutions (return None)
"""
from __future__ import annotations

from svrptw.config import Settings
from svrptw.council.proposal import OperatorContext
from svrptw.council.seeds.capacity_rebalance_swap import operator as cap_rebalance
from svrptw.council.seeds.cross_route_2opt import operator as cross_2opt
from svrptw.council.seeds.tw_anchor_relocate import operator as tw_anchor
from svrptw.instances_gen.synthetic import generate
from svrptw.solvers.classical import greedy as greedy_mod


def _ctx(inst, deadline=1.0):
    return OperatorContext(instance=inst, settings=Settings(),
                           rng_seed=0, deadline_seconds=deadline)


def test_tw_anchor_relocate_no_regret():
    inst = generate(N=20, seed=0)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    cand = tw_anchor(sol, _ctx(inst))
    if cand is not None:
        # Cost must be strictly lower than the input (no-regret).
        assert cand.metrics["operational_cost"] < sol.metrics["operational_cost"]
        assert cand.metrics.get("capacity_overload", 0.0) == 0.0


def test_tw_anchor_relocate_returns_solution_or_none():
    inst = generate(N=15, seed=1)
    sol = greedy_mod.solve(inst, Settings())
    cand = tw_anchor(sol, _ctx(inst))
    # Either None or a Solution with metrics populated.
    from svrptw.solvers.common import Solution
    assert cand is None or (isinstance(cand, Solution)
                             and "operational_cost" in cand.metrics)


def test_capacity_rebalance_swap_no_regret():
    inst = generate(N=30, seed=2)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    cand = cap_rebalance(sol, _ctx(inst))
    if cand is not None:
        assert cand.metrics["operational_cost"] < sol.metrics["operational_cost"]
        assert cand.metrics.get("capacity_overload", 0.0) == 0.0


def test_capacity_rebalance_swap_requires_imbalance():
    """If every route is mid-util, the operator finds no candidates and
    returns None."""
    from svrptw.solvers.common import Route, Solution, evaluate
    inst = generate(N=12, seed=3)
    # Build a single-route degenerate solution — no second route to swap with.
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[Route(customers=[c.id for c in inst.customers[:6]])],
        solver="hand", wall_clock_seconds=0.0, budget_seconds=0.0,
        feasible=True, metrics={},
    )
    sol.metrics = evaluate(inst, sol, Settings())
    out = cap_rebalance(sol, _ctx(inst))
    assert out is None


def test_cross_route_2opt_no_regret():
    """Cross-route 2-opt must never return a strictly worse solution."""
    inst = generate(N=20, seed=5)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    cand = cross_2opt(sol, _ctx(inst))
    if cand is not None:
        assert cand.metrics["operational_cost"] < sol.metrics["operational_cost"]
        assert cand.metrics.get("capacity_overload", 0.0) == 0.0


def test_cross_route_2opt_single_route_returns_none():
    """No second route → no cross-route move available."""
    from svrptw.solvers.common import Route, Solution, evaluate
    inst = generate(N=10, seed=6)
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[Route(customers=[c.id for c in inst.customers[:5]])],
        solver="hand", wall_clock_seconds=0.0, budget_seconds=0.0,
        feasible=True, metrics={},
    )
    sol.metrics = evaluate(inst, sol, Settings())
    assert cross_2opt(sol, _ctx(inst)) is None


def test_seeds_respect_deadline():
    """Operators must not run past context.deadline_seconds (within a small
    grace). Pass 0.05s and assert wall < 0.5s."""
    import time
    inst = generate(N=25, seed=4)
    sol = greedy_mod.solve(inst, Settings())
    for op in (tw_anchor, cap_rebalance, cross_2opt):
        t0 = time.perf_counter()
        op(sol, _ctx(inst, deadline=0.05))
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"{op.__module__} exceeded deadline grace: {elapsed:.2f}s"
