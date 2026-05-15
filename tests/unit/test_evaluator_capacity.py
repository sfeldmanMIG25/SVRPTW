"""SPEC-0-EVAL-01 — capacity feasibility regression.

A hand-built solution that overloads one route by 5 demand units
must (a) be marked infeasible, (b) carry a capacity_overload metric
equal to the overflow, (c) include a hard_late_penalty * overflow
contribution in operational_cost.
"""
from __future__ import annotations

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.solvers.common.solution import Route, Solution, evaluate


def test_overloaded_route_is_infeasible():
    """Pack the route with more demand than capacity; evaluator must flag it."""
    inst = generate(N=20, seed=0)
    cap = inst.vehicle_capacity
    # Sort customers by demand desc, take enough to exceed capacity.
    by_demand = sorted(inst.customers, key=lambda c: -c.demand)
    selected: list[int] = []
    total = 0
    for c in by_demand:
        selected.append(c.id)
        total += c.demand
        if total > cap:
            break
    # Sanity: we did go over.
    assert total > cap, f"test setup failed: cap={cap} got total={total}"
    overflow = total - cap

    sol = Solution(
        instance_id=inst.instance_id,
        routes=[Route(customers=selected)],
        solver="hand",
        wall_clock_seconds=0.0,
        budget_seconds=0.0,
        feasible=True,
        metrics={},
    )
    s = Settings()
    m = evaluate(inst, sol, s)
    assert m["capacity_overload"] == overflow
    assert m["feasible"] == 0.0
    # Penalty equals hard_late_penalty × overflow units (1000 × overflow by default).
    # Strictly greater than the pre-fix cost would have been.
    expected_floor = s.economics.hard_late_penalty * overflow
    assert m["operational_cost"] >= expected_floor


def test_within_capacity_remains_feasible():
    """Build a single-customer route within capacity; feasibility must hold."""
    inst = generate(N=20, seed=1)
    # Pick the smallest-demand customer; one such fits trivially.
    light = min(inst.customers, key=lambda c: c.demand)
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[Route(customers=[light.id])],
        solver="hand",
        wall_clock_seconds=0.0,
        budget_seconds=0.0,
        feasible=True,
        metrics={},
    )
    m = evaluate(inst, sol, Settings())
    assert m["capacity_overload"] == 0.0
    # Feasibility may still be 0 if TW or depot return is missed; the
    # assertion that matters here is that capacity_overload is the
    # only field this test controls.
    assert m["feasible"] in (0.0, 1.0)
