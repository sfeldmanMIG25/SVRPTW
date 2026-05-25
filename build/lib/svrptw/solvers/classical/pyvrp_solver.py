"""PyVRP 0.9+ HGS solver wrapper for asymmetric VRPTW.

PyVRP won 2024 DIMACS VRPTW; this gives us a real SOTA bar to compete
against.  Note: PyVRP minimizes its own (distance + duration + penalty)
objective; our `evaluate()` re-scores the resulting routes under our
canonical cost model so the leaderboard is apples-to-apples.

Prize-collecting mode is enabled (`required=False`, prize=miss_penalty)
so PyVRP can drop customers when serving them costs more than the prize
— the LKH-3 trick.
"""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate

_SCALE = 100   # 1/100-minute resolution, matches OR-Tools wrapper


def _have_pyvrp() -> bool:
    try:
        import pyvrp  # noqa: F401
        return True
    except Exception:
        return False


def solve(inst: Instance, settings: Settings, budget_seconds: float = 10.0) -> Solution:
    t0 = time.perf_counter()
    if not _have_pyvrp():
        # Fallback: empty solution with a marker error metric.
        sol = Solution(
            instance_id=inst.instance_id, routes=[], solver="pyvrp_hgs",
            wall_clock_seconds=time.perf_counter() - t0,
            budget_seconds=float(budget_seconds), feasible=False,
        )
        sol.metrics = {"error": -1, "operational_cost": float("inf"),
                       "missed_deliveries": float(inst.num_customers),
                       "feasible": 0.0}
        return sol

    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    m = Model()
    # Coordinates as ints (PyVRP works in integer space).  Scale lat/lon
    # generously since we don't actually use Euclidean for routing.
    cx = int(inst.depot.x * 1_000_000)
    cy = int(inst.depot.y * 1_000_000)
    depot = m.add_depot(x=cx, y=cy)
    m.add_vehicle_type(
        num_available=inst.num_vehicles,
        capacity=[int(inst.vehicle_capacity)],
        start_depot=depot,
        end_depot=depot,
        tw_early=int(inst.depot.ready * _SCALE),
        tw_late=int(inst.depot.due * _SCALE),
        unit_distance_cost=int(settings.economics.cost_per_mile * 100),
        unit_duration_cost=int(settings.economics.wage_per_minute * 100),
    )

    clients = []
    # Every customer is REQUIRED. Prize-collecting (required=False) let
    # PyVRP drop customers under time pressure, inflating the bench mean
    # by ~10× because each drop costs hard_late_penalty in our evaluator.
    # See bench/figures/portfolio_vs_pyvrp_n50_v1.md (2026-05-13).
    for c in inst.customers:
        cl = m.add_client(
            x=int(c.x * 1_000_000),
            y=int(c.y * 1_000_000),
            delivery=[int(c.demand)],
            service_duration=int(c.service * _SCALE),
            tw_early=int(c.ready * _SCALE),
            tw_late=int(c.due * _SCALE),
            required=True,
        )
        clients.append(cl)

    nodes = [depot] + clients   # index 0 = depot, 1..N = customers
    T = inst.travel_time
    D = inst.travel_dist
    N = inst.num_customers
    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                continue
            m.add_edge(
                nodes[i], nodes[j],
                distance=int(D[i, j] * 100),
                duration=int(T[i, j] * _SCALE),
            )

    try:
        res = m.solve(stop=MaxRuntime(max(1.0, budget_seconds)),
                      seed=int(settings.seed), display=False)
    except Exception as e:
        sol = Solution(
            instance_id=inst.instance_id, routes=[], solver="pyvrp_hgs",
            wall_clock_seconds=time.perf_counter() - t0,
            budget_seconds=float(budget_seconds), feasible=False,
        )
        sol.metrics = {"error": -1, "operational_cost": float("inf"),
                       "missed_deliveries": float(inst.num_customers),
                       "feasible": 0.0, "error_msg": repr(e)}
        return sol

    # Extract routes — PyVRP's client indices map 1:1 to our 1..N.
    best = res.best
    routes: list[Route] = []
    for r in best.routes():
        visits = list(r.visits()) if hasattr(r, "visits") else list(r)
        # Visits should already be in our customer-id space (1..N).
        # Filter the depot just in case.
        cust_ids = [int(v) for v in visits if int(v) != 0]
        if cust_ids:
            routes.append(Route(customers=cust_ids))

    sol = Solution(
        instance_id=inst.instance_id, routes=routes, solver="pyvrp_hgs",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=float(budget_seconds), feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    return sol


if __name__ == "__main__":
    import argparse
    import json

    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=10.0)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), args.budget)
    print(json.dumps({"solver": sol.solver, "metrics": sol.metrics,
                      "wall_clock_s": sol.wall_clock_seconds}, indent=2))
