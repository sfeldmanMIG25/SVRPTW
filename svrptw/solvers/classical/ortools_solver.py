"""OR-Tools CP-SAT routing solver for asymmetric VRPTW.

Single budget point per call; the bench harness iterates {1, 10, 60} seconds.
"""
from __future__ import annotations

import time

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate

# Scale travel times to integers (CP solver wants ints).
_SCALE = 100  # 1/100 minute resolution
_BIG = 10**9


def solve(inst: Instance, settings: Settings, budget_seconds: float = 10.0) -> Solution:
    t0 = time.perf_counter()
    N = inst.num_customers
    K = inst.num_vehicles
    T = (inst.travel_time * _SCALE).round().astype(np.int64)
    services = np.array([0] + [c.service * _SCALE for c in inst.customers], dtype=np.int64)
    demands  = np.array([0] + [c.demand for c in inst.customers], dtype=np.int64)
    ready    = np.array([inst.depot.ready * _SCALE] + [c.ready * _SCALE for c in inst.customers], dtype=np.int64)
    due      = np.array([inst.depot.due * _SCALE]   + [c.due   * _SCALE for c in inst.customers], dtype=np.int64)

    # Pre-detect customers that cannot possibly be visited within their TWs from
    # the depot (round-trip exceeds the depot day, OR depot->cust travel exceeds
    # the customer's TW close).  These are forced-drop via a singleton disjunction.
    depot_due = int(due[0])
    forced_drop: list[int] = []
    for i in range(1, N + 1):
        depot_to = int(T[0, i])
        back     = int(T[i, 0])
        earliest_arrive = depot_to
        latest_start    = max(earliest_arrive, int(ready[i]))
        end_at_depot    = latest_start + int(services[i]) + back
        if earliest_arrive > int(due[i]) or end_at_depot > depot_due:
            forced_drop.append(i)

    manager = pywrapcp.RoutingIndexManager(N + 1, K, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(i, j):
        a = manager.IndexToNode(i)
        b = manager.IndexToNode(j)
        return int(T[a, b] + services[a])
    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_cb(i):
        a = manager.IndexToNode(i)
        return int(demands[a])
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(inst.vehicle_capacity)] * K, True, "Capacity")

    # Disjunctions FIRST so SetRange propagation can fall back to dropping the node
    # if its time window is infeasible.
    drop_penalty = int(settings.economics.hard_late_penalty * _SCALE)
    forced_set = set(forced_drop)
    for node in range(1, N + 1):
        idx = manager.NodeToIndex(node)
        # Pre-detected infeasibles get a much lower penalty so the solver will
        # always pick the drop branch.  Newer OR-Tools rejects max_cardinality=0,
        # so we encode "forced drop" as economically dominant.
        penalty = 1 if node in forced_set else drop_penalty
        routing.AddDisjunction([idx], penalty)

    # Time dimension
    horizon = int(due.max() + services.max())
    routing.AddDimension(transit_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(N + 1):
        if node in forced_set:
            continue
        idx = manager.NodeToIndex(node)
        try:
            time_dim.CumulVar(idx).SetRange(int(ready[node]), int(due[node]))
        except Exception:
            # Propagation couldn't fit this node; the disjunction already allows
            # the solver to drop it.  If it routes anyway, evaluate() treats any
            # post-hoc TW violation as a missed delivery (correct operational
            # semantics for VRPTW).
            pass
    for v in range(K):
        try:
            time_dim.CumulVar(routing.Start(v)).SetRange(int(ready[0]), int(due[0]))
        except Exception:
            pass

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(max(1, budget_seconds)))

    assignment = routing.SolveWithParameters(params)
    elapsed = time.perf_counter() - t0

    routes: list[Route] = []
    if assignment:
        for v in range(K):
            r = Route()
            idx = routing.Start(v)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != 0:
                    r.customers.append(node)
                idx = assignment.Value(routing.NextVar(idx))
            routes.append(r)
    else:
        routes = [Route() for _ in range(K)]

    sol = Solution(
        instance_id=inst.instance_id,
        routes=routes,
        solver="ortools",
        wall_clock_seconds=elapsed,
        budget_seconds=float(budget_seconds),
        feasible=bool(assignment is not None),
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
    print(json.dumps({"solver": sol.solver, "budget_s": sol.budget_seconds,
                      "metrics": sol.metrics, "wall_clock_s": sol.wall_clock_seconds}, indent=2))
