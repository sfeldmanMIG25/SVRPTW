"""Seed operator: cross-route 2-opt* (Potvin & Rousseau 1995).

For every pair of routes (i, j) and every cut-pair (p, q), reconnect
by exchanging the suffixes: route i = head_i(p) + tail_j(q) and
route j = head_j(q) + tail_i(p). This is the standard CVRPTW
2-opt* operator. Distinct from the existing seeds (relocate-1 and
swap-1) because the move size scales with route length, allowing
long-range untangling that single-customer moves can't reach.

Rationale: routes drawn from greedy construction often cross each
other geometrically. Single relocates can't undo a cross — you need
to cut both routes at the crossing and swap suffixes. This is the
single most-cited LNS operator after the original Lin-Kernighan moves.
"""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.council.proposal import OperatorContext
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.common.local_search import _cust_by_id, _route_arrival_and_close


def operator(solution: Solution, context: OperatorContext) -> Solution | None:
    inst: Instance = context.instance
    settings: Settings = context.settings
    deadline = time.perf_counter() + context.deadline_seconds
    cust_by_id = _cust_by_id(inst)
    cap = float(inst.vehicle_capacity) if inst.vehicle_capacity else float("inf")

    routes = [r for r in solution.routes if r.customers]
    if len(routes) < 2:
        return None

    best_cost = solution.metrics["operational_cost"]
    best_sol: Solution | None = None

    n_routes = len(solution.routes)
    for ri in range(n_routes):
        if time.perf_counter() >= deadline:
            break
        r_i = solution.routes[ri].customers
        if not r_i:
            continue
        for rj in range(ri + 1, n_routes):
            if time.perf_counter() >= deadline:
                break
            r_j = solution.routes[rj].customers
            if not r_j:
                continue
            # Cut points p in [0..len(r_i)], q in [0..len(r_j)].
            # p == 0 and q == 0 reproduces the input (skip); p == len
            # and q == len also reproduces (skip).
            for p in range(0, len(r_i) + 1):
                for q in range(0, len(r_j) + 1):
                    if (p == 0 and q == 0) or (p == len(r_i) and q == len(r_j)):
                        continue
                    new_i = r_i[:p] + r_j[q:]
                    new_j = r_j[:q] + r_i[p:]
                    # Capacity pre-filter.
                    load_i = sum(cust_by_id[c].demand for c in new_i)
                    load_j = sum(cust_by_id[c].demand for c in new_j)
                    if load_i > cap or load_j > cap:
                        continue
                    # Feasibility pre-filter (TW).
                    ok_i, _ = _route_arrival_and_close(inst, new_i)
                    ok_j, _ = _route_arrival_and_close(inst, new_j)
                    if not (ok_i and ok_j):
                        continue
                    new_routes: list[Route] = []
                    for k, r in enumerate(solution.routes):
                        if k == ri:
                            new_routes.append(Route(customers=new_i))
                        elif k == rj:
                            new_routes.append(Route(customers=new_j))
                        else:
                            new_routes.append(Route(customers=list(r.customers)))
                    cand = Solution(
                        instance_id=inst.instance_id, routes=new_routes,
                        solver=solution.solver,
                        wall_clock_seconds=solution.wall_clock_seconds,
                        budget_seconds=solution.budget_seconds, feasible=False,
                    )
                    cand.metrics = evaluate(inst, cand, settings)
                    cand.feasible = bool(cand.metrics["feasible"])
                    if cand.metrics.get("capacity_overload", 0.0) > 0.0:
                        continue
                    if cand.metrics["operational_cost"] < best_cost - 1e-6:
                        best_cost = cand.metrics["operational_cost"]
                        best_sol = cand
    return best_sol
