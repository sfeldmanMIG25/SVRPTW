"""Seed operator: capacity-rebalance swap.

Find two routes where one is at > 90 % capacity and the other at
< 50 %; swap the last-inserted customer of the over-utilized one
with the first customer of the under-utilized one. Targets the
under-utilisation the logic-axis judge penalises.

Rationale: routes drift toward extreme utilisation when greedy
insertion picks the cheapest spot regardless of route balance.
A targeted swap pulls one customer from the heavy route into the
light one, smoothing the load profile without churning the rest
of the solution.
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
    cap = max(1.0, float(inst.vehicle_capacity))

    # Per-route util.
    routes_info: list[tuple[float, int]] = []   # (util, route_idx)
    for ri, r in enumerate(solution.routes):
        if not r.customers:
            continue
        load = sum(cust_by_id[c].demand for c in r.customers)
        routes_info.append((load / cap, ri))
    if len(routes_info) < 2:
        return None

    heavy = [(u, ri) for u, ri in routes_info if u > 0.90]
    light = [(u, ri) for u, ri in routes_info if u < 0.50]
    if not heavy or not light:
        return None

    best_cost = solution.metrics["operational_cost"]
    best_sol: Solution | None = None

    for _, h_ri in heavy:
        if time.perf_counter() >= deadline:
            break
        for _, l_ri in light:
            heavy_route = solution.routes[h_ri].customers
            light_route = solution.routes[l_ri].customers
            if not heavy_route or not light_route:
                continue
            # Swap heavy's last customer with light's first.
            h_cid = heavy_route[-1]
            l_cid = light_route[0]
            new_heavy = heavy_route[:-1] + [l_cid]
            new_light = [h_cid] + light_route[1:]
            # Quick feasibility on both reshuffled routes.
            ok_h, _ = _route_arrival_and_close(inst, new_heavy)
            ok_l, _ = _route_arrival_and_close(inst, new_light)
            if not (ok_h and ok_l):
                continue
            new_routes: list[Route] = []
            for ri, r in enumerate(solution.routes):
                if ri == h_ri:
                    new_routes.append(Route(customers=new_heavy))
                elif ri == l_ri:
                    new_routes.append(Route(customers=new_light))
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
