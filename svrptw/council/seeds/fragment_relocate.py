"""Seed operator: fragment-relocate (multi-customer contiguous slice move).

Extract a contiguous fragment of length k ∈ {2, 3} from any route and
try inserting it as a single block at every position of every other
route. This is structurally distinct from PyVRP's relocate-1 (single
customer) and 2-opt-star (cuts at ONE point per route, reattaches
whole tails). Fragment-relocate moves a MIDDLE slice as a unit.

Why this can complement PyVRP at production budget: PyVRP's relocate
loop visits single customers. Reaching a state where a contiguous
3-tuple should move to another route requires three independent
relocate steps that all happen to find improving moves. With the
3-tuple held together this becomes a single one-step move.

Rationale: tight TW chains (customer i → i+1 forced by overlapping
windows) want to move as a block. Single relocate breaks the chain
and probably fails feasibility. Fragment-relocate preserves it.
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

    routes = solution.routes
    if sum(1 for r in routes if r.customers) < 2:
        return None

    best_cost = solution.metrics["operational_cost"]
    best_sol: Solution | None = None

    for k in (2, 3):
        if time.perf_counter() >= deadline:
            break
        for src_ri, src_r in enumerate(routes):
            if time.perf_counter() >= deadline:
                break
            cs = src_r.customers
            if len(cs) < k:
                continue
            for start in range(0, len(cs) - k + 1):
                if time.perf_counter() >= deadline:
                    break
                fragment = cs[start:start + k]
                remainder = cs[:start] + cs[start + k:]
                # Quick TW + capacity on the donor route after removal.
                ok_src, _ = _route_arrival_and_close(inst, remainder)
                if not ok_src:
                    continue
                frag_demand = sum(cust_by_id[c].demand for c in fragment)
                for dst_ri in range(len(routes)):
                    if dst_ri == src_ri:
                        continue
                    dst_cs = routes[dst_ri].customers
                    # Capacity pre-filter.
                    dst_load = sum(cust_by_id[c].demand for c in dst_cs)
                    if dst_load + frag_demand > cap:
                        continue
                    for pos in range(len(dst_cs) + 1):
                        new_dst = dst_cs[:pos] + fragment + dst_cs[pos:]
                        ok_dst, _ = _route_arrival_and_close(inst, new_dst)
                        if not ok_dst:
                            continue
                        new_routes: list[Route] = []
                        for ri, r in enumerate(routes):
                            if ri == src_ri:
                                new_routes.append(Route(customers=list(remainder)))
                            elif ri == dst_ri:
                                new_routes.append(Route(customers=new_dst))
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
