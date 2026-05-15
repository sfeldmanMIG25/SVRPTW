"""Seed operator: TW-anchor relocate.

Pick the customer with the tightest TW slack in any route; try
inserting it as the *first* customer of every other route. This
mimics what dispatchers do by hand — anchor the schedule around
the most-constrained customer first.

Rationale: the customer with the tightest TW is the hardest to
fit. If it's currently in the middle of a route, swapping it to
the start of an empty-leading route gives it maximum slack and
lets the rest of the schedule fall into place.
"""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.council.proposal import OperatorContext
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.common.local_search import _cust_by_id, _route_arrival_and_close


def operator(solution: Solution, context: OperatorContext) -> Solution | None:
    """SPEC-8-COUNCIL-01 seed: TW-anchor relocate.

    Returns the improved solution or None if no candidate placement
    is strictly better. Respects context.deadline_seconds.
    """
    inst: Instance = context.instance
    settings: Settings = context.settings
    deadline = time.perf_counter() + context.deadline_seconds
    cust_by_id = _cust_by_id(inst)

    # Find tightest-TW customer (smallest due - ready) currently assigned.
    served: list[tuple[int, int, int]] = []   # (tw_width, route_idx, pos)
    for ri, r in enumerate(solution.routes):
        for pos, cid in enumerate(r.customers):
            c = cust_by_id[cid]
            served.append((c.due - c.ready, ri, pos))
    if not served:
        return None
    served.sort()
    # Take the top-3 tightest; one of them might be relocatable.
    candidates = served[:3]

    best_cost = solution.metrics["operational_cost"]
    best_sol: Solution | None = None

    for tw_width, src_ri, src_pos in candidates:
        if time.perf_counter() >= deadline:
            break
        cid = solution.routes[src_ri].customers[src_pos]
        # Try inserting at position 0 of every other route.
        for dst_ri in range(len(solution.routes)):
            if dst_ri == src_ri:
                continue
            new_routes: list[Route] = []
            for ri, r in enumerate(solution.routes):
                seq = list(r.customers)
                if ri == src_ri:
                    seq = seq[:src_pos] + seq[src_pos + 1:]
                if ri == dst_ri:
                    seq = [cid] + seq
                new_routes.append(Route(customers=seq))
            # Quick feasibility check on the destination route only.
            ok, _ = _route_arrival_and_close(inst, new_routes[dst_ri].customers)
            if not ok:
                continue
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
