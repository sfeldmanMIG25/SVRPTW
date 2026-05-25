"""iter-5w-bis -- shift_route_start operator.

Greedy local-search over per-route start_offset_minutes. For each active
route, tries a small set of candidate offsets and keeps the offset that
reduces total cost most. This unblocks iter-5x peak_hour: a route whose
default start at depot.ready=480 lands inside the 8-10am peak can shift
to start at 600 (after peak), trading off slightly later customer
arrivals for avoiding the wage surcharge.

Step 0 met: this operator EXISTS now, so per-segment-of-time cost terms
(peak_hour, embargo) have a direct bandit lever for the time-shift axis.
"""
from __future__ import annotations

import time
from copy import deepcopy

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common.solution import Solution, evaluate


# Candidate offsets in minutes. 0 = default; positive = start later.
# Negative offsets aren't supported because depot.ready is the earliest
# legal start; routes starting before depot.ready would arrive at customers
# before they're "open" and incur waiting.
_OFFSETS = (0.0, 30.0, 60.0, 90.0, 120.0)


def shift_route_start(
    inst: Instance, sol: Solution, settings: Settings,
    *, max_seconds: float = 1.0,
) -> Solution:
    """Try each candidate start_offset for each active route; keep the
    combination that minimises operational_cost. Greedy one-route-at-a-time.
    """
    t0 = time.perf_counter()
    deadline = t0 + max_seconds
    best_sol = sol
    best_cost = float(evaluate(inst, sol, settings)["operational_cost"])

    # Walk routes; for each, sweep offsets and keep best. Re-eval AFTER
    # each route change so subsequent decisions see the updated cost.
    n_routes = len(sol.routes)
    for ri in range(n_routes):
        if time.perf_counter() >= deadline:
            break
        if not best_sol.routes[ri].customers:
            continue
        local_best_cost = best_cost
        local_best_offset = float(getattr(best_sol.routes[ri], "start_offset_minutes", 0.0))
        for off in _OFFSETS:
            if off == local_best_offset:
                continue
            # Mutate in-place for cheap eval, then revert if it doesn't win
            old_off = float(getattr(best_sol.routes[ri], "start_offset_minutes", 0.0))
            best_sol.routes[ri].start_offset_minutes = float(off)
            c = float(evaluate(inst, best_sol, settings)["operational_cost"])
            if c < local_best_cost - 1e-9:
                local_best_cost = c
                local_best_offset = float(off)
            # Revert before next trial; we'll commit after the sweep.
            best_sol.routes[ri].start_offset_minutes = old_off
        # Commit the best offset for this route
        best_sol.routes[ri].start_offset_minutes = float(local_best_offset)
        best_cost = local_best_cost

    # Re-score metrics on the final solution so callers see updated cost.
    best_sol.metrics = evaluate(inst, best_sol, settings)
    return best_sol
