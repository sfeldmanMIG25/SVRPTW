"""iter-6a-7-bis -- class_shift operator.

Greedy local-search over per-route vehicle_class_idx. For each active
route, tries each class index (and -1 = auto) and keeps the assignment
that minimises total cost. Unblocks iter-6a-7 skills: explicit
class assignment lets the bandit upgrade a route to a high-skill class
even when capacity-greedy auto-assignment would pick a low-skill class.

Step 0 met: operator now exists for the per-route-class axis.
"""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common.solution import Solution, evaluate


def class_shift(
    inst: Instance, sol: Solution, settings: Settings,
    *, max_seconds: float = 1.0,
) -> Solution:
    """Try each class index for each active route; keep best per route."""
    t0 = time.perf_counter()
    deadline = t0 + max_seconds
    # Available class indices depend on Settings.economics.vehicle_class_capacities.
    n_classes = len(settings.economics.vehicle_class_capacities)
    if n_classes == 0:
        return sol  # no classes defined; nothing to shift

    # Candidate set: -1 (auto / smallest-viable) plus each explicit index.
    candidates = [-1] + list(range(n_classes))

    best_cost = float(evaluate(inst, sol, settings)["operational_cost"])
    for ri, r in enumerate(sol.routes):
        if time.perf_counter() >= deadline:
            break
        if not r.customers:
            continue
        cur = int(getattr(r, "vehicle_class_idx", -1))
        local_best = cur
        local_best_cost = best_cost
        for ci in candidates:
            if ci == cur:
                continue
            r.vehicle_class_idx = ci
            c = float(evaluate(inst, sol, settings)["operational_cost"])
            if c < local_best_cost - 1e-9:
                local_best_cost = c
                local_best = ci
        # Commit
        r.vehicle_class_idx = local_best
        best_cost = local_best_cost

    sol.metrics = evaluate(inst, sol, settings)
    return sol
