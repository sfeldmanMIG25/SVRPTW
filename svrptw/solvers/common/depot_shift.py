"""iter-6a-5-bis -- depot_shift operator.

Greedy local-search over per-route depot_idx. For each active route,
tries each available depot index (only when inst.depots is set with
multiple entries) and keeps the assignment that minimises cost.

Completes the meta-recipe trilogy (iter-5w-bis shift_start +
iter-6a-7-bis class_shift + iter-6a-5-bis depot_shift): for any
constraint axis that's auto-decided by the evaluator OR plumbed via a
Route field, the canonical Step 0 remediation is a (Route field +
dedicated operator) pair.

No-op when inst.depots is None or has only one depot.
"""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common.solution import Solution, evaluate


def depot_shift(
    inst: Instance, sol: Solution, settings: Settings,
    *, max_seconds: float = 1.0,
) -> Solution:
    """Try each depot_idx for each active route; keep best per route."""
    if inst.depots is None or len(inst.depots) <= 1:
        return sol  # no choice to make
    n_depots = len(inst.depots)
    deadline = time.perf_counter() + max_seconds
    best_cost = float(evaluate(inst, sol, settings)["operational_cost"])
    for r in sol.routes:
        if time.perf_counter() >= deadline:
            break
        if not r.customers:
            continue
        cur = int(getattr(r, "depot_idx", 0))
        local_best = cur
        local_best_cost = best_cost
        for di in range(n_depots):
            if di == cur:
                continue
            r.depot_idx = di
            c = float(evaluate(inst, sol, settings)["operational_cost"])
            if c < local_best_cost - 1e-9:
                local_best_cost = c
                local_best = di
        r.depot_idx = local_best
        best_cost = local_best_cost
    sol.metrics = evaluate(inst, sol, settings)
    return sol
