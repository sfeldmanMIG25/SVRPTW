"""POMO+EAS-as-operator: a bandit arm that runs short EAS refinement.

The bandit can call this arm with a per-op time budget. EAS-Emb runs
for the budget, returns its best rollout. If the result improves the
input cost, the bandit gets positive reward; otherwise the input is
returned unchanged (no-regret contract).

The bandit's LinUCB exploration will discover instance/state contexts
where EAS adds value beyond classical operators, and prefer it there.
If it never helps, the bandit converges to ignoring it.

Bounded EAS iterations make this fast enough to be a viable arm:
  per_op_seconds=1.5 → ~5-15 EAS iterations at N=50, ~2-5 at N=200.
"""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Solution


def eas_operator(
    inst: Instance, sol: Solution, settings: Settings,
    max_seconds: float = 1.5,
) -> Solution:
    """Bandit-arm signature: (inst, sol, settings, max_seconds) -> Solution.

    Calls POMO+EAS-Emb with a small iteration budget bounded by
    max_seconds. Returns the input solution unchanged if EAS fails
    to improve or the EAS module/checkpoint is unavailable.
    """
    try:
        from svrptw.solvers.learning.pomo.eas import solve_eas
    except Exception:
        return sol
    t0 = time.perf_counter()
    # Map wall budget to iteration count. EAS-Emb per-iter cost scales
    # with N; conservative estimate ~0.4s/iter at N=50, ~1.5s/iter at
    # N=200. Cap at 50 iters so we don't overrun for tiny budgets.
    if inst.num_customers <= 60:
        iters_per_s = 2.0
    elif inst.num_customers <= 120:
        iters_per_s = 1.0
    else:
        iters_per_s = 0.5
    n_iter = max(2, min(50, int(max_seconds * iters_per_s)))
    try:
        cand = solve_eas(
            inst, settings,
            n_iterations=n_iter,
            n_starts=min(16, inst.num_customers),  # smaller K for speed
            log_every=0,
            max_wall_seconds=max_seconds,
        )
    except Exception:
        return sol
    if cand is None or "operational_cost" not in cand.metrics:
        return sol
    if cand.metrics["operational_cost"] < sol.metrics["operational_cost"] - 1e-6:
        return cand
    return sol
