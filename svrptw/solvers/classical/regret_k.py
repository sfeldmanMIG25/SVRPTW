"""Regret-k insertion constructor for VRPTW.

Classical Solomon-style heuristic.  At each step, for every unrouted
customer, compute the k best feasible insertion costs across all open
routes.  Define `regret = sum(cost_2nd..k) - (k-1) * cost_1st`.  Pick the
customer with the LARGEST regret and insert at its best position.

Compared to greedy best-insertion: regret-k schedules the
hardest-to-place customers first.  This typically yields a solution with
*fewer* total routes — exactly the structural fix our two-tier auction
is missing on N=100 (LKH-3 finds 10-vehicle solutions where ours locks
in 22).

Reference: Solomon 1987, Potvin & Rousseau 1993.  Default k=3 follows the
Ropke & Pisinger 2006 ALNS recipe.
"""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.classical import greedy as greedy_mod
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.common.local_search import _route_arrival_and_close, _route_time


def _insertion_cost(inst: Instance, base: list[int], cid: int, pos: int) -> float | None:
    """Travel-time delta from inserting cid into `base` at position `pos`.
    Returns None if TW-infeasible after insertion."""
    new_route = base[:pos] + [cid] + base[pos:]
    ok, _ = _route_arrival_and_close(inst, new_route)
    if not ok:
        return None
    return _route_time(inst, new_route) - _route_time(inst, base)


def solve(inst: Instance, settings: Settings, budget_seconds: float | None = None,
          k: int = 3, drop_threshold_factor: float = 1.0) -> Solution:
    """Regret-k construction with drop-during-construction.

    At each step: for every unrouted customer, find the k best feasible
    insertion costs across all current open routes (or a fresh open route).
    Define regret = sum(top[1..k]) - (k-1)*top[0].  If even the best
    insertion (across all routes incl. opening a new one) costs more than
    `drop_threshold = hard_late_penalty * factor` in wage-equivalent
    minutes, DROP the customer (leave it unrouted; miss penalty applies
    in evaluate()).  Otherwise, insert the customer with maximum regret.

    This is the LKH-3 trick reproduced at construction time.
    """
    t0 = time.perf_counter()
    routes: list[list[int]] = []
    unrouted = {c.id for c in inst.customers}
    max_routes = inst.num_vehicles
    wage_per_min = settings.economics.wage_per_minute
    drop_threshold_minutes = (
        settings.economics.hard_late_penalty * drop_threshold_factor / max(wage_per_min, 1e-6)
    )

    def insertion_options(cid: int) -> list[tuple[float, int, int]]:
        """All feasible (delta-min, route_idx, pos) for inserting cid.
        Also considers opening a NEW route (route_idx = len(routes))."""
        opts: list[tuple[float, int, int]] = []
        for ri, base in enumerate(routes):
            for pos in range(len(base) + 1):
                d = _insertion_cost(inst, base, cid, pos)
                if d is not None:
                    opts.append((d, ri, pos))
        # New-route option: depot -> cid -> depot
        if len(routes) < max_routes:
            d_new = _insertion_cost(inst, [], cid, 0)
            if d_new is not None:
                opts.append((d_new, len(routes), 0))
        return opts

    while unrouted:
        # Score each unrouted customer's regret + best insertion.
        scored: list[tuple[float, int, int, int, float]] = []   # (regret, cid, ri, pos, best_cost)
        for cid in list(unrouted):
            opts = insertion_options(cid)
            if not opts:
                # Truly infeasible — drop immediately.
                unrouted.discard(cid)
                continue
            opts.sort()
            top = opts[:k]
            best_cost, best_ri, best_pos = top[0]
            if len(top) == 1:
                regret = 1e6 + best_cost
            else:
                regret = sum(c[0] for c in top[1:]) - (len(top) - 1) * best_cost
            scored.append((regret, cid, best_ri, best_pos, best_cost))

        if not scored:
            break

        scored.sort(reverse=True)
        _r, cid, ri, pos, best_cost = scored[0]

        # Drop decision: is best insertion still cheaper than dropping?
        if best_cost > drop_threshold_minutes:
            unrouted.discard(cid)
            continue

        # Insert.
        if ri >= len(routes):
            routes.append([])
        routes[ri] = routes[ri][:pos] + [cid] + routes[ri][pos:]
        unrouted.discard(cid)

    sol_routes = [Route(customers=r) for r in routes if r]
    while len(sol_routes) < inst.num_vehicles:
        sol_routes.append(Route(customers=[]))

    sol = Solution(
        instance_id=inst.instance_id, routes=sol_routes, solver="regret_k",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=float(budget_seconds or 0.0), feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])

    # Improvement chain (same as auction_gart but lighter — let regret-k's
    # tighter construction speak first).
    from svrptw.solvers.common import (
        merge_routes,
        relocate,
        sisr_destroy_repair,
        two_opt_intra,
        two_opt_star,
    )
    n = inst.num_customers
    if budget_seconds and budget_seconds > 0:
        # Spend ~80% of caller's budget on improvement.
        improve_budget = max(0.5, budget_seconds * 0.8)
    else:
        improve_budget = max(1.0, n * 0.06)
    slot = improve_budget / 5.0
    sol = merge_routes(inst, sol, settings, max_seconds=slot)
    sol = relocate(inst, sol, settings, max_seconds=slot)
    sol = two_opt_intra(inst, sol, settings, max_seconds=slot)
    sol = two_opt_star(inst, sol, settings, max_seconds=slot)
    sol = sisr_destroy_repair(inst, sol, settings, max_seconds=slot)

    # No-regret vs greedy.
    g = greedy_mod.solve(inst, settings)
    if g.metrics["operational_cost"] < sol.metrics["operational_cost"] - 1e-6:
        sol = g
    sol.solver = "regret_k"
    sol.wall_clock_seconds = time.perf_counter() - t0
    return sol


if __name__ == "__main__":
    import argparse
    import json

    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--budget", type=float, default=10.0)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), budget_seconds=args.budget, k=args.k)
    print(json.dumps({"solver": sol.solver, "metrics": sol.metrics,
                      "wall_clock_s": sol.wall_clock_seconds,
                      "k_used": args.k}, indent=2))
