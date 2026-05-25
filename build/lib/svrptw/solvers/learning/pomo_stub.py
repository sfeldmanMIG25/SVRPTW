"""POMO-lite construction policy.  SPEC-4-POMO-01.

This is a SCAFFOLD: the encoder, decoder, action mask, training loop, and
shaped-reward hook are all stubbed.  Phase 4 fills these in.  The interface
is designed so the bench harness can already register `pomo` as a solver
name — until training data exists, `solve()` falls back to a deterministic
greedy + 2-opt improvement so callers get a valid Solution.

Once a checkpoint exists at `models/pomo/svrptw_v1.pt`, replace
`_greedy_with_2opt` with the actual policy rollout.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.classical import greedy as greedy_mod
from svrptw.solvers.common import Route, Solution, evaluate

CHECKPOINT = Path("models/pomo/svrptw_v1.pt")


def _two_opt_route(route: list[int], T: np.ndarray, depot: int = 0,
                   max_iter: int = 200) -> list[int]:
    """In-place 2-opt on a single route under the asymmetric travel-time matrix."""
    if len(route) < 4:
        return route
    seq = [depot] + list(route) + [depot]
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, len(seq) - 2):
            for j in range(i + 1, len(seq) - 1):
                a, b = seq[i - 1], seq[i]
                c, d = seq[j], seq[j + 1]
                if b == c:
                    continue
                # Cost of edges to remove
                rm = T[a, b] + T[c, d]
                # Reversing segment seq[i..j] -> we now go a -> c -> ... -> b -> d
                # but asymmetric, so internal edges also flip; full cost diff:
                seg = seq[i:j + 1]
                rev = seg[::-1]
                new_internal = sum(T[rev[k], rev[k + 1]] for k in range(len(rev) - 1))
                old_internal = sum(T[seg[k], seg[k + 1]] for k in range(len(seg) - 1))
                add = T[a, rev[0]] + T[rev[-1], d] + new_internal
                rm_total = rm + old_internal
                if add + 1e-6 < rm_total:
                    seq[i:j + 1] = rev
                    improved = True
                    break
            if improved:
                break
    return seq[1:-1]


def solve(inst: Instance, settings: Settings, budget_seconds: float | None = None) -> Solution:
    """Until the POMO checkpoint exists, fall back to greedy + per-route 2-opt
    on the asymmetric travel-time matrix.  This keeps the interface live for
    the bench harness."""
    t0 = time.perf_counter()
    if CHECKPOINT.exists():
        # TODO Phase 4: load policy + run K-start rollout.
        pass

    # Fallback: greedy then per-route 2-opt with TW-feasibility check.
    base = greedy_mod.solve(inst, settings)
    new_routes: list[Route] = []
    T = inst.travel_time
    for r in base.routes:
        if not r.customers:
            new_routes.append(r)
            continue
        candidate = _two_opt_route(r.customers, T)
        if _route_tw_feasible(inst, candidate):
            new_routes.append(Route(customers=candidate))
        else:
            new_routes.append(r)

    sol = Solution(
        instance_id=inst.instance_id, routes=new_routes, solver="pomo_stub",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=float(budget_seconds or 0.0),
        feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    return sol


def _route_tw_feasible(inst: Instance, route: list[int]) -> bool:
    T = inst.travel_time
    cust_by_id = {c.id: c for c in inst.customers}
    clock = float(inst.depot.ready)
    cur = 0
    for cid in route:
        c = cust_by_id[cid]
        arrive = clock + float(T[cur, cid])
        start = max(arrive, float(c.ready))
        if start > c.due:
            return False
        clock = start + c.service
        cur = cid
    if clock + float(T[cur, 0]) > inst.depot.due:
        return False
    return True


if __name__ == "__main__":
    import argparse
    import json

    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings())
    print(json.dumps({"solver": sol.solver, "metrics": sol.metrics,
                      "wall_clock_s": sol.wall_clock_seconds}, indent=2))
