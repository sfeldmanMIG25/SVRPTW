"""fast_construct_v3 -- Clarke-Wright savings on the OSM travel-time matrix.

Diagnosed 2026-05-16: v2's Louvain-community-bounded NN fragments routes
(K = 45-67 at N=500 vs pyvrp's 16-17). v1 produces good K but overruns
its wall budget by 3-4x at large N because seeders/fills aren't deadline-
aware.

v3 picks the simplest scale-friendly construction that naturally saturates
K close to optimum and uses graph distances directly:

    1. Each customer starts as its own route [c].
    2. Compute savings s(i,j) = T[0,i] + T[0,j] - T[i,j] using the
       precomputed OSM travel-time matrix (already on inst.travel_time --
       no Euclidean translation, no rendering, no community boundaries).
    3. Sort pairs by descending s, walk in order.
    4. Merge routes of i and j whenever:
         - i and j are each at an endpoint of their (different) routes
         - capacity feasible
         - time-window feasible after merge
    5. Optional one pass of two_opt_intra per surviving route as polish.

Complexity: O(N^2) savings list + O(N^2 log N) sort + at most O(N) merges
with O(avg_route) TW check per merge => ~few seconds at N=1000.

Public API mirrors v1/v2:
    solve(inst, settings, budget_seconds=2.0, seed=0) -> Solution
"""
from __future__ import annotations

import time
from typing import Optional

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.common.local_search import (
    _route_arrival_and_close,
    two_opt_intra,
)


def _customer_lookup(inst: Instance) -> dict:
    return {c.id: c for c in inst.customers}


def _route_load(customers: list[int], lookup: dict) -> int:
    return sum(int(lookup[cid].demand) for cid in customers)


def _route_tw_ok(inst: Instance, route: list[int]) -> bool:
    """True if the given customer sequence is time-window feasible."""
    if not route:
        return True
    ok, _ = _route_arrival_and_close(inst, route)
    return bool(ok)


def _maybe_merge(
    inst: Instance, i: int, j: int,
    route_of: dict[int, int], routes: list[Optional[list[int]]],
    loads: list[int], lookup: dict, capacity: int,
) -> bool:
    """Attempt to merge the routes of i and j so they become adjacent.

    Returns True if merged. On success, the route of j is destroyed
    (set to None in the routes list) and route_of[] entries updated.
    """
    ri, rj = route_of[i], route_of[j]
    if ri == rj:
        return False
    Ri = routes[ri]
    Rj = routes[rj]
    if Ri is None or Rj is None:
        return False
    # i must be at an endpoint of Ri; j must be at an endpoint of Rj
    i_first = Ri[0] == i
    i_last = Ri[-1] == i
    if not (i_first or i_last):
        return False
    j_first = Rj[0] == j
    j_last = Rj[-1] == j
    if not (j_first or j_last):
        return False
    # Orient: we want Ri ending in i and Rj starting with j so i adj j
    Ri_oriented = Ri if i_last else list(reversed(Ri))
    Rj_oriented = Rj if j_first else list(reversed(Rj))
    merged = Ri_oriented + Rj_oriented
    new_load = loads[ri] + loads[rj]
    if new_load > capacity:
        return False
    if not _route_tw_ok(inst, merged):
        return False
    # Commit
    routes[ri] = merged
    loads[ri] = new_load
    routes[rj] = None
    loads[rj] = 0
    for cid in Rj:
        route_of[cid] = ri
    return True


def solve(inst: Instance, settings: Settings,
          budget_seconds: float = 2.0, seed: int = 0,
          *, polish: bool = True) -> Solution:
    """Clarke-Wright savings construction on the OSM travel-time matrix.

    Budget is best-effort. The savings + sort + merge is deterministic
    (no seed dependence except for tie-breaking, none here). `seed` is
    accepted for API parity with v1/v2 but unused.
    """
    t0 = time.perf_counter()
    deadline = t0 + max(0.05, float(budget_seconds))
    lookup = _customer_lookup(inst)
    capacity = int(inst.vehicle_capacity)
    cust_ids = sorted(lookup.keys())  # depot is 0; customers start at 1

    # Initialize: each customer is its own route, if singleton TW-feasible.
    # Customers whose [depot -> c -> depot] is infeasible get dropped now
    # (caller can re-handle via missing).
    routes: list[Optional[list[int]]] = []
    loads: list[int] = []
    route_of: dict[int, int] = {}
    for cid in cust_ids:
        if int(lookup[cid].demand) > capacity:
            continue  # impossible to serve at all
        single = [cid]
        if not _route_tw_ok(inst, single):
            continue  # tight TW; skip from initial routes
        route_of[cid] = len(routes)
        routes.append(single)
        loads.append(int(lookup[cid].demand))

    # Build the savings list on T = inst.travel_time.
    T = inst.travel_time
    n_cust = len(cust_ids)
    savings: list[tuple[float, int, int]] = []
    cap = (n_cust * (n_cust - 1)) // 2
    # Optionally truncate at very large N to keep wall bounded (savings
    # below 0 cannot reduce cost vs separate routes); we keep all positive.
    for ai in range(len(cust_ids)):
        i = cust_ids[ai]
        Ti0 = float(T[i, 0])
        T0i = float(T[0, i])
        for aj in range(ai + 1, len(cust_ids)):
            j = cust_ids[aj]
            s = T0i + float(T[0, j]) - float(T[i, j])
            # Use symmetric estimate for tie-stability; many OSM matrices
            # are nearly symmetric but not bit-identical.
            s2 = Ti0 + float(T[j, 0]) - float(T[j, i])
            sv = 0.5 * (s + s2)
            if sv > 0:
                savings.append((sv, i, j))
    # Sort descending.
    savings.sort(reverse=True)

    # Walk savings list; merge greedily.
    n_merges = 0
    n_attempted = 0
    for sv, i, j in savings:
        if time.perf_counter() >= deadline:
            break
        n_attempted += 1
        if _maybe_merge(inst, i, j, route_of, routes, loads,
                        lookup, capacity):
            n_merges += 1

    # Compact (drop the None routes).
    final_routes = [r for r in routes if r]

    # Optional per-route 2-opt polish (cheap; only if budget allows).
    if polish and time.perf_counter() < deadline:
        per_slot = max(0.02,
                       (deadline - time.perf_counter()) / max(1, len(final_routes)))
        polished: list[list[int]] = []
        for r in final_routes:
            if time.perf_counter() >= deadline:
                polished.append(r)
                continue
            # two_opt_intra works on a full Solution; build a tiny one.
            try:
                tiny = Solution(
                    instance_id=inst.instance_id,
                    routes=[Route(customers=list(r))],
                    solver="fast_construct_v3_polish",
                    wall_clock_seconds=0.0, budget_seconds=per_slot,
                    feasible=False,
                )
                tiny.metrics = evaluate(inst, tiny, settings)
                tiny = two_opt_intra(inst, tiny, settings, max_seconds=per_slot)
                polished.append(list(tiny.routes[0].customers))
            except Exception:
                polished.append(r)
        final_routes = polished

    # Build the Solution.
    sol_routes = [Route(customers=list(r)) for r in final_routes if r]
    # Pad with empty Route objects up to the fleet size for downstream
    # solvers that expect num_vehicles entries.
    while len(sol_routes) < inst.num_vehicles:
        sol_routes.append(Route(customers=[]))
    sol = Solution(
        instance_id=inst.instance_id, routes=sol_routes,
        solver="fast_construct_v3",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=float(budget_seconds),
        feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    sol.metrics["savings_merges"] = int(n_merges)
    sol.metrics["savings_attempted"] = int(n_attempted)
    return sol


if __name__ == "__main__":
    import argparse
    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-polish", action="store_true")
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), budget_seconds=args.budget,
                seed=args.seed, polish=not args.no_polish)
    print(f"K={int(sol.metrics['num_vehicles_used'])} "
          f"cost=${sol.metrics['operational_cost']:.1f} "
          f"feas={int(sol.feasible)} "
          f"wall={sol.wall_clock_seconds:.2f}s "
          f"merges={sol.metrics.get('savings_merges', 0)}/"
          f"{sol.metrics.get('savings_attempted', 0)}")
