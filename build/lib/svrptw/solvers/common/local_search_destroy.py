"""SPEC-7-OPS-DESTROY-01 — drop_route operator.

Picks one route stochastically biased by under-utilisation,
deletes it, and re-inserts its customers into surrounding routes
via regret-2 cheapest-insertion. Restores the prior solution if
any customer cannot be re-inserted feasibly (no-regret invariant).

`destroy_island` (Voronoi) and `drop_leg` to follow as separate
files; this one is the highest-leverage of the three because it
attacks the entire-route waste in one move.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common.local_search import _cust_by_id, _route_arrival_and_close
from svrptw.solvers.common.solution import Route, Solution, evaluate


def _route_util(inst: Instance, customers: list[int], cust_by_id: dict) -> float:
    if not customers:
        return 0.0
    load = sum(cust_by_id[cid].demand for cid in customers)
    return load / max(1.0, float(inst.vehicle_capacity))


def _try_insert(inst: Instance, base: list[int], cid: int
                ) -> Optional[tuple[int, list[int], float]]:
    """Find the cheapest feasible position to insert cid into base.

    Returns (position, new_route, insertion_delta_time) or None if no
    feasible position exists.
    """
    from svrptw.solvers.common.local_search import _route_time

    best: Optional[tuple[int, list[int], float]] = None
    best_delta = float("inf")
    base_time = _route_time(inst, base) if base else 0.0
    for pos in range(len(base) + 1):
        new_route = base[:pos] + [cid] + base[pos:]
        ok, _ = _route_arrival_and_close(inst, new_route)
        if not ok:
            continue
        new_time = _route_time(inst, new_route)
        delta = new_time - base_time
        if delta < best_delta:
            best_delta = delta
            best = (pos, new_route, delta)
    return best


def _regret2_score(
    inst: Instance,
    cid: int,
    routes: list[list[int]],
) -> Optional[tuple[float, int, list[int]]]:
    """Per Pisinger & Ropke regret-2: rank by (2nd_best - best) so customers
    with one obvious home get inserted first, customers with many equal
    options get postponed."""
    candidates: list[tuple[float, int, list[int]]] = []
    for ri, base in enumerate(routes):
        out = _try_insert(inst, base, cid)
        if out is None:
            continue
        _, new_route, delta = out
        candidates.append((delta, ri, new_route))
    if not candidates:
        return None
    candidates.sort()
    if len(candidates) == 1:
        return candidates[0]
    # Regret-2: prefer customers with high (2nd - 1st) score.
    # But for this internal helper we return the best insertion and let
    # the outer loop track regret across customers separately.
    return candidates[0]


def drop_route(
    inst: Instance,
    sol: Solution,
    settings: Settings,
    *,
    max_seconds: float = 1.0,
    rng_seed: int = 0,
    epsilon: float = 0.10,
) -> Solution:
    """Drop one entire under-utilised route and redistribute its customers.

    Algorithm:
      1. Compute per-route utilisation.
      2. Score each non-empty route by (target_util - util)^2.
        target_util defaults to 0.70 (matches SPEC-7-COST-01).
      3. With probability (1-epsilon) pick the worst-scoring route; with
        probability epsilon pick uniformly from the top-3 worst.
      4. Remove the route; re-insert its customers via regret-1 cheapest
        insertion into the remaining routes.
      5. Accept iff (a) every customer was re-insertable AND
        (b) the resulting operational_cost is strictly less.
    """
    deadline = time.perf_counter() + max_seconds
    rng = np.random.default_rng(rng_seed if rng_seed != 0 else int(time.time_ns()) & 0xFFFF)
    cust_by_id = _cust_by_id(inst)
    target_util = 0.70

    non_empty_idx = [i for i, r in enumerate(sol.routes) if r.customers]
    if len(non_empty_idx) <= 1:
        return sol

    # Score routes; the lower the utilisation, the worse the score → higher (target-util)^2.
    scored = []
    for i in non_empty_idx:
        u = _route_util(inst, sol.routes[i].customers, cust_by_id)
        shortfall = max(0.0, target_util - u)
        scored.append((shortfall * shortfall, i, u))
    scored.sort(reverse=True)  # worst first

    if scored[0][0] == 0.0:
        # No under-utilised route — nothing to do.
        return sol

    # Epsilon-greedy pick.
    if rng.random() < epsilon and len(scored) >= 3:
        pick = scored[int(rng.integers(0, 3))]
    else:
        pick = scored[0]
    drop_idx = pick[1]
    removed = list(sol.routes[drop_idx].customers)

    # Try to re-insert each customer into the remaining routes.
    # Greedy order: process customers with highest demand first (hardest to fit).
    removed_sorted = sorted(removed, key=lambda cid: -cust_by_id[cid].demand)
    scratch = [list(r.customers) for i, r in enumerate(sol.routes) if i != drop_idx]
    deferred: list[int] = []
    for cid in removed_sorted:
        if time.perf_counter() >= deadline:
            # Out of budget; restore.
            return sol
        # Capacity check first: skip insertions into routes already at cap.
        viable = [
            (ri, base) for ri, base in enumerate(scratch)
            if _route_util(inst, base + [cid], cust_by_id) <= 1.0
        ]
        if not viable:
            deferred.append(cid)
            continue
        best_route = None
        best_delta = float("inf")
        best_pos = -1
        for ri, base in viable:
            out = _try_insert(inst, base, cid)
            if out is None:
                continue
            pos, new_route, delta = out
            if delta < best_delta:
                best_delta = delta
                best_route = (ri, new_route)
                best_pos = pos
        if best_route is None:
            deferred.append(cid)
        else:
            ri, new_route = best_route
            scratch[ri] = new_route

    if deferred:
        # Could not re-insert all customers feasibly → no-regret restore.
        return sol

    # Rebuild the Solution skipping the dropped route slot.
    new_routes: list[Route] = []
    j = 0
    for i in range(len(sol.routes)):
        if i == drop_idx:
            new_routes.append(Route(customers=[]))
        else:
            new_routes.append(Route(customers=scratch[j]))
            j += 1

    cand = Solution(
        instance_id=inst.instance_id,
        routes=new_routes,
        solver=sol.solver,
        wall_clock_seconds=sol.wall_clock_seconds,
        budget_seconds=sol.budget_seconds,
        feasible=False,
    )
    cand.metrics = evaluate(inst, cand, settings)
    cand.feasible = bool(cand.metrics["feasible"])
    if cand.metrics["operational_cost"] < sol.metrics["operational_cost"] - 1e-6:
        return cand
    return sol


def _customer_xy(inst: Instance, cid: int, cust_by_id: dict) -> tuple[float, float]:
    c = cust_by_id[cid]
    return (float(c.x), float(c.y))


def drop_leg(
    inst: Instance,
    sol: Solution,
    settings: Settings,
    *,
    max_seconds: float = 1.0,
    rng_seed: int = 0,
    top_k_legs: int = 3,
) -> Solution:
    """Pick the worst-load-utilisation leg in the solution, remove both
    endpoints, and re-insert each via greedy-cheapest (capacity-aware).

    A "leg" is one edge (c_i → c_{i+1}) within a route. The load
    utilisation of a leg = on-board load after serving c_i ÷ capacity.
    Low utilisation means the vehicle is carrying very little between
    those stops — a candidate for re-routing.

    Distinct from `relocate` (moves one customer) and `drop_route`
    (deletes a whole route): drop_leg removes TWO consecutive customers,
    which unblocks cases where neither alone could be moved feasibly.
    """
    deadline = time.perf_counter() + max_seconds
    rng = np.random.default_rng(rng_seed if rng_seed != 0 else int(time.time_ns()) & 0xFFFF)
    cust_by_id = _cust_by_id(inst)
    cap = max(1.0, float(inst.vehicle_capacity))

    # Enumerate (load-util-after-c_i, route_idx, leg_idx) for every leg
    # of length ≥ 2; pick the lowest k as candidates.
    candidates: list[tuple[float, int, int]] = []
    for ri, r in enumerate(sol.routes):
        if len(r.customers) < 2:
            continue
        load = 0.0
        for i, cid in enumerate(r.customers):
            load += float(cust_by_id[cid].demand)
            if i + 1 < len(r.customers):
                # Leg goes from c_i to c_{i+1}; load AT c_i is what's
                # being carried across that leg.
                candidates.append((load / cap, ri, i))
    if not candidates:
        return sol

    candidates.sort()
    pool = candidates[: max(1, top_k_legs)]
    pick = pool[int(rng.integers(0, len(pool)))]
    _, route_idx, leg_idx = pick

    # The two customers spanning the leg.
    removed = [
        sol.routes[route_idx].customers[leg_idx],
        sol.routes[route_idx].customers[leg_idx + 1],
    ]

    # Build the scratch with the leg removed.
    scratch: list[list[int]] = []
    for i, r in enumerate(sol.routes):
        if i == route_idx:
            scratch.append(
                r.customers[:leg_idx] + r.customers[leg_idx + 2:]
            )
        else:
            scratch.append(list(r.customers))

    # Greedy-cheapest reinsertion (capacity-aware).
    for cid in removed:
        if time.perf_counter() >= deadline:
            return sol
        viable = [(ri, base) for ri, base in enumerate(scratch)
                  if _route_util(inst, base + [cid], cust_by_id) <= 1.0]
        if not viable:
            return sol
        best_ri, best_new = -1, None
        best_delta = float("inf")
        for ri, base in viable:
            out = _try_insert(inst, base, cid)
            if out is None:
                continue
            _, new_route, delta = out
            if delta < best_delta:
                best_delta = delta
                best_ri = ri
                best_new = new_route
        if best_new is None:
            return sol
        scratch[best_ri] = best_new

    new_routes = [Route(customers=r) for r in scratch]
    cand = Solution(
        instance_id=inst.instance_id,
        routes=new_routes,
        solver=sol.solver,
        wall_clock_seconds=sol.wall_clock_seconds,
        budget_seconds=sol.budget_seconds,
        feasible=False,
    )
    cand.metrics = evaluate(inst, cand, settings)
    cand.feasible = bool(cand.metrics["feasible"])
    if cand.metrics["operational_cost"] < sol.metrics["operational_cost"] - 1e-6:
        return cand
    return sol


def destroy_island(
    inst: Instance,
    sol: Solution,
    settings: Settings,
    *,
    max_seconds: float = 1.0,
    rng_seed: int = 0,
    island_size: int = 8,
) -> Solution:
    """Remove a geographic cluster of customers (a Voronoi-like island)
    that may span multiple routes, then re-insert each via greedy-cheapest.

    Different from `drop_route` (single-route, util-biased) — this picks a
    seed customer and removes its k-nearest geographic neighbours
    *regardless of which route they're on*, which is the move that
    untangles cross-route spaghetti.

    Algorithm:
      1. Pick a seed customer uniformly at random.
      2. Find its `island_size - 1` nearest customers by Euclidean (x, y).
      3. Remove all `island_size` customers from their respective routes.
      4. Greedy-cheapest re-insertion (capacity-aware) into the remaining
         partial routes.
      5. Accept iff cost strictly improves AND every customer was placed.
    """
    deadline = time.perf_counter() + max_seconds
    cust_by_id = _cust_by_id(inst)
    rng = np.random.default_rng(rng_seed if rng_seed != 0 else int(time.time_ns()) & 0xFFFF)

    all_cids = [c.id for c in inst.customers]
    if len(all_cids) < island_size + 1:
        return sol

    seed_cid = int(rng.choice(all_cids))
    sx, sy = _customer_xy(inst, seed_cid, cust_by_id)
    dists = [(((cust_by_id[cid].x - sx) ** 2 + (cust_by_id[cid].y - sy) ** 2),
              cid) for cid in all_cids]
    dists.sort()
    island = {cid for _, cid in dists[:island_size]}

    # Strip island from each route.
    scratch: list[list[int]] = []
    for r in sol.routes:
        scratch.append([cid for cid in r.customers if cid not in island])

    # Greedy-cheapest reinsertion. Highest-demand first.
    by_demand = sorted(island, key=lambda cid: -cust_by_id[cid].demand)
    for cid in by_demand:
        if time.perf_counter() >= deadline:
            return sol
        viable = [(ri, base) for ri, base in enumerate(scratch)
                  if _route_util(inst, base + [cid], cust_by_id) <= 1.0]
        if not viable:
            return sol
        best_route_idx = -1
        best_new = None
        best_delta = float("inf")
        for ri, base in viable:
            out = _try_insert(inst, base, cid)
            if out is None:
                continue
            _, new_route, delta = out
            if delta < best_delta:
                best_delta = delta
                best_route_idx = ri
                best_new = new_route
        if best_new is None:
            return sol
        scratch[best_route_idx] = best_new

    new_routes = [Route(customers=r) for r in scratch]
    cand = Solution(
        instance_id=inst.instance_id,
        routes=new_routes,
        solver=sol.solver,
        wall_clock_seconds=sol.wall_clock_seconds,
        budget_seconds=sol.budget_seconds,
        feasible=False,
    )
    cand.metrics = evaluate(inst, cand, settings)
    cand.feasible = bool(cand.metrics["feasible"])
    if cand.metrics["operational_cost"] < sol.metrics["operational_cost"] - 1e-6:
        return cand
    return sol
