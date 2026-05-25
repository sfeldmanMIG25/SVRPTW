"""Fast multi-start construction for VRPTW (Phase E2).

Public API matches PyVRP's:
    solve(inst, settings, budget_seconds=2.0, seed=0) -> Solution

Strategy: build K diverse starts (NN-from-depot, Clarke-Wright savings,
polar-angle sweep, random-greedy), regret-3 fill any unrouted leftovers,
then run a small budget of swap_star + two_opt_intra per start. Return
the best by `operational_cost`.

Goal: land in a basin comparable to PyVRP's construction phase without
paying its convergence cost. Designed to be a drop-in replacement
for the PyVRP construction stage of `portfolio_pyvrp_warm.solve()`.

Stdlib + numpy only; reuses operators from `svrptw.solvers.common`.
"""
from __future__ import annotations

import math
import random
import time
from typing import Callable

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.metrics import score_solution
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.common.local_search import (
    _route_arrival_and_close,
    _route_time,
    swap_star,
    two_opt_intra,
    two_opt_star,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _customer_lookup(inst: Instance) -> dict:
    return {c.id: c for c in inst.customers}


def _depot_polar_angle(inst: Instance, cid: int, lookup: dict) -> float:
    c = lookup[cid]
    return math.atan2(c.y - inst.depot.y, c.x - inst.depot.x)


def _route_load(inst: Instance, customers: list[int], lookup: dict) -> int:
    return sum(int(lookup[cid].demand) for cid in customers)


def _try_insert_best(inst: Instance, base: list[int], cid: int,
                     lookup: dict, capacity: int) -> tuple[float, int] | None:
    """Cheapest TW+capacity-feasible position for cid in `base`.
    Returns (delta_route_time, position) or None if infeasible."""
    if _route_load(inst, base, lookup) + lookup[cid].demand > capacity:
        return None
    base_time = _route_time(inst, base) if base else 0.0
    best: tuple[float, int] | None = None
    for pos in range(len(base) + 1):
        cand = base[:pos] + [cid] + base[pos:]
        ok, _ = _route_arrival_and_close(inst, cand)
        if not ok:
            continue
        delta = _route_time(inst, cand) - base_time
        if best is None or delta < best[0]:
            best = (delta, pos)
    return best


def _build_solution(inst: Instance, settings: Settings,
                    routes: list[list[int]],
                    *, t0: float, budget_seconds: float) -> Solution:
    sol_routes = [Route(customers=list(r)) for r in routes if r]
    while len(sol_routes) < inst.num_vehicles:
        sol_routes.append(Route(customers=[]))
    sol = Solution(
        instance_id=inst.instance_id, routes=sol_routes,
        solver="fast_construct",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=float(budget_seconds), feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    return sol


# ---------------------------------------------------------------------------
# Seed strategy 1 — Nearest-neighbor from depot
# ---------------------------------------------------------------------------

def _seed_nn_from_depot(inst: Instance, settings: Settings,
                        lookup: dict, *, rng: random.Random) -> list[list[int]]:
    """Classic nearest-neighbor: each route starts at depot's nearest
    unrouted customer and greedily extends until capacity/TW exhausted."""
    T = inst.travel_time
    capacity = int(inst.vehicle_capacity)
    unrouted = set(lookup.keys())
    routes: list[list[int]] = []
    max_routes = inst.num_vehicles

    while unrouted and len(routes) < max_routes:
        # Pick depot-nearest unrouted customer that's TW-feasible alone.
        candidates: list[tuple[float, int]] = []
        for cid in unrouted:
            res = _try_insert_best(inst, [], cid, lookup, capacity)
            if res is None:
                continue
            candidates.append((float(T[0, cid]), cid))
        if not candidates:
            break
        candidates.sort()
        seed_cid = candidates[0][1]
        route = [seed_cid]
        unrouted.discard(seed_cid)

        # Extend greedily until no feasible cheap insertion exists.
        while True:
            best: tuple[float, int] | None = None
            for cid in unrouted:
                res = _try_insert_best(inst, route, cid, lookup, capacity)
                if res is None:
                    continue
                delta, pos = res
                if best is None or delta < best[0]:
                    best = (delta, cid)
                    best_pos = pos
            if best is None:
                break
            route = route[:best_pos] + [best[1]] + route[best_pos:]
            unrouted.discard(best[1])
        routes.append(route)
    return routes


# ---------------------------------------------------------------------------
# Seed strategy 2 — Clarke-Wright savings (TW-aware)
# ---------------------------------------------------------------------------

def _seed_savings(inst: Instance, settings: Settings,
                  lookup: dict, *, rng: random.Random) -> list[list[int]]:
    """Classic Clarke-Wright: start with one route per customer, merge
    by savings = d(0,i) + d(0,j) - d(i,j). Only accept TW+capacity feasible
    merges. Greedy descending-savings order."""
    T = inst.travel_time
    capacity = int(inst.vehicle_capacity)
    cust_ids = list(lookup.keys())

    # Initial: one route per feasible customer.
    routes: list[list[int]] = []
    cust_to_route: dict[int, int] = {}
    for cid in cust_ids:
        ok, _ = _route_arrival_and_close(inst, [cid])
        if not ok:
            continue
        if lookup[cid].demand > capacity:
            continue
        cust_to_route[cid] = len(routes)
        routes.append([cid])

    # Build savings list; only consider distinct customer pairs.
    savings: list[tuple[float, int, int]] = []
    for i in cust_ids:
        for j in cust_ids:
            if j <= i:
                continue
            s = float(T[0, i]) + float(T[0, j]) - float(T[i, j])
            savings.append((s, i, j))
    savings.sort(reverse=True)
    max_routes = inst.num_vehicles
    for s, i, j in savings:
        if s <= 0:
            break  # negative savings unlikely to help
        ri = cust_to_route.get(i)
        rj = cust_to_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        ri_route = routes[ri]
        rj_route = routes[rj]
        # Only merge endpoints (i tail of ri, j head of rj OR symmetric).
        if ri_route[-1] == i and rj_route[0] == j:
            merged = ri_route + rj_route
        elif rj_route[-1] == j and ri_route[0] == i:
            merged = rj_route + ri_route
        else:
            continue
        # Capacity + TW check.
        if _route_load(inst, merged, lookup) > capacity:
            continue
        ok, _ = _route_arrival_and_close(inst, merged)
        if not ok:
            continue
        # Commit: keep the lower-index slot, blank the other.
        keep = min(ri, rj)
        drop = max(ri, rj)
        routes[keep] = merged
        routes[drop] = []
        for cid in merged:
            cust_to_route[cid] = keep

    routes = [r for r in routes if r]
    # If we somehow have more than max_routes, keep the largest.
    if len(routes) > max_routes:
        routes.sort(key=len, reverse=True)
        routes = routes[:max_routes]
    return routes


# ---------------------------------------------------------------------------
# Seed strategy 3 — Polar-angle sweep
# ---------------------------------------------------------------------------

def _seed_polar_sweep(inst: Instance, settings: Settings,
                      lookup: dict, *, rng: random.Random) -> list[list[int]]:
    """Sort customers by polar angle around depot, fill routes greedily
    until capacity/TW prevent further extension. Random rotation seed
    breaks ties between calls."""
    capacity = int(inst.vehicle_capacity)
    cust_ids = list(lookup.keys())
    angle_offset = rng.uniform(0.0, 2.0 * math.pi)
    ordered = sorted(
        cust_ids,
        key=lambda cid: (_depot_polar_angle(inst, cid, lookup) + angle_offset)
                        % (2.0 * math.pi),
    )

    routes: list[list[int]] = []
    max_routes = inst.num_vehicles
    cur_route: list[int] = []
    for cid in ordered:
        if len(routes) >= max_routes and not cur_route:
            break
        # Try to append/insert into current route at best position.
        if cur_route:
            res = _try_insert_best(inst, cur_route, cid, lookup, capacity)
        else:
            res = _try_insert_best(inst, [], cid, lookup, capacity)
        if res is None:
            # Start a new route if room.
            if len(routes) + 1 >= max_routes:
                continue
            if cur_route:
                routes.append(cur_route)
            cur_route = []
            res2 = _try_insert_best(inst, [], cid, lookup, capacity)
            if res2 is None:
                continue
            cur_route = [cid]
        else:
            _, pos = res
            cur_route = cur_route[:pos] + [cid] + cur_route[pos:]
    if cur_route:
        routes.append(cur_route)
    return routes


# ---------------------------------------------------------------------------
# Seed strategy 4 — Random-greedy
# ---------------------------------------------------------------------------

def _seed_random_greedy(inst: Instance, settings: Settings,
                        lookup: dict, *, rng: random.Random) -> list[list[int]]:
    """Random customer order + cheapest-feasible-insertion. Adds variety
    when the deterministic seeds (NN/savings/polar) all land in the
    same basin."""
    capacity = int(inst.vehicle_capacity)
    cust_ids = list(lookup.keys())
    rng.shuffle(cust_ids)
    routes: list[list[int]] = []
    max_routes = inst.num_vehicles
    for cid in cust_ids:
        # Cheapest insertion across existing routes.
        best: tuple[float, int, int] | None = None  # (delta, ri, pos)
        for ri, route in enumerate(routes):
            res = _try_insert_best(inst, route, cid, lookup, capacity)
            if res is None:
                continue
            delta, pos = res
            if best is None or delta < best[0]:
                best = (delta, ri, pos)
        if best is not None:
            delta, ri, pos = best
            r = routes[ri]
            routes[ri] = r[:pos] + [cid] + r[pos:]
            continue
        # Open new route if room.
        if len(routes) < max_routes:
            res = _try_insert_best(inst, [], cid, lookup, capacity)
            if res is not None:
                routes.append([cid])
    return routes


# ---------------------------------------------------------------------------
# Regret-3 fill for unrouted customers
# ---------------------------------------------------------------------------

def _regret_fill(inst: Instance, routes: list[list[int]],
                 unrouted: set[int], lookup: dict, *,
                 capacity: int, max_routes: int, k: int = 3,
                 sample_routes: int | None = None,
                 rng: random.Random | None = None) -> list[list[int]]:
    """For each unrouted customer, compute its k cheapest feasible
    insertion costs. Insert the customer with maximum
    `sum(cost_2..k) - (k-1)*cost_1` regret. Repeat until either every
    customer is placed or none has any feasible slot.

    For large N, regret-k is O(N^2 * R) per outer step which becomes
    O(N^3) when R ~= N. Pass `sample_routes=S` to only consider the
    `S` nearest open routes (by distance to the customer). Falls back
    to full enumeration when sample_routes is None or >= len(routes).
    """
    T = inst.travel_time
    while unrouted:
        scored: list[tuple[float, int, int, int]] = []  # (regret, cid, ri, pos)
        for cid in list(unrouted):
            opts: list[tuple[float, int, int]] = []
            # Subsample candidate routes by proximity at large N.
            if sample_routes is not None and len(routes) > sample_routes:
                # Score each open route by distance to its centroid (cheap).
                ranked: list[tuple[float, int]] = []
                for ri, base in enumerate(routes):
                    if not base:
                        continue
                    # Use travel_time(cid, midpoint_customer) as a proxy.
                    mid = base[len(base) // 2]
                    ranked.append((float(T[cid, mid]), ri))
                ranked.sort()
                ri_iter = [ri for _, ri in ranked[:sample_routes]]
            else:
                ri_iter = list(range(len(routes)))
            for ri in ri_iter:
                base = routes[ri]
                res = _try_insert_best(inst, base, cid, lookup, capacity)
                if res is not None:
                    opts.append((res[0], ri, res[1]))
            # New-route option.
            if len(routes) < max_routes:
                res = _try_insert_best(inst, [], cid, lookup, capacity)
                if res is not None:
                    opts.append((res[0], len(routes), res[1]))
            if not opts:
                # Truly infeasible — drop.
                unrouted.discard(cid)
                continue
            opts.sort()
            top = opts[:k]
            best_cost, best_ri, best_pos = top[0]
            if len(top) == 1:
                regret = 1e6 + best_cost  # rare singletons get priority
            else:
                regret = sum(c[0] for c in top[1:]) - (len(top) - 1) * best_cost
            scored.append((regret, cid, best_ri, best_pos))
        if not scored:
            break
        scored.sort(reverse=True)  # largest regret first
        _r, cid, ri, pos = scored[0]
        if ri >= len(routes):
            routes.append([])
        routes[ri] = routes[ri][:pos] + [cid] + routes[ri][pos:]
        unrouted.discard(cid)
    return routes


def _greedy_fill(inst: Instance, routes: list[list[int]],
                 unrouted: set[int], lookup: dict, *,
                 capacity: int, max_routes: int,
                 deadline: float | None = None) -> list[list[int]]:
    """Cheapest-insertion fallback for very large N where regret-k is
    too expensive. O(N^2 * R) per outer step but no inner sort overhead.

    iter-7-bis fix A: optional ``deadline`` (a ``time.perf_counter()``
    epoch). When hit, exits early with whatever's been placed -- caller
    accepts a partial routes list. Without a deadline, behaviour is
    bit-identical to the pre-fix code.
    """
    while unrouted:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        progressed = False
        for cid in list(unrouted):
            if deadline is not None and time.perf_counter() >= deadline:
                # Bail out of inner loop too; outer while will exit next.
                break
            best: tuple[float, int, int] | None = None  # (delta, ri, pos)
            for ri, base in enumerate(routes):
                res = _try_insert_best(inst, base, cid, lookup, capacity)
                if res is None:
                    continue
                delta, pos = res
                if best is None or delta < best[0]:
                    best = (delta, ri, pos)
            if best is None and len(routes) < max_routes:
                res = _try_insert_best(inst, [], cid, lookup, capacity)
                if res is not None:
                    routes.append([cid])
                    unrouted.discard(cid)
                    progressed = True
                    continue
            if best is not None:
                _, ri, pos = best
                routes[ri] = routes[ri][:pos] + [cid] + routes[ri][pos:]
                unrouted.discard(cid)
                progressed = True
        if not progressed:
            break
    return routes


# ---------------------------------------------------------------------------
# Unified-smoke scoring helper (cost + quality blend)
# ---------------------------------------------------------------------------

def _segments_cross(a1, a2, b1, b2) -> bool:
    """Strict-cross orientation test (no shared endpoints)."""
    def _ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    d1 = _ccw(b1, b2, a1); d2 = _ccw(b1, b2, a2)
    d3 = _ccw(a1, a2, b1); d4 = _ccw(a1, a2, b2)
    return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
            and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))


def _count_inter_crossings(inst: Instance, routes: list[list[int]]) -> int:
    """Count strict cross-route segment intersections (matches the
    metrics suite's `inter_route_crossings` aggregate)."""
    cust_xy = {c.id: (c.x, c.y) for c in inst.customers}
    depot_xy = (inst.depot.x, inst.depot.y)
    paths = []
    for r in routes:
        if not r:
            continue
        path = [depot_xy] + [cust_xy[c] for c in r] + [depot_xy]
        paths.append([(path[i], path[i+1]) for i in range(len(path)-1)])
    total = 0
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            for s1 in paths[i]:
                for s2 in paths[j]:
                    if _segments_cross(s1[0], s1[1], s2[0], s2[1]):
                        total += 1
    return total


def _crossing_polish(inst: Instance, settings: Settings, sol: Solution,
                     *, max_seconds: float) -> Solution:
    """Greedy crossing remover: scan pairs of route segments. When two
    segments (depot->c, c->c, c->depot) from different routes cross,
    try the four tail-swap reconnections from 2-opt* and accept the
    one that strictly lowers `(crossings, operational_cost)` lexicographically.

    Falls back to returning `sol` unchanged when wall budget exhausted
    or no productive swap exists.
    """
    deadline = time.perf_counter() + max(0.05, float(max_seconds))
    cust_xy = {c.id: (c.x, c.y) for c in inst.customers}
    depot_xy = (inst.depot.x, inst.depot.y)
    routes = [list(r.customers) for r in sol.routes if r.customers]
    capacity = int(inst.vehicle_capacity)
    lookup = _customer_lookup(inst)
    cur_cross = _count_inter_crossings(inst, routes)
    cur_cost = float(sol.metrics["operational_cost"])
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        # Build per-route segment lists once per outer pass.
        segs = []
        for r in routes:
            path = [depot_xy] + [cust_xy[c] for c in r] + [depot_xy]
            segs.append([(path[i], path[i+1], i) for i in range(len(path)-1)])
        outer_break = False
        for ai in range(len(routes)):
            if outer_break or time.perf_counter() >= deadline:
                break
            for bi in range(ai + 1, len(routes)):
                if outer_break or time.perf_counter() >= deadline:
                    break
                ra, rb = routes[ai], routes[bi]
                for sa1, sa2, ia in segs[ai]:
                    if outer_break:
                        break
                    for sb1, sb2, ib in segs[bi]:
                        if not _segments_cross(sa1, sa2, sb1, sb2):
                            continue
                        # 2-opt* tail-swap: route a keeps prefix [0..ia],
                        # route b keeps prefix [0..ib], tails swap.
                        new_a = ra[:ia] + rb[ib:]
                        new_b = rb[:ib] + ra[ia:]
                        if _route_load(inst, new_a, lookup) > capacity:
                            continue
                        if _route_load(inst, new_b, lookup) > capacity:
                            continue
                        ok_a, _ = _route_arrival_and_close(inst, new_a)
                        if not ok_a:
                            continue
                        ok_b, _ = _route_arrival_and_close(inst, new_b)
                        if not ok_b:
                            continue
                        trial = [r for r in routes]
                        trial[ai] = new_a
                        trial[bi] = new_b
                        new_cross = _count_inter_crossings(inst, trial)
                        if new_cross >= cur_cross:
                            continue
                        # Probe cost via evaluate.
                        probe_sol = _build_solution(
                            inst, settings, trial,
                            t0=time.perf_counter(),
                            budget_seconds=0.0,
                        )
                        new_cost = float(probe_sol.metrics["operational_cost"])
                        # Accept only if cost stays within 5% (no-regret).
                        if new_cost > cur_cost * 1.05:
                            continue
                        routes = trial
                        cur_cross = new_cross
                        cur_cost = new_cost
                        improved = True
                        outer_break = True
                        break
    out = _build_solution(inst, settings, routes,
                          t0=time.perf_counter(),
                          budget_seconds=0.0)
    return out


def _unified_smoke_score(inst: Instance, sol: Solution,
                         cost_ref: float | None = None) -> float:
    """Weighted blend matching the user's reported smoke: 0.5*cost_norm +
    0.5*quality_index. Lower cost + higher quality = larger score.

    cost_ref defaults to `cost` itself, giving cost_norm = 1.0; the
    multi-start picker normalizes against the best cost in its own pool
    so the ranking is invariant to absolute scale.
    """
    cost = float(sol.metrics.get("operational_cost", float("inf")))
    if not math.isfinite(cost):
        return -float("inf")
    try:
        q = score_solution(inst, sol).quality_index
    except Exception:
        q = 0.0
    if cost_ref is None or cost_ref <= 0 or not math.isfinite(cost_ref):
        cost_norm = 1.0
    else:
        # Lower cost is better; normalize so ref->1.0, 2x ref -> 0.5.
        cost_norm = max(0.0, min(1.0, cost_ref / cost))
    return 0.5 * cost_norm + 0.5 * float(q)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Order matters: deterministic seeds come first, then the random one for
# variety. The first K (default 4) are used by `solve()`.
_SEED_STRATEGIES: list[tuple[str, Callable]] = [
    ("nn_from_depot", _seed_nn_from_depot),
    ("savings",       _seed_savings),
    ("polar_sweep",   _seed_polar_sweep),
    ("random_greedy", _seed_random_greedy),
]


def solve(inst: Instance, settings: Settings,
          budget_seconds: float = 2.0, seed: int = 0,
          *, k_starts: int = 4,
          polish: bool = True,
          quality_pick: bool = True) -> Solution:
    """Multi-start fast construction (quality-aware, polish-by-default).

    Drops K diverse seeds, fills any unrouted with regret-k or sampled-
    regret/greedy at large N, then applies one round each of swap_star
    and two_opt_intra per start. With ``polish=True``, ends with one
    pass of ``two_opt_star`` to eliminate cross-route crossings (the
    main quality lever vs cost-only multi-start picks).

    With ``quality_pick=True`` (default), the multi-start winner is the
    candidate with the highest unified smoke score
    (``0.5*cost_norm + 0.5*quality_index``) instead of cost-only —
    matches the iter-5g leaderboard objective.

    Wall-time bound: best-effort under ``budget_seconds``. Each start
    gets ``budget_seconds / (k_starts + polish_slot)`` total. At N>=100
    the per-start LS budgets are bumped (>=0.4s each) so swap_star /
    two_opt_intra can actually finish a pass.
    """
    t0 = time.perf_counter()
    deadline = t0 + max(0.05, float(budget_seconds))
    lookup = _customer_lookup(inst)
    capacity = int(inst.vehicle_capacity)
    max_routes = inst.num_vehicles
    n = inst.num_customers
    k_starts = max(1, int(k_starts))
    # Reserve ~25% of budget for the end-of-pipeline two_opt_star polish.
    polish_share = 0.25 if polish else 0.0
    per_start = max(0.05,
                    float(budget_seconds) * (1.0 - polish_share) / k_starts)
    # Per-op LS slot: at least 0.4s at N>=100, scale down for tiny budgets.
    base_slot = max(0.02, per_start * 0.375)
    if n >= 100:
        op_slot = max(0.4, base_slot)
    else:
        op_slot = base_slot

    # Pick the regret strategy by N.
    if n <= 200:
        regret_k = 3
        sample_routes: int | None = None
    elif n <= 400:
        regret_k = 2
        sample_routes = 8
    else:
        regret_k = 0  # signal greedy fallback below
        sample_routes = None

    candidates: list[Solution] = []

    for i, (name, seeder) in enumerate(_SEED_STRATEGIES[:k_starts]):
        if time.perf_counter() >= deadline:
            break
        rng = random.Random(int(seed) * 1009 + i * 17 + 1)
        try:
            routes = seeder(inst, settings, lookup, rng=rng)
        except Exception:
            routes = []
        # Identify unrouted customers.
        placed: set[int] = set()
        for r in routes:
            placed.update(r)
        unrouted = set(lookup.keys()) - placed
        if unrouted:
            if regret_k == 0:
                routes = _greedy_fill(
                    inst, routes, unrouted, lookup,
                    capacity=capacity, max_routes=max_routes,
                    deadline=deadline,
                )
            else:
                routes = _regret_fill(
                    inst, routes, unrouted, lookup,
                    capacity=capacity, max_routes=max_routes, k=regret_k,
                    sample_routes=sample_routes, rng=rng,
                )
        sol = _build_solution(inst, settings, routes,
                              t0=t0, budget_seconds=budget_seconds)
        # Light LS on each start; only if we still have wall budget.
        remaining = deadline - time.perf_counter()
        if remaining > 0.05:
            slot = min(op_slot, remaining * 0.5)
            try:
                sol = swap_star(inst, sol, settings, max_seconds=slot)
            except Exception:
                pass
        remaining = deadline - time.perf_counter()
        if remaining > 0.05:
            slot = min(op_slot, remaining)
            try:
                sol = two_opt_intra(inst, sol, settings, max_seconds=slot)
            except Exception:
                pass
        candidates.append(sol)

    if not candidates:
        # Fallback: single empty solution (should not happen on a sane
        # instance) -- evaluate to populate metrics.
        best_sol = _build_solution(inst, settings, [], t0=t0,
                                   budget_seconds=budget_seconds)
    else:
        # Quality-aware best-of-multi-start: pick by unified smoke score
        # instead of cost-only, then polish the survivor.
        if quality_pick:
            cost_ref = min(float(c.metrics.get("operational_cost", float("inf")))
                           for c in candidates)
            best_sol = max(candidates,
                           key=lambda c: _unified_smoke_score(inst, c, cost_ref))
        else:
            best_sol = min(candidates,
                           key=lambda c: float(c.metrics.get(
                               "operational_cost", float("inf"))))

    # End-of-pipeline polish: one pass of cross-route 2-opt* (cost-driven),
    # then a crossing-targeted swap pass that strictly reduces
    # inter_route_crossings within a 5% cost tolerance. The two together
    # form a Pareto-aware polish: cost first, then visual cleanup.
    if polish:
        remaining = deadline - time.perf_counter()
        if remaining > 0.1:
            polish_slot = min(remaining,
                              max(0.2, float(budget_seconds) * polish_share))
            # Split: 35% to cost-driven 2-opt*, 65% to crossing cleanup.
            # Cost-side gets a quick pass; crossing cleanup is the
            # quality lever and dominates the budget.
            cost_slot = polish_slot * 0.35
            cross_slot = polish_slot * 0.65
            try:
                best_sol = two_opt_star(inst, best_sol, settings,
                                        max_seconds=cost_slot)
            except Exception:
                pass
            remaining = deadline - time.perf_counter()
            if remaining > 0.05:
                try:
                    best_sol = _crossing_polish(
                        inst, settings, best_sol,
                        max_seconds=min(cross_slot, remaining),
                    )
                except Exception:
                    pass

    best_sol.solver = "fast_construct"
    best_sol.wall_clock_seconds = time.perf_counter() - t0
    best_sol.budget_seconds = float(budget_seconds)
    return best_sol


if __name__ == "__main__":
    import argparse
    import json
    from svrptw.io import load_instance

    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=4, help="number of starts")
    p.add_argument("--no-polish", action="store_true",
                   help="disable end-of-pipeline two_opt_star polish")
    p.add_argument("--cost-only", action="store_true",
                   help="pick multi-start winner by cost instead of unified smoke")
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), budget_seconds=args.budget,
                seed=args.seed, k_starts=args.k,
                polish=not args.no_polish,
                quality_pick=not args.cost_only)
    print(json.dumps({
        "solver": sol.solver,
        "operational_cost": float(sol.metrics["operational_cost"]),
        "n_routes": int(sol.metrics["num_vehicles_used"]),
        "feasible": bool(sol.feasible),
        "wall_s": sol.wall_clock_seconds,
    }, indent=2))
