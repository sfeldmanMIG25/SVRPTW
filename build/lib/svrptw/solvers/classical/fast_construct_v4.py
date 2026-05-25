"""fast_construct_v4 -- Solomon I1 sequential insertion (Solomon 1987).

Goal: close the +37% cost gap that ``fast_construct_v2`` (Louvain + merge_routes
polish) leaves vs pyvrp. v2 is fast and pyvrp-independent but uses NN + merge,
which produces good-but-not-great routes. I1 is the classical TW-aware
insertion heuristic that's much closer to pyvrp's HGS-construction quality.

Algorithm (Solomon 1987, "Algorithms for the VRP with Time Window
Constraints", Operations Research 35(2)):

  1. Pick a seed customer (default: farthest from depot, breaking ties by
     earliest due time). Start a new route with just this customer.
  2. Repeat while unrouted customers remain that fit ANYWHERE in the
     current route:
       a. For each unrouted u, find the best insertion position (i,j)
          in the current route that's TW + capacity feasible.
          Score c1(i,u,j) = alpha1*c11 + alpha2*c12, where
            c11 = d(i,u) + d(u,j) - mu*d(i,j)         (distance increase)
            c12 = b_j_new - b_j                       (push-forward in arrival at j)
       b. Among all u with a feasible position, pick the one that
          maximizes  c2(u) = lambda*d(0,u) - c1(i,u,j)
          (encourages picking customers far from the depot first).
       c. Insert u at its best position.
  3. When no unrouted u fits the current route, start a new route with
     the next-best seed.
  4. Optional one-pass two_opt_intra polish per route.

Standard parameter set: mu=1, alpha1=1, alpha2=0, lambda=1 (cheapest-
insertion + farthest-first seed). These match Solomon's published "I1"
configuration.

Scale optimization: at large N, instead of scoring ALL unrouted customers
each iteration, score only the ``nearest_k`` customers nearest to either
endpoint of the current route. Brings per-iter cost from O(N*L) to
O(K'*L) with K' << N at the cost of slightly worse routing decisions.

Constraint composition: the I1 inner loop only checks TW + capacity (the
hard feasibility constraints). The full cost model (embargo, skills, etc.)
is the bandit's job after construction. This keeps construction fast and
deterministic; the cost-term constraints fold in naturally via the bandit
refinement that follows.

Public API mirrors v1/v2/v3:
    solve(inst, settings, budget_seconds=2.0, seed=0) -> Solution
"""
from __future__ import annotations

import math
import time
from typing import Optional

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.common.local_search import (
    _route_arrival_and_close,
    two_opt_intra,
    two_opt_star,
    swap_star,
    relocate,
)


# Solomon I1 default parameters (the "I1" config from the 1987 paper).
_MU = 1.0
_ALPHA1 = 1.0
_ALPHA2 = 0.0
_LAMBDA = 1.0


def _customer_lookup(inst: Instance) -> dict:
    return {c.id: c for c in inst.customers}


def _route_load(customers: list[int], lookup: dict) -> int:
    return sum(int(lookup[cid].demand) for cid in customers)


def _arrival_times(inst: Instance, route: list[int]) -> Optional[list[float]]:
    """Per-stop SERVICE-START times for depot->c1->c2->...->cN->depot.
    Mirrors svrptw.solvers.common.local_search._route_arrival_and_close
    so the depot.ready / depot.due / customer TW semantics stay aligned
    with the rest of the evaluator. Returns None if any visit is TW-
    infeasible or the route can't make it back to depot by depot.due.

    arr[k] = service-start clock at stop route[k] (post-wait if needed).
    """
    if not route:
        return []
    T = inst.travel_time
    cust = inst.customers
    arr = [0.0] * len(route)
    clock = float(inst.depot.ready)
    cur = 0
    for k, cid in enumerate(route):
        c = cust[cid - 1]
        arrive = clock + float(T[cur, cid])
        start = max(arrive, float(c.ready))
        if start > float(c.due):
            return None
        arr[k] = start
        clock = start + float(c.service)
        cur = cid
    end = clock + float(T[cur, 0])
    if end > float(inst.depot.due):
        return None
    return arr


def _try_insert_with_pushforward(
    inst: Instance, route: list[int], pos: int, u: int,
    *, embargo_windows: tuple = (), embargo_pen: float = 0.0,
) -> Optional[tuple[float, float, float]]:
    """Try inserting u at position `pos` in `route` (i.e., between
    route[pos-1] and route[pos]; pos=0 means at start, pos=len(route)
    means at end). Returns (c11, c12, embargo_delta) if TW+depot feasible,
    else None.

    c11 = d(i,u) + d(u,j) - mu*d(i,j)   (distance increase, oriented)
    c12 = push-forward in arrival time at the customer formerly at pos
          (= 0 if pos == len(route))
    embargo_delta = delta in embargo penalty for the inserted-at-u arrival
          time (constraint-aware insertion). Computed only if
          ``embargo_windows`` non-empty and ``embargo_pen`` > 0. Cheap
          single-window check on u's new arrival time -- doesn't account
          for push-forward into other customers' embargo overlap.
    """
    T = inst.travel_time
    n = len(route)
    if pos < 0 or pos > n:
        return None
    # i = predecessor (0 = depot), j = successor (0 = depot)
    i = route[pos - 1] if pos > 0 else 0
    j = route[pos] if pos < n else 0
    new_route = route[:pos] + [u] + route[pos:]
    new_arr = _arrival_times(inst, new_route)
    if new_arr is None:
        return None
    # c11: distance increase from inserting u between i and j
    c11 = float(T[i, u]) + float(T[u, j]) - _MU * float(T[i, j])
    # c12: push-forward at j (was b_j; now b_j_new). If pos == n, j is the
    # depot -- push-forward is undefined; use 0.
    if pos < n:
        old_arr = _arrival_times(inst, route)
        if old_arr is None:
            c12 = 0.0
        else:
            c12 = new_arr[pos + 1] - old_arr[pos]
    else:
        c12 = 0.0
    # embargo_delta: penalty cost for u's new arrival time landing in any
    # embargo window. Lightweight single-customer check (doesn't account for
    # push-forward into other customers' embargo overlap, but a cheap
    # approximation that captures the dominant signal).
    embargo_delta = 0.0
    if embargo_windows and embargo_pen > 0.0:
        u_arr = new_arr[pos]  # service-start clock at u in new_route
        for ws, we in embargo_windows:
            if ws <= u_arr <= we:
                embargo_delta = embargo_pen
                break
    return (c11, c12, embargo_delta)


def _best_insertion(
    inst: Instance, route: list[int], u: int, capacity: int, demand_u: int,
    route_load: int,
    *, embargo_windows: tuple = (), embargo_pen: float = 0.0,
    cost_weight: float = 1.0,
) -> Optional[tuple[float, float, int]]:
    """Find the cheapest TW+capacity feasible position for u in route.
    Returns (c1, c11, position) or None.

    c1 = alpha1*c11 + alpha2*c12 + cost_weight * embargo_delta

    When ``embargo_windows`` is empty (constraint-unaware mode), the
    embargo_delta term is always 0 and c1 reduces to pure I1.
    """
    if route_load + demand_u > capacity:
        return None
    best: Optional[tuple[float, float, int]] = None
    for pos in range(len(route) + 1):
        res = _try_insert_with_pushforward(
            inst, route, pos, u,
            embargo_windows=embargo_windows, embargo_pen=embargo_pen,
        )
        if res is None:
            continue
        c11, c12, embargo_delta = res
        c1 = _ALPHA1 * c11 + _ALPHA2 * c12 + cost_weight * embargo_delta
        if best is None or c1 < best[0]:
            best = (c1, c11, pos)
    return best


def _farthest_seed(unrouted: list[int], inst: Instance) -> Optional[int]:
    """Pick the seed = unrouted customer farthest from depot whose
    [depot -> c -> depot] is TW-feasible. Ties broken by earliest due time.
    """
    T = inst.travel_time
    best_cid: Optional[int] = None
    best_d = -1.0
    best_due = math.inf
    for cid in unrouted:
        # Singleton feasibility check
        if _arrival_times(inst, [cid]) is None:
            continue
        d = float(T[0, cid])
        due = float(inst.customers[cid - 1].due)
        if d > best_d or (abs(d - best_d) < 1e-9 and due < best_due):
            best_d = d; best_due = due; best_cid = cid
    return best_cid


def _nearest_k_unrouted(
    inst: Instance, route: list[int], unrouted: set[int], k: int,
) -> list[int]:
    """Return up to k unrouted customers nearest to either endpoint of
    the current route (depot if route is empty). Cheap scan O(|unrouted|).
    """
    T = inst.travel_time
    anchors = []
    if route:
        anchors.append(route[0])
        if len(route) > 1:
            anchors.append(route[-1])
    else:
        anchors.append(0)
    if not unrouted:
        return []
    scored: list[tuple[float, int]] = []
    for cid in unrouted:
        d = min(float(T[a, cid]) for a in anchors)
        scored.append((d, cid))
    scored.sort()
    return [c for _, c in scored[:k]]


def solve(inst: Instance, settings: Settings,
          budget_seconds: float = 2.0, seed: int = 0,
          *, nearest_k: int = 24, polish: bool = True,
          deep_polish: bool = False,
          cost_aware: bool = True) -> Solution:
    """Solomon I1 sequential insertion construction.

    Budget is best-effort. `seed` unused except for API parity (I1 is
    deterministic given the same instance + params).

    `nearest_k`: at each step within a route, only consider this many
    unrouted customers (nearest to a route endpoint) as insertion
    candidates. Default 24 keeps the scoring loop O(K' * L) instead of
    O(N * L). Set to 0 (or >= N) for full enumeration.

    `polish`: enable end-of-construction polish. Default True.

    `deep_polish`: after the per-route two_opt_intra pass, also run a
    cross-route polish chain (two_opt_star -> relocate -> swap_star) to
    deepen the construction basin. This matches what HGS does internally
    inside PyVRP construction. Without it, v4 standalone beats pyvrp on
    cost but v4-warmstarted solve_auto loses to pyvrp-warmstarted
    solve_auto (the bandit can't dig as deep from v4's shallower basin).
    Default True. Adds ~30-50% to construction wall.

    `cost_aware`: when True AND ``settings.economics`` has embargo
    constraints active, add the per-customer embargo penalty to the I1
    insertion score c1. This makes I1 actively avoid embargo windows at
    construction time, closing the iter-7-v4-embargo gap (where vanilla
    v4 lost 0/6 to pyvrp under embargo cost term). Cheap single-customer
    check; doesn't account for push-forward into other customers'
    embargo overlap. Default True (no-op when no cost term active).
    """
    t0 = time.perf_counter()
    deadline = t0 + max(0.05, float(budget_seconds))
    lookup = _customer_lookup(inst)
    capacity = int(inst.vehicle_capacity)
    max_routes = inst.num_vehicles
    cust_ids = sorted(lookup.keys())
    # Drop customers that can't fit any vehicle (capacity infeasible).
    cust_ids = [c for c in cust_ids if int(lookup[c].demand) <= capacity]
    unrouted: set[int] = set(cust_ids)

    # Constraint-aware insertion params: pre-extract embargo windows from
    # settings so we don't hit the pydantic attribute access cost in the
    # inner loop. Empty tuple disables cost-aware scoring.
    embargo_windows: tuple = ()
    embargo_pen = 0.0
    if cost_aware:
        e = settings.economics
        if (e.embargo_violation_penalty_per_visit > 0.0
                and e.embargo_window_starts
                and e.embargo_window_ends):
            embargo_windows = tuple(
                (float(s), float(t))
                for s, t in zip(e.embargo_window_starts, e.embargo_window_ends)
            )
            embargo_pen = float(e.embargo_violation_penalty_per_visit)

    routes: list[list[int]] = []
    while unrouted and len(routes) < max_routes:
        if time.perf_counter() >= deadline:
            break
        seed_cid = _farthest_seed(list(unrouted), inst)
        if seed_cid is None:
            break  # nothing TW-feasible left
        route = [seed_cid]
        unrouted.discard(seed_cid)
        route_load = int(lookup[seed_cid].demand)

        # Inner loop: keep inserting until no candidate fits.
        while unrouted:
            if time.perf_counter() >= deadline:
                break
            # Candidate set: nearest_k unrouted to current route endpoints
            if nearest_k and nearest_k < len(unrouted):
                cands = _nearest_k_unrouted(inst, route, unrouted, nearest_k)
            else:
                cands = list(unrouted)
            # For each candidate, find its best (c1, position).
            # Score for selection: c2 = lambda*d(0,u) - c1 (max wins).
            T = inst.travel_time
            best_choice: Optional[tuple[float, int, int]] = None  # (c2, cid, pos)
            for u in cands:
                bi = _best_insertion(
                    inst, route, u, capacity,
                    int(lookup[u].demand), route_load,
                    embargo_windows=embargo_windows,
                    embargo_pen=embargo_pen,
                )
                if bi is None:
                    continue
                c1, c11, pos = bi
                c2 = _LAMBDA * float(T[0, u]) - c1
                if best_choice is None or c2 > best_choice[0]:
                    best_choice = (c2, u, pos)
            if best_choice is None:
                break  # current route can't accept anything more
            _, u, pos = best_choice
            route = route[:pos] + [u] + route[pos:]
            route_load += int(lookup[u].demand)
            unrouted.discard(u)
        routes.append(route)

    # Optional per-route two_opt_intra polish (cheap; intra-only).
    if polish and time.perf_counter() < deadline:
        per_slot = max(0.02,
                       (deadline - time.perf_counter()) / max(1, len(routes)))
        polished: list[list[int]] = []
        for r in routes:
            if time.perf_counter() >= deadline:
                polished.append(r); continue
            try:
                tiny = Solution(
                    instance_id=inst.instance_id,
                    routes=[Route(customers=list(r))],
                    solver="fast_construct_v4_polish",
                    wall_clock_seconds=0.0, budget_seconds=per_slot,
                    feasible=False,
                )
                tiny.metrics = evaluate(inst, tiny, settings)
                tiny = two_opt_intra(inst, tiny, settings, max_seconds=per_slot)
                polished.append(list(tiny.routes[0].customers))
            except Exception:
                polished.append(r)
        routes = polished

    # Build intermediate Solution for cross-route polish (deep_polish).
    sol_routes = [Route(customers=list(r)) for r in routes if r]
    while len(sol_routes) < inst.num_vehicles:
        sol_routes.append(Route(customers=[]))
    sol_intermediate = Solution(
        instance_id=inst.instance_id, routes=sol_routes,
        solver="fast_construct_v4",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=float(budget_seconds),
        feasible=False,
    )
    sol_intermediate.metrics = evaluate(inst, sol_intermediate, settings)
    sol_intermediate.feasible = bool(sol_intermediate.metrics["feasible"])

    # deep_polish: cross-route LS chain to match HGS-construction depth.
    # Bandit operators that come downstream do better from the deeper basin.
    if deep_polish and time.perf_counter() < deadline:
        remaining = deadline - time.perf_counter()
        # Split: 40% two_opt_star, 30% relocate, 30% swap_star.
        for op_fn, frac in (
            (two_opt_star, 0.40),
            (relocate, 0.30),
            (swap_star, 0.30),
        ):
            if time.perf_counter() >= deadline:
                break
            slot = max(0.05, remaining * frac)
            try:
                sol_intermediate = op_fn(
                    inst, sol_intermediate, settings, max_seconds=slot,
                )
            except Exception:
                pass
    sol = sol_intermediate
    # Final routes list is now whatever the deep polish produced.
    sol_routes = sol.routes
    while len(sol_routes) < inst.num_vehicles:
        sol_routes.append(Route(customers=[]))
    sol = Solution(
        instance_id=inst.instance_id, routes=sol_routes,
        solver="fast_construct_v4",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=float(budget_seconds),
        feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    sol.metrics["i1_unrouted"] = int(len(unrouted))
    return sol


if __name__ == "__main__":
    import argparse
    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-polish", action="store_true")
    p.add_argument("--deep-polish", action="store_true",
                   help="Enable cross-route LS polish (two_opt_star + "
                        "relocate + swap_star). Standalone gives marginal "
                        "improvement at higher wall; may break the bandit "
                        "downstream when used as warmstart through "
                        "solve_auto (under investigation).")
    p.add_argument("--nearest-k", type=int, default=24)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), budget_seconds=args.budget,
                seed=args.seed, polish=not args.no_polish,
                deep_polish=args.deep_polish,
                nearest_k=args.nearest_k)
    print(f"K={int(sol.metrics['num_vehicles_used'])} "
          f"cost=${sol.metrics['operational_cost']:.1f} "
          f"feas={int(sol.feasible)} "
          f"wall={sol.wall_clock_seconds:.2f}s "
          f"unrouted={sol.metrics.get('i1_unrouted', 0)}")
