"""Scale-aware construction for VRPTW (Phase G3).

Targets N >= 1000 at sub-second wall. Reuses the cached OSM driving
graph for clustering signals (Louvain communities) so the per-call work
is O(|E_graph|) at preprocess + O(N log N) at construction time, NOT
the O(N**2) typical of a generic NN/Clarke-Wright multi-start.

This explicitly drops PyVRP-quality matching at scale -- the goal is a
cheap one-shot construction whose quality is graded by the graph-aware
quality_index from svrptw.metrics.graph_quality.

Public API mirrors PyVRP / fast_construct (v1):

    solve(inst, settings, budget_seconds=1.0, seed=0, *,
          n_starts=2, G=None) -> Solution

At N < 200 the call falls through to fast_construct (v1) -- v1's
multi-start + polish is the right thing for small instances; v2's
graph machinery is overhead at that scale.
"""
from __future__ import annotations

import math
import random
import time
from typing import Optional

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.common.local_search import (
    _route_arrival_and_close,
    _route_time,
    merge_routes,
    two_opt_intra,
)


# Threshold below which we delegate to v1's multi-start path.
_FALLBACK_N = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _customer_lookup(inst: Instance) -> dict:
    return {c.id: c for c in inst.customers}


def _route_load(inst: Instance, customers: list[int], lookup: dict) -> int:
    return sum(int(lookup[cid].demand) for cid in customers)


def _try_insert_best(inst: Instance, base: list[int], cid: int,
                     lookup: dict, capacity: int) -> Optional[tuple[float, int]]:
    """Cheapest TW+capacity-feasible position for cid in base.
    Returns (delta_route_time, position) or None.
    """
    if _route_load(inst, base, lookup) + lookup[cid].demand > capacity:
        return None
    base_time = _route_time(inst, base) if base else 0.0
    best: Optional[tuple[float, int]] = None
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
        solver="fast_construct_v2",
        wall_clock_seconds=time.perf_counter() - t0,
        budget_seconds=float(budget_seconds), feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    return sol


# ---------------------------------------------------------------------------
# Graph helpers (Louvain communities on customer-induced subgraph)
# ---------------------------------------------------------------------------


def _maybe_load_graph(inst: Instance):
    """Lazy-fetch the OSM graph for this instance, sharing the
    in-process cache used by graph_quality so repeat calls are O(1).
    """
    try:
        from svrptw.metrics.graph_quality import _get_cached_graph
        G, _ = _get_cached_graph(inst)
        return G
    except Exception:
        pass
    # Fallback: direct fetch (no in-process cache).
    try:
        from webui.basemap import instance_bbox
        from webui.road_network import get_network
    except Exception:
        return None
    try:
        bbox = instance_bbox(inst)
    except Exception:
        return None
    return get_network(bbox)


def _customer_communities(inst: Instance, G) -> list[list[int]]:
    """Group customer ids by Louvain community on the OSM graph.
    Customers whose node_id is missing or off-graph land in a singleton
    "tail" group at the end. Returns groups in descending size order.
    """
    cust_to_node: dict[int, int] = {}
    for c in inst.customers:
        nid = getattr(c, "node_id", None)
        if nid is None:
            continue
        try:
            cust_to_node[int(c.id)] = int(nid)
        except Exception:
            continue
    tail = [c.id for c in inst.customers if c.id not in cust_to_node]
    if not cust_to_node or G is None:
        return [[c.id for c in inst.customers]] if not tail else [tail]

    try:
        from svrptw.metrics.graph_quality import (
            _bbox_hash,
            _instance_bbox,
            _compute_or_load_communities,
        )
        cache_key = _bbox_hash(_instance_bbox(inst))
        node_to_comm = _compute_or_load_communities(G, cache_key)
    except Exception:
        node_to_comm = {}

    if not node_to_comm:
        return [[c.id for c in inst.customers]]

    buckets: dict[int, list[int]] = {}
    cross_tail: list[int] = []
    for cid, nid in cust_to_node.items():
        comm = node_to_comm.get(nid)
        if comm is None:
            cross_tail.append(cid)
            continue
        buckets.setdefault(comm, []).append(cid)
    groups = sorted(buckets.values(), key=len, reverse=True)
    if cross_tail:
        groups.append(cross_tail)
    if tail:
        groups.append(tail)
    return groups


# ---------------------------------------------------------------------------
# Per-community NN sweep
# ---------------------------------------------------------------------------


def _seed_nn_within_community(inst: Instance, group: list[int],
                              lookup: dict, *, capacity: int,
                              max_routes_remaining: int,
                              rng: random.Random) -> tuple[list[list[int]], list[int]]:
    """Greedy NN sweep restricted to a single community group. Caps at
    `max_routes_remaining` routes. Returns (routes, unrouted_in_group).
    """
    T = inst.travel_time
    unrouted = set(group)
    routes: list[list[int]] = []
    while unrouted and len(routes) < max_routes_remaining:
        # Seed with the depot-nearest unrouted in this community.
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
        # Extend greedily.
        while True:
            best: Optional[tuple[float, int, int]] = None
            for cid in unrouted:
                res = _try_insert_best(inst, route, cid, lookup, capacity)
                if res is None:
                    continue
                delta, pos = res
                if best is None or delta < best[0]:
                    best = (delta, cid, pos)
            if best is None:
                break
            _, cid, pos = best
            route = route[:pos] + [cid] + route[pos:]
            unrouted.discard(cid)
        routes.append(route)
    return routes, list(unrouted)


def _greedy_fill_unrouted(inst: Instance, routes: list[list[int]],
                          unrouted: list[int], lookup: dict, *,
                          capacity: int, max_routes: int,
                          deadline: float) -> list[list[int]]:
    """Regret-1 (greedy cheapest-insertion) fill for cross-community
    leftovers. O(N * R) per outer pass. Bails out at deadline.
    """
    pending = list(unrouted)
    while pending and time.perf_counter() < deadline:
        progressed = False
        for cid in list(pending):
            best: Optional[tuple[float, int, int]] = None  # (delta, ri, pos)
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
                    pending.remove(cid)
                    progressed = True
                    continue
            if best is not None:
                _, ri, pos = best
                routes[ri] = routes[ri][:pos] + [cid] + routes[ri][pos:]
                pending.remove(cid)
                progressed = True
        if not progressed:
            break
    return routes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve(inst: Instance, settings: Settings,
          budget_seconds: float = 1.0, seed: int = 0,
          n_starts: int = 2,
          *, G=None) -> Solution:
    """Scale-aware construction: Louvain-cluster, NN within each
    community, regret-1 fill the cross-community tail, one pass of
    two_opt_intra per route.

    Below ``_FALLBACK_N`` (=200), delegates to fast_construct (v1).

    Notes
    -----
    * No multi-start at N >= 500 (speed budget can't afford it).
    * No regret-3 at N >= 500 (O(n**3) blows up).
    * No swap_star (O(n**2) neighbourhood).
    """
    n = inst.num_customers
    if n < _FALLBACK_N:
        from svrptw.solvers.classical import fast_construct as fc_v1
        sol = fc_v1.solve(inst, settings,
                          budget_seconds=budget_seconds, seed=seed)
        sol.solver = "fast_construct_v2"
        return sol

    wall_t0 = time.perf_counter()  # for total wall reporting only

    # 1. One-time per-process per-city graph fetch. Done BEFORE the
    # construction budget starts so first-call graph download isn't
    # charged against budget_seconds (the call would otherwise return
    # K=0 routes when the graph fetch alone exceeds the budget).
    if G is None:
        G = _maybe_load_graph(inst)
    # 2. Louvain communities on customer-induced subgraph (cached).
    groups = _customer_communities(inst, G)

    t0 = time.perf_counter()
    deadline = t0 + max(0.05, float(budget_seconds))
    lookup = _customer_lookup(inst)
    capacity = int(inst.vehicle_capacity)
    max_routes = inst.num_vehicles

    # Force single-start at very large N -- the marginal value of a
    # second seed is tiny relative to the wall it costs.
    if n >= 500:
        n_starts = 1
    n_starts = max(1, int(n_starts))

    candidates: list[Solution] = []
    for s_idx in range(n_starts):
        if time.perf_counter() >= deadline:
            break
        rng = random.Random(int(seed) * 1009 + s_idx * 17 + 1)
        # Shuffle the order of community processing across starts; the
        # first start uses the descending-size order, later starts get a
        # randomised order so we sample different basins cheaply.
        ordered_groups = list(groups)
        if s_idx > 0:
            rng.shuffle(ordered_groups)
        routes: list[list[int]] = []
        cross_unrouted: list[int] = []
        for group in ordered_groups:
            if time.perf_counter() >= deadline:
                cross_unrouted.extend(c for c in group
                                      if not any(c in r for r in routes))
                continue
            slack = max_routes - len(routes)
            if slack <= 0:
                cross_unrouted.extend(group)
                continue
            new_routes, leftover = _seed_nn_within_community(
                inst, group, lookup,
                capacity=capacity,
                max_routes_remaining=slack,
                rng=rng,
            )
            routes.extend(new_routes)
            cross_unrouted.extend(leftover)
        # 4. Regret-1 fill cross-community customers.
        if cross_unrouted and time.perf_counter() < deadline:
            routes = _greedy_fill_unrouted(
                inst, routes, cross_unrouted, lookup,
                capacity=capacity, max_routes=max_routes,
                deadline=deadline,
            )
        sol = _build_solution(inst, settings, routes,
                              t0=t0, budget_seconds=budget_seconds)
        candidates.append(sol)


    if not candidates:
        best_sol = _build_solution(inst, settings, [],
                                   t0=t0, budget_seconds=budget_seconds)
    else:
        best_sol = min(candidates,
                       key=lambda c: float(c.metrics.get(
                           "operational_cost", float("inf"))))

    # 5a. iter-7-bis fix B: merge_routes polish. v2's per-community NN
    # produced K_built >> K_opt (Louvain communities became hard route
    # boundaries -- at v1_large that's 45-67 routes vs pyvrp's 16-32).
    # Each merge_routes call empties at most one route; loop while it
    # keeps reducing K and we have wall budget. Use ~60% of remaining
    # budget here; reserve 40% for the two_opt_intra polish below.
    remaining = deadline - time.perf_counter()
    if remaining > 0.1:
        merge_budget = remaining * 0.6
        merge_deadline = time.perf_counter() + merge_budget
        prev_k = int(best_sol.metrics.get("num_vehicles_used", 0))
        # Cap iterations so a pathological no-progress case can't spin.
        for _ in range(64):
            if time.perf_counter() >= merge_deadline:
                break
            try:
                trial = merge_routes(
                    inst, best_sol, settings,
                    max_seconds=max(0.05, merge_deadline - time.perf_counter()),
                )
            except Exception:
                break
            new_k = int(trial.metrics.get("num_vehicles_used", prev_k))
            new_cost = float(trial.metrics.get("operational_cost", float("inf")))
            old_cost = float(best_sol.metrics.get("operational_cost", float("inf")))
            # merge_routes returns sol unchanged when no profitable merge;
            # we detect "no progress" via K or cost staying flat.
            if new_k >= prev_k and new_cost >= old_cost - 1e-6:
                break
            best_sol = trial
            prev_k = new_k

    # 5b. ONE pass of two_opt_intra per route. Skips swap_star -- O(n**2)
    # neighbourhood is too costly at this scale.
    remaining = deadline - time.perf_counter()
    if remaining > 0.05:
        try:
            best_sol = two_opt_intra(inst, best_sol, settings,
                                     max_seconds=min(remaining, 0.5))
        except Exception:
            pass

    best_sol.solver = "fast_construct_v2"
    # Report total wall (including the one-time graph fetch) so callers
    # see the real elapsed time, not just the construction budget.
    best_sol.wall_clock_seconds = time.perf_counter() - wall_t0
    best_sol.budget_seconds = float(budget_seconds)
    return best_sol


__all__ = ["solve"]


if __name__ == "__main__":
    import argparse
    import json
    from svrptw.io import load_instance

    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-starts", type=int, default=2)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), budget_seconds=args.budget,
                seed=args.seed, n_starts=args.n_starts)
    print(json.dumps({
        "solver": sol.solver,
        "operational_cost": float(sol.metrics["operational_cost"]),
        "n_routes": int(sol.metrics["num_vehicles_used"]),
        "feasible": bool(sol.feasible),
        "wall_s": sol.wall_clock_seconds,
    }, indent=2))
