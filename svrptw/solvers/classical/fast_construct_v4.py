"""fast_construct_v4 -- Solomon I1 sequential insertion with cost-aware
extensions and auto-multi-start (final form, iter-7 series).

Final form. Solomon 1987 I1 insertion (farthest-from-depot seed by default,
alpha1=1, alpha2=0, lambda=1, mu=1) with three production extensions that
landed across the iter-7 / iter-7-bis / iter-7-v4 series:

  1. ``cost_aware=True`` (default): when ``settings.economics`` has embargo
     OR driver_breaks active, insertion score c1 includes the per-customer
     embargo penalty (push-forward-aware: counts all customers shifted
     into windows by the insertion) plus the per-route driver_breaks delta
     (delta_driving = c11 when mu=1; piecewise-linear over-cap penalty).
     No-op when no cost term active.

  2. ``n_starts=None`` (default, auto-selects): 3 starts when a cost term
     creates seed-strategy spread (embargo or driver_breaks active), else
     1 start. Strategies are farthest-from-depot, earliest-due, highest-
     demand. First pass gets up to the full remaining budget (guaranteed-
     completion semantics); subsequent passes share whatever time is left.

  3. ``nearest_k=24`` (default): per-step candidate set limited to the k
     unrouted customers nearest to either endpoint of the current route,
     keeping per-step cost O(K'*L) instead of O(N*L) at large N.

Calibrated bench truth (sequential, max_workers=1, contention-free; the
parallel benches had +/-$200-$2,600/inst noise from CPU contention --
see iter-7-determinism-audit):

  * Baseline (no cost term): v4 ~ ties pyvrp on cost; uses 1-2 fewer K
    on 4/6 v1_large instances. Standalone v4 also beats pyvrp by -3% cost
    at 3.8x faster wall.
  * Embargo (per-visit cost term): v4 ties pyvrp at solve_auto level (3/6
    paired wins, mean -$122/inst aggregate); v4 saves 1-3 routes on 3/6.
  * Driver_breaks (per-route cost term): v4 loses to pyvrp; v4's tighter
    K compounds per-route penalty. Construction-side cost-aware addition
    didn't close the gap (per-route caps are global to route topology,
    not local to single insertions).
  * Stack-16 (mixed per-visit + per-route): pyvrp wins because per-route
    terms dominate the cost mix.

Public API: solve(inst, settings, budget_seconds=2.0, seed=0,
                  *, nearest_k=24, polish=True, cost_aware=True,
                  n_starts=None) -> Solution
Plumbed into svrptw.solvers.classical.portfolio_pyvrp_warm.solve_auto via
``construction="fast_construct_v4"``.
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


def _route_driving_time(inst: Instance, route: list[int]) -> float:
    """Sum of travel-time edges including depot->first and last->depot.
    Excludes waiting and service. Used by per-route cost-aware
    (driver_breaks: cap on cumulative driving minutes).
    """
    if not route:
        return 0.0
    T = inst.travel_time
    total = float(T[0, route[0]])
    for k in range(1, len(route)):
        total += float(T[route[k - 1], route[k]])
    total += float(T[route[-1], 0])
    return total


def _try_insert_with_pushforward(
    inst: Instance, route: list[int], pos: int, u: int,
    *, embargo_windows: tuple = (), embargo_pen: float = 0.0,
    breaks_cap: float = 0.0, breaks_pen: float = 0.0,
    old_driving: float = 0.0,
) -> Optional[tuple[float, float, float, float]]:
    """Try inserting u at position `pos`. Returns
    (c11, c12, embargo_delta, breaks_delta) if TW+depot feasible, else None.

    c11 = d(i,u) + d(u,j) - mu*d(i,j)   (distance increase, oriented)
    c12 = push-forward in arrival time at the customer formerly at pos
    embargo_delta = delta in embargo penalty (per-visit cost term)
    breaks_delta = delta in driver_breaks penalty (per-route cost term).
        Uses precomputed ``old_driving`` (computed once per route by the
        caller). When ``breaks_pen`` = 0, always 0.
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
    # depot -- push-forward is undefined; use 0. We may need old_arr also
    # for the push-forward-aware embargo check below, so compute it once.
    old_arr: Optional[list[float]] = None
    if pos < n or (embargo_windows and embargo_pen > 0.0 and n > 0):
        old_arr = _arrival_times(inst, route)
    if pos < n:
        if old_arr is None:
            c12 = 0.0
        else:
            c12 = new_arr[pos + 1] - old_arr[pos]
    else:
        c12 = 0.0
    # embargo_delta: change in embargo penalty between old and new routes.
    # Iter-7-v4-pushfwd (2026-05-16): full push-forward-aware check --
    # counts customers landing in embargo windows in the NEW route minus
    # those in the OLD route, times the per-visit penalty. Captures both
    # u's direct contribution AND push-forward into subsequent customers'
    # embargo overlap (which the prior single-customer check missed,
    # leaving a $130-$507 gap at N=1000 vs pyvrp under embargo).
    embargo_delta = 0.0
    if embargo_windows and embargo_pen > 0.0:
        def _in_window(t: float) -> bool:
            for ws, we in embargo_windows:
                if ws <= t <= we:
                    return True
            return False
        new_count = sum(1 for t in new_arr if _in_window(t))
        old_count = 0
        if old_arr is not None:
            old_count = sum(1 for t in old_arr if _in_window(t))
        embargo_delta = (new_count - old_count) * embargo_pen
    # breaks_delta: change in driver_breaks penalty (per-route cap).
    # iter-7-v4-breaks-aware (2026-05-16): the per-route equivalent of the
    # embargo cost_aware extension. Because mu=1 (Solomon I1 default), the
    # delta in route driving-time from inserting u is exactly c11. We
    # compute old vs new penalty using piecewise-linear over-cap penalty.
    breaks_delta = 0.0
    if breaks_pen > 0.0 and breaks_cap > 0.0:
        new_driving = old_driving + c11  # c11 = T[i,u] + T[u,j] - T[i,j]
        old_pen_cost = max(0.0, old_driving - breaks_cap) * breaks_pen
        new_pen_cost = max(0.0, new_driving - breaks_cap) * breaks_pen
        breaks_delta = new_pen_cost - old_pen_cost
    return (c11, c12, embargo_delta, breaks_delta)


def _best_insertion(
    inst: Instance, route: list[int], u: int, capacity: int, demand_u: int,
    route_load: int,
    *, embargo_windows: tuple = (), embargo_pen: float = 0.0,
    breaks_cap: float = 0.0, breaks_pen: float = 0.0,
    cost_weight: float = 1.0,
) -> Optional[tuple[float, float, int]]:
    """Find the cheapest TW+capacity feasible position for u in route.
    Returns (c1, c11, position) or None.

    c1 = alpha1*c11 + alpha2*c12 + cost_weight * (embargo_delta + breaks_delta)

    When all cost term knobs are off, c1 reduces to pure Solomon I1.
    """
    if route_load + demand_u > capacity:
        return None
    # Precompute old route driving time (independent of pos) for breaks.
    old_driving = (_route_driving_time(inst, route)
                   if breaks_pen > 0.0 and breaks_cap > 0.0 else 0.0)
    best: Optional[tuple[float, float, int]] = None
    for pos in range(len(route) + 1):
        res = _try_insert_with_pushforward(
            inst, route, pos, u,
            embargo_windows=embargo_windows, embargo_pen=embargo_pen,
            breaks_cap=breaks_cap, breaks_pen=breaks_pen,
            old_driving=old_driving,
        )
        if res is None:
            continue
        c11, c12, embargo_delta, breaks_delta = res
        c1 = (_ALPHA1 * c11 + _ALPHA2 * c12
              + cost_weight * (embargo_delta + breaks_delta))
        if best is None or c1 < best[0]:
            best = (c1, c11, pos)
    return best


def _farthest_seed(unrouted: list[int], inst: Instance) -> Optional[int]:
    """Pick the seed = unrouted customer farthest from depot whose
    [depot -> c -> depot] is TW-feasible. Ties broken by earliest due time.
    Solomon's classical "I1" seed.
    """
    T = inst.travel_time
    best_cid: Optional[int] = None
    best_d = -1.0
    best_due = math.inf
    for cid in unrouted:
        if _arrival_times(inst, [cid]) is None:
            continue
        d = float(T[0, cid])
        due = float(inst.customers[cid - 1].due)
        if d > best_d or (abs(d - best_d) < 1e-9 and due < best_due):
            best_d = d; best_due = due; best_cid = cid
    return best_cid


def _earliest_due_seed(unrouted: list[int], inst: Instance) -> Optional[int]:
    """Alternative seed: pick the unrouted customer with the earliest due
    time. Tends to anchor routes around the tightest TW constraints first,
    producing structurally different routes than _farthest_seed. Used by
    multi-start v4."""
    best_cid: Optional[int] = None
    best_due = math.inf
    best_d = -1.0
    T = inst.travel_time
    for cid in unrouted:
        if _arrival_times(inst, [cid]) is None:
            continue
        due = float(inst.customers[cid - 1].due)
        d = float(T[0, cid])
        # Pick smallest due; tie-break by farthest from depot
        if due < best_due or (abs(due - best_due) < 1e-9 and d > best_d):
            best_due = due; best_d = d; best_cid = cid
    return best_cid


def _highest_demand_seed(unrouted: list[int], inst: Instance) -> Optional[int]:
    """Alternative seed: pick the unrouted customer with the highest demand
    (capacity-greedy). Forces big customers placed first, which often
    produces tighter K because small customers can fill leftover capacity.
    Used by multi-start v4."""
    best_cid: Optional[int] = None
    best_demand = -1
    best_d = -1.0
    T = inst.travel_time
    for cid in unrouted:
        if _arrival_times(inst, [cid]) is None:
            continue
        d_cust = int(inst.customers[cid - 1].demand)
        d_dep = float(T[0, cid])
        if d_cust > best_demand or (d_cust == best_demand and d_dep > best_d):
            best_demand = d_cust; best_d = d_dep; best_cid = cid
    return best_cid


_SEED_STRATEGIES = {
    "farthest": _farthest_seed,
    "earliest_due": _earliest_due_seed,
    "highest_demand": _highest_demand_seed,
}


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


def _construct_one_pass(
    inst: Instance, settings: Settings, deadline: float,
    *, seed_fn, lookup: dict, capacity: int, max_routes: int,
    cust_ids: list, nearest_k: int,
    embargo_windows: tuple, embargo_pen: float,
    breaks_cap: float = 0.0, breaks_pen: float = 0.0,
) -> list[list[int]]:
    """Run one I1 construction pass with a specified seed strategy.
    Returns the routes list. Used by the multi-start dispatcher in solve().
    """
    unrouted: set[int] = set(cust_ids)
    routes: list[list[int]] = []
    while unrouted and len(routes) < max_routes:
        if time.perf_counter() >= deadline:
            break
        seed_cid = seed_fn(list(unrouted), inst)
        if seed_cid is None:
            break
        route = [seed_cid]
        unrouted.discard(seed_cid)
        route_load = int(lookup[seed_cid].demand)
        while unrouted:
            if time.perf_counter() >= deadline:
                break
            if nearest_k and nearest_k < len(unrouted):
                cands = _nearest_k_unrouted(inst, route, unrouted, nearest_k)
            else:
                cands = list(unrouted)
            T = inst.travel_time
            best_choice: Optional[tuple[float, int, int]] = None
            for u in cands:
                bi = _best_insertion(
                    inst, route, u, capacity,
                    int(lookup[u].demand), route_load,
                    embargo_windows=embargo_windows, embargo_pen=embargo_pen,
                    breaks_cap=breaks_cap, breaks_pen=breaks_pen,
                )
                if bi is None:
                    continue
                c1, c11, pos = bi
                c2 = _LAMBDA * float(T[0, u]) - c1
                if best_choice is None or c2 > best_choice[0]:
                    best_choice = (c2, u, pos)
            if best_choice is None:
                break
            _, u, pos = best_choice
            route = route[:pos] + [u] + route[pos:]
            route_load += int(lookup[u].demand)
            unrouted.discard(u)
        routes.append(route)
    return routes


def solve(inst: Instance, settings: Settings,
          budget_seconds: float = 2.0, seed: int = 0,
          *, nearest_k: int = 24, polish: bool = True,
          cost_aware: bool = True,
          n_starts: int | None = None) -> Solution:
    """Solomon I1 sequential insertion construction (final form).

    See module docstring for the calibrated bench truth and the design
    rationale for each knob.

    Budget is best-effort. `seed` unused except for API parity (I1 is
    deterministic given the same instance + params).

    `nearest_k`: at each step within a route, only consider this many
    unrouted customers (nearest to a route endpoint) as insertion
    candidates. Default 24 keeps the scoring loop O(K' * L) instead of
    O(N * L). Set to 0 (or >= N) for full enumeration.

    `polish`: enable end-of-construction per-route two_opt_intra polish.

    `cost_aware`: when True AND ``settings.economics`` has embargo
    constraints active, add the per-customer embargo penalty to the I1
    insertion score c1. This makes I1 actively avoid embargo windows at
    construction time, closing the iter-7-v4-embargo gap (where vanilla
    v4 lost 0/6 to pyvrp under embargo cost term). Cheap single-customer
    check; doesn't account for push-forward into other customers'
    embargo overlap. Default True (no-op when no cost term active).

    `n_starts`: number of construction passes with different seed
    strategies (farthest-from-depot, earliest-due, highest-demand). The
    best solution under the full evaluator cost is returned. Approximates
    HGS's randomized restart behaviour at low cost.

    When ``n_starts is None`` (default), auto-select: use 3 starts when a
    cost-term creates seed-strategy spread (currently: embargo active),
    else 1 start. Multi-start trades polish budget for seed diversity --
    helpful when alternative basins exist (cost-term workloads), neutral-
    to-hurt at baseline (single-start + more polish wins).
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

    # Constraint-aware insertion params: pre-extract from settings so we
    # don't hit the pydantic attribute access cost in the inner loop.
    embargo_windows: tuple = ()
    embargo_pen = 0.0
    breaks_cap = 0.0
    breaks_pen = 0.0
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
        # iter-7-v4-breaks-aware: per-route driver_breaks penalty
        # (per-route cap on cumulative driving-minutes; closes the
        # iter-7-v4-driver_breaks gap where v4's tighter-K solutions
        # had more cap violations than pyvrp's looser-K ones).
        if (e.driving_max_minutes > 0.0
                and e.break_violation_penalty_per_min > 0.0):
            breaks_cap = float(e.driving_max_minutes)
            breaks_pen = float(e.break_violation_penalty_per_min)

    # Multi-start dispatch: pick the best of n_starts construction passes,
    # each with a different seed strategy. n_starts=1 reproduces the
    # original single-pass I1 with farthest-from-depot seed.
    #
    # iter-7-v4-multistart-budget-fix (2026-05-16): the FIRST pass gets up
    # to the full remaining budget (guaranteed-completion semantics);
    # subsequent passes share whatever time is left. Without this, at
    # n_starts=3 with a tight budget all three passes truncate mid-
    # construction and leave many customers unrouted -- we'd return the
    # best of three partial solutions, which can be worse than just one
    # complete solution. Paris-N500 8s standalone: pre-fix produced 145
    # unrouted; post-fix correctly produces K=16 complete solution.
    #
    # Auto-select n_starts when caller didn't specify: 3 starts when
    # there's seed-strategy spread to exploit (cost-term active),
    # else 1 start (single-start + polish > multi-start - polish at
    # baseline workloads).
    if n_starts is None:
        has_cost_term = (
            (embargo_windows and embargo_pen > 0.0)
            or (breaks_cap > 0.0 and breaks_pen > 0.0)
        )
        n_starts = 3 if has_cost_term else 1
    n_starts = max(1, int(n_starts))
    strategy_names = list(_SEED_STRATEGIES.keys())[:n_starts]
    best_routes: Optional[list[list[int]]] = None
    best_cost = math.inf
    for s_idx, strat in enumerate(strategy_names):
        time_left = deadline - time.perf_counter()
        if time_left < 0.1:
            break
        if s_idx == 0:
            # First pass: up to full remaining budget (will return early
            # when construction naturally finishes -- typically 1.5-5s).
            start_deadline = deadline
        else:
            # Remaining passes share the leftover time equally.
            remaining_passes = len(strategy_names) - s_idx
            start_deadline = time.perf_counter() + time_left / remaining_passes
        seed_fn = _SEED_STRATEGIES[strat]
        try:
            cand_routes = _construct_one_pass(
                inst, settings, start_deadline,
                seed_fn=seed_fn,
                lookup=lookup, capacity=capacity, max_routes=max_routes,
                cust_ids=cust_ids, nearest_k=nearest_k,
                embargo_windows=embargo_windows, embargo_pen=embargo_pen,
                breaks_cap=breaks_cap, breaks_pen=breaks_pen,
            )
        except Exception:
            continue
        if not cand_routes:
            continue
        # Score under the active settings (so cost-term workloads pick the
        # construction that already minimises the penalty).
        try:
            cand_sol_routes = [Route(customers=list(r)) for r in cand_routes if r]
            while len(cand_sol_routes) < inst.num_vehicles:
                cand_sol_routes.append(Route(customers=[]))
            cand_sol = Solution(
                instance_id=inst.instance_id, routes=cand_sol_routes,
                solver="fast_construct_v4_cand",
                wall_clock_seconds=0.0,
                budget_seconds=float(budget_seconds),
                feasible=False,
            )
            cand_sol.metrics = evaluate(inst, cand_sol, settings)
            cand_cost = float(cand_sol.metrics.get("operational_cost", math.inf))
        except Exception:
            cand_cost = math.inf
        if cand_cost < best_cost:
            best_cost = cand_cost
            best_routes = cand_routes
    routes = best_routes if best_routes is not None else []

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

    # Final Solution build (post per-route polish; cost-aware insertion
    # has already shaped the routes via c1 during construction).
    sol_routes = [Route(customers=list(r)) for r in routes if r]
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
    # Recompute unrouted from the FINAL routes (multi-start has its own
    # unrouted sets per pass; the outer `unrouted` is the initial superset).
    placed = set()
    for r in sol_routes:
        placed.update(r.customers)
    sol.metrics["i1_unrouted"] = int(len(set(cust_ids) - placed))
    return sol


if __name__ == "__main__":
    import argparse
    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-polish", action="store_true")
    p.add_argument("--nearest-k", type=int, default=24)
    p.add_argument("--n-starts", type=int, default=1,
                   help="Multi-start: number of seed strategies to try "
                        "(1=farthest only, 2=+earliest_due, 3=+highest_demand). "
                        "Best by full-objective cost wins. solve() default "
                        "auto-selects 3 with cost terms, 1 baseline.")
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), budget_seconds=args.budget,
                seed=args.seed, polish=not args.no_polish,
                nearest_k=args.nearest_k,
                n_starts=args.n_starts)
    print(f"K={int(sol.metrics['num_vehicles_used'])} "
          f"cost=${sol.metrics['operational_cost']:.1f} "
          f"feas={int(sol.feasible)} "
          f"wall={sol.wall_clock_seconds:.2f}s "
          f"unrouted={sol.metrics.get('i1_unrouted', 0)}")
