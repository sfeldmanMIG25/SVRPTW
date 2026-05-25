"""split_route operator -- inverse of merge_routes.

Diagnosed iter-7-v4-stack16-diagnosis (2026-05-16): under per-route cost
terms (driver_breaks, shift_overrun, EV_range), the bandit's existing
operator pool cannot escape v4's "greedy-locally-optimal" basin. v4 wall
plateaus at 45s of 300s budget on Manhattan-N1000 stack-16.

The hypothesis: the pool has `merge_routes` (combine two -> one) but no
inverse. Under per-route cost terms, the bandit needs a way to SPLIT a
long route (high cumulative driving / over the cap) into two shorter
routes that don't violate the per-route cap.

This operator:
  1. Sort routes by longest driving-time first
  2. For each candidate route (length >= 4), try splitting at midpoint
     and at the point of largest TW slack (highest wait time)
  3. Build candidate solution by replacing the route with the two halves
  4. Accept if total cost decreases under the active settings

Signature matches the portfolio's operator contract:
    split_route(inst, sol, settings, max_seconds) -> Solution
"""
from __future__ import annotations

import time
from copy import deepcopy

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common.solution import Route, Solution, evaluate


def _route_driving(inst: Instance, customers: list[int]) -> float:
    """Sum of travel-time edges (depot->first, between, last->depot)."""
    if not customers:
        return 0.0
    T = inst.travel_time
    total = float(T[0, customers[0]])
    for k in range(1, len(customers)):
        total += float(T[customers[k - 1], customers[k]])
    total += float(T[customers[-1], 0])
    return total


def _largest_tw_gap_idx(inst: Instance, customers: list[int]) -> int:
    """Position in customers where the gap between (prev_close, this_ready)
    is largest -- a natural split point with minimal disruption to TWs.
    Returns the index k such that the new split would be customers[:k] and
    customers[k:]. Falls back to midpoint if no clear gap."""
    if len(customers) < 4:
        return len(customers) // 2
    T = inst.travel_time
    best_k = len(customers) // 2
    best_gap = -1.0
    for k in range(1, len(customers)):
        prev_cust = inst.customers[customers[k - 1] - 1]
        prev_close = float(prev_cust.due) + float(prev_cust.service)
        cur_cust = inst.customers[customers[k] - 1]
        cur_ready = float(cur_cust.ready)
        travel = float(T[customers[k - 1], customers[k]])
        gap = cur_ready - (prev_close + travel)
        if gap > best_gap:
            best_gap = gap
            best_k = k
    return best_k


def split_route(inst: Instance, sol: Solution, settings: Settings,
                max_seconds: float = 1.0) -> Solution:
    """Try splitting long routes into two; accept profitable splits.

    Always returns a Solution (input if no profitable split found).
    Deadline-aware: exits early when wall time exhausted.
    """
    deadline = time.perf_counter() + max(0.05, float(max_seconds))
    # Evaluate input if not already done
    if not getattr(sol, "metrics", None) or "operational_cost" not in sol.metrics:
        sol.metrics = evaluate(inst, sol, settings)
    best_sol = sol
    best_cost = float(sol.metrics["operational_cost"])
    # Candidate routes: non-empty, length >= 4 (need at least 2-customer halves),
    # sorted by longest driving time first (those most likely violating caps).
    candidates = []
    for ri, r in enumerate(sol.routes):
        if not r.customers or len(r.customers) < 4:
            continue
        candidates.append((ri, _route_driving(inst, r.customers)))
    candidates.sort(key=lambda x: -x[1])
    # Try each, with two split strategies: midpoint and largest-TW-gap
    for ri, _ in candidates:
        if time.perf_counter() >= deadline:
            break
        custs = list(sol.routes[ri].customers)
        mid = len(custs) // 2
        gap_idx = _largest_tw_gap_idx(inst, custs)
        split_points = [mid]
        if gap_idx != mid:
            split_points.append(gap_idx)
        for split_k in split_points:
            if time.perf_counter() >= deadline:
                break
            if split_k < 2 or split_k > len(custs) - 2:
                continue
            left = custs[:split_k]
            right = custs[split_k:]
            # Build a candidate solution: replace route ri with left,
            # insert right into the first empty slot (or append).
            new_routes = [Route(customers=list(r.customers)) for r in best_sol.routes]
            new_routes[ri] = Route(customers=left)
            placed = False
            for j, rj in enumerate(new_routes):
                if j != ri and not rj.customers:
                    new_routes[j] = Route(customers=right)
                    placed = True
                    break
            if not placed:
                new_routes.append(Route(customers=right))
            new_sol = Solution(
                instance_id=inst.instance_id, routes=new_routes,
                solver=getattr(sol, "solver", "split_route_cand"),
                wall_clock_seconds=0.0,
                budget_seconds=float(max_seconds),
                feasible=False,
            )
            new_sol.metrics = evaluate(inst, new_sol, settings)
            cand_cost = float(new_sol.metrics["operational_cost"])
            if cand_cost < best_cost - 1e-6:
                best_sol = new_sol
                best_cost = cand_cost
                best_sol.feasible = bool(new_sol.metrics.get("feasible", False))
                # First-improvement: take it and exit inner loop
                break
    return best_sol
