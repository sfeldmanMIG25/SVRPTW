"""Seed operator: GART-guided ML ruin-and-repair.

Use the GART v4 LightGBM tour-length estimator to score each customer
by its predicted marginal contribution to its current route's tour
length. Ruin the top-k worst (highest predicted-marginal) customers,
then greedy-reinsert at the position of lowest TRUE asymmetric cost.

Why this is structurally distinct from PyVRP's local search:
PyVRP optimizes against the true (asymmetric) distance matrix using
2-opt / relocate / swap. It cannot see WHICH customers are "globally
expensive" relative to the asymptotic-area baseline — it only sees
pairwise edge costs. GART's predicted marginal `L_with - L_without`
captures the asymptotic Beardwood-Halton-Hammersley signal that PyVRP
moves don't expose. So this operator can find moves PyVRP misses.

Rationale: routes that look locally OK to a 2-opt-style search may
still contain a customer whose removal lets the route re-form much
shorter under the area-and-distribution-free estimator's signal —
typically a far-flung outlier or a TW-tight customer forced into a
non-natural position. Ruin-and-repair on those candidates is the
classical Shaw recipe with an ML-driven heuristic for ruin selection.
"""
from __future__ import annotations

import time

import numpy as np

from svrptw.config import Settings
from svrptw.council.proposal import OperatorContext
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate
from svrptw.solvers.common.local_search import _cust_by_id, _route_arrival_and_close


def _gart():
    """Late-load — keeps the seed importable even if the estimator artifact
    is missing on the host."""
    try:
        from svrptw.models.gart import get_default_estimator
        return get_default_estimator()
    except Exception:
        return None


def operator(solution: Solution, context: OperatorContext) -> Solution | None:
    inst: Instance = context.instance
    settings: Settings = context.settings
    deadline = time.perf_counter() + context.deadline_seconds
    cust_by_id = _cust_by_id(inst)
    cap = float(inst.vehicle_capacity) if inst.vehicle_capacity else float("inf")
    D = np.asarray(inst.travel_dist, dtype=np.float64)

    est = _gart()
    if est is None:
        return None  # estimator unavailable on this host
    # Index space: route customer ids are 1..N; depot is 0. GART takes
    # indices into D, so ids are already correct.

    # Score each customer by predicted marginal contribution to its route.
    # marginal = L(route) - L(route - {cid})  → bigger = "more expensive"
    candidates: list[tuple[float, int, int]] = []  # (marginal, route_idx, cid)
    for ri, r in enumerate(solution.routes):
        if not r.customers or len(r.customers) < 2:
            continue
        nodes_full = np.asarray(r.customers, dtype=np.int64)
        try:
            L_full = est.estimate(nodes_full, dist_matrix=D)
        except Exception:
            continue
        for cid in r.customers:
            if time.perf_counter() >= deadline:
                break
            others = np.asarray([c for c in r.customers if c != cid], dtype=np.int64)
            if len(others) == 0:
                continue
            try:
                L_minus = est.estimate(others, dist_matrix=D)
            except Exception:
                continue
            marginal = L_full - L_minus
            candidates.append((marginal, ri, cid))
        if time.perf_counter() >= deadline:
            break

    if not candidates:
        return None
    # Pick top-k expensive customers; small k keeps the repair tractable.
    k = min(3, len(candidates))
    candidates.sort(reverse=True)
    ruined = candidates[:k]
    ruined_ids = {cid for _, _, cid in ruined}
    ruined_origin = {cid: ri for _, ri, cid in ruined}

    # Build the ruined skeleton (remove ruined customers from their routes).
    skeleton: list[list[int]] = [
        [c for c in r.customers if c not in ruined_ids]
        for r in solution.routes
    ]

    best_cost = solution.metrics["operational_cost"]
    # Greedy re-insertion: for each ruined customer (in original order),
    # try every (route, position) and keep the lowest-true-cost feasible
    # insertion. The "true" cost is just the edge-sum delta, which is
    # what evaluate() will see.
    for cid in ruined_ids:
        if time.perf_counter() >= deadline:
            break
        best_route = best_pos = None
        best_delta = float("inf")
        for ri, route in enumerate(skeleton):
            # Capacity pre-check for this route.
            load = sum(cust_by_id[c].demand for c in route) + cust_by_id[cid].demand
            if load > cap:
                continue
            for pos in range(len(route) + 1):
                trial = route[:pos] + [cid] + route[pos:]
                # TW pre-filter.
                ok, _ = _route_arrival_and_close(inst, trial)
                if not ok:
                    continue
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                # Edge-sum delta on the true asymmetric matrix.
                delta = float(D[prev, cid] + D[cid, nxt] - D[prev, nxt])
                if delta < best_delta:
                    best_delta = delta
                    best_route, best_pos = ri, pos
        if best_route is None:
            # Couldn't reinsert — fall back to origin route at end.
            skeleton[ruined_origin[cid]].append(cid)
        else:
            skeleton[best_route].insert(best_pos, cid)

    cand = Solution(
        instance_id=inst.instance_id,
        routes=[Route(customers=rt) for rt in skeleton],
        solver=solution.solver,
        wall_clock_seconds=solution.wall_clock_seconds,
        budget_seconds=solution.budget_seconds, feasible=False,
    )
    cand.metrics = evaluate(inst, cand, settings)
    cand.feasible = bool(cand.metrics["feasible"])
    if cand.metrics.get("capacity_overload", 0.0) > 0.0:
        return None
    if cand.metrics["operational_cost"] < best_cost - 1e-6:
        return cand
    return None
