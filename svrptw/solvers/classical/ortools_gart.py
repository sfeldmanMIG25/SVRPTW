"""OR-Tools augmented with GART pre-clustering + post-rescore.  SPEC-2-OR-01.

Strategy:
  pass 1 (warm start)   : GART k-medoid cluster the customers, build per-cluster
                          candidate routes, feed as InitialAssignment.
  pass 2 (CP solve)     : run OR-Tools at (70%) of the budget.
  pass 3 (GART rescore) : try cheap swaps using nearest-insertion deltas;
                          accept only if total cost decreases.
"""
from __future__ import annotations

import time

import numpy as np

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.models.gart import get_default_estimator
from svrptw.solvers.classical import ortools_solver as base_ortools
from svrptw.solvers.common import Route, Solution, evaluate


def _gart_kmedoid(inst: Instance, k: int, seed: int) -> list[list[int]]:
    """Cheap k-medoid clustering with GART marginal as the cost.  Returns
    a list of customer-id lists, one per medoid."""
    del seed  # reserved for stochastic-init variants; deterministic for now
    customer_ids = np.array([c.id for c in inst.customers])
    if k >= len(customer_ids):
        return [[int(c)] for c in customer_ids]

    # Initial medoids = farthest-point sampling using row-mean distances from depot.
    T = inst.travel_time
    score = T[0, customer_ids].copy()
    medoids = [int(customer_ids[np.argmax(score)])]
    while len(medoids) < k:
        dists = np.array([min(T[c, m] + T[m, c] for m in medoids) for c in customer_ids])
        idx = int(np.argmax(dists))
        medoids.append(int(customer_ids[idx]))

    # Assign each customer to its nearest medoid (asymmetric: average both legs).
    for _ in range(10):  # a few Lloyd-style passes
        clusters: list[list[int]] = [[] for _ in medoids]
        for cid in customer_ids:
            costs = [T[m, cid] + T[cid, m] for m in medoids]
            clusters[int(np.argmin(costs))].append(int(cid))
        # Re-pick medoid as the customer that minimizes sum of intra-cluster cost.
        new_medoids: list[int] = []
        for grp in clusters:
            if not grp:
                new_medoids.append(medoids[len(new_medoids)])
                continue
            sub = np.array(grp)
            costs = np.array([np.sum(T[c, sub] + T[sub, c]) for c in sub])
            new_medoids.append(int(sub[int(np.argmin(costs))]))
        if new_medoids == medoids:
            break
        medoids = new_medoids

    return [grp for grp in clusters if grp]


def _route_from_cluster(inst: Instance, cluster: list[int]) -> Route:
    """Order a cluster as a TSP via nearest-neighbor from the depot."""
    T = inst.travel_time
    remaining = set(cluster)
    cur = 0
    seq: list[int] = []
    while remaining:
        nxt = min(remaining, key=lambda c: T[cur, c])
        seq.append(int(nxt))
        remaining.discard(nxt)
        cur = nxt
    return Route(customers=seq)


def _gart_rescore(inst: Instance, sol: Solution, settings: Settings,
                  budget_seconds: float) -> Solution:
    """Try `swap one customer between two adjacent routes` moves; accept if
    operational cost drops.  Uses GART marginals to rank candidates."""
    est = get_default_estimator()
    T = inst.travel_time
    deadline = time.perf_counter() + budget_seconds

    best = sol
    best_cost = sol.metrics["operational_cost"]
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        for r_from_idx in range(len(best.routes)):
            for r_to_idx in range(len(best.routes)):
                if r_from_idx == r_to_idx:
                    continue
                if time.perf_counter() >= deadline:
                    break
                r_from = best.routes[r_from_idx]
                if not r_from.customers:
                    continue
                # rank from-route customers by GART marginal contribution
                marginals = [
                    est.estimate_marginal(
                        np.array([c for c in r_from.customers if c != cid], dtype=np.int64),
                        cid, dist_matrix=T,
                    )
                    for cid in r_from.customers
                ]
                # candidate: the one whose removal saves the most.
                idx = int(np.argmax(marginals))
                cid = r_from.customers[idx]

                new_routes = [Route(list(r.customers)) for r in best.routes]
                new_routes[r_from_idx].customers.pop(idx)
                new_routes[r_to_idx].customers.append(cid)
                cand = Solution(
                    instance_id=inst.instance_id, routes=new_routes,
                    solver=best.solver, wall_clock_seconds=best.wall_clock_seconds,
                    budget_seconds=best.budget_seconds, feasible=False,
                )
                cand.metrics = evaluate(inst, cand, settings)
                cand.feasible = bool(cand.metrics["feasible"])
                if cand.metrics["operational_cost"] < best_cost - 1e-6:
                    best = cand
                    best_cost = cand.metrics["operational_cost"]
                    improved = True
                    break
            if improved:
                break
    return best


def solve(inst: Instance, settings: Settings, budget_seconds: float = 10.0) -> Solution:
    t0 = time.perf_counter()
    cp_budget = budget_seconds * 0.7
    rescore_budget = budget_seconds * 0.3

    # Currently the base OR-Tools solver does not accept a warm start; we run it,
    # then rescore.  When SPEC-2-OR-01 graduates to InitialAssignment support,
    # the GART pre-cluster is wired here.
    sol = base_ortools.solve(inst, settings, cp_budget)
    sol.solver = "ortools_gart"
    sol = _gart_rescore(inst, sol, settings, rescore_budget * 0.5)
    from svrptw.solvers.common import relocate, two_opt_intra
    sol = relocate(inst, sol, settings, max_seconds=max(0.5, rescore_budget * 0.35))
    sol = two_opt_intra(inst, sol, settings, max_seconds=max(0.2, rescore_budget * 0.15))
    sol.solver = "ortools_gart"
    sol.wall_clock_seconds = time.perf_counter() - t0
    return sol


if __name__ == "__main__":
    import argparse
    import json

    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=10.0)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), args.budget)
    print(json.dumps({"solver": sol.solver, "budget_s": sol.budget_seconds,
                      "metrics": sol.metrics, "wall_clock_s": sol.wall_clock_seconds}, indent=2))
