"""SPEC-6-LOGIC-01 — 32-d solution-feature extractor for the LogicStudent.

The features capture *structural* properties of a (instance, solution)
pair that a small MLP can use to predict dispatcher acceptance.

Output: torch.float32 tensor of shape (32,). All features are
normalised so that the student doesn't need a separate scaler.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from svrptw.io import Instance
from svrptw.solvers.common import Solution


FEATURE_DIM = 32


def _route_load(inst: Instance, customers: list[int], cust_by_id: dict) -> float:
    return float(sum(cust_by_id[c].demand for c in customers))


def _route_length(inst: Instance, customers: list[int]) -> float:
    """Sum of travel-distance along the route + depot return."""
    if not customers:
        return 0.0
    T = inst.travel_dist
    total = float(T[0, customers[0]])
    for i in range(len(customers) - 1):
        total += float(T[customers[i], customers[i + 1]])
    total += float(T[customers[-1], 0])
    return total


def _intra_route_overlap_proxy(inst: Instance, customers: list[int],
                                cust_by_id: dict) -> float:
    """Cheap proxy for visual self-crossing: ratio of route length to
    convex-hull-perimeter of the visited points. A high ratio means
    the route is wiggly relative to its shape — visually messy.
    """
    if len(customers) < 3:
        return 0.0
    coords = np.array(
        [[cust_by_id[c].x, cust_by_id[c].y] for c in customers],
        dtype=np.float32,
    )
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords)
        perim = float(hull.area)  # 2-D Convex Hull `.area` is the perimeter
        rl = _route_length(inst, customers)
        return rl / max(perim, 1e-6)
    except Exception:
        return 0.0


def extract(inst: Instance, sol: Solution) -> torch.Tensor:
    """Return a (32,) float32 tensor of solution features.

    Features grouped by family:
      0-3:    coverage  (frac_served, miss_rate, miss_cost_share, late_share)
      4-9:    route count + utilisation (n_routes_norm, mean_util, util_std,
              min_util, max_util, util_lt_05_frac)
      10-15:  route geometry (mean_route_length_norm, length_std, length_gini,
              mean_overlap_proxy, max_overlap_proxy, longest_dist_share)
      16-21:  customers-per-route (mean, std, min, max, gini, singleton_frac)
      22-27:  time windows (avg_tw_slack, tw_late_norm, early_wait_norm,
              tightest_tw_frac, mean_arrive_vs_ready, mean_arrive_vs_due)
      28-31:  cost decomposition (wage_share, mile_share, miss_share, late_share)
    """
    from svrptw.solvers.common.local_search import _cust_by_id
    cust_by_id = _cust_by_id(inst)

    # Block 0-3: coverage
    n_total = inst.num_customers
    served = sum(len(r.customers) for r in sol.routes)
    missed = max(0, n_total - served)
    frac_served = served / max(1, n_total)
    miss_rate = missed / max(1, n_total)
    cost = max(1e-3, float(sol.metrics.get("operational_cost", 1.0)))
    miss_cost_share = float(sol.metrics.get("missed_deliveries", 0.0)) * 1000.0 / cost
    late_share = float(sol.metrics.get("tw_late_minutes", 0.0)) / max(1.0, cost)

    # Block 4-9: route count + utilisation
    cap = float(inst.vehicle_capacity)
    utils = []
    for r in sol.routes:
        if r.customers:
            utils.append(_route_load(inst, r.customers, cust_by_id) / max(cap, 1.0))
    n_routes_norm = len(utils) / max(1, n_total / 5)  # ~5 cust/route baseline
    if utils:
        mean_util = float(np.mean(utils))
        util_std = float(np.std(utils))
        min_util = float(np.min(utils))
        max_util = float(np.max(utils))
        util_lt_05_frac = float(np.mean(np.array(utils) < 0.5))
    else:
        mean_util = util_std = min_util = max_util = util_lt_05_frac = 0.0

    # Block 10-15: route geometry
    lengths = []
    overlaps = []
    for r in sol.routes:
        if r.customers:
            ln = _route_length(inst, r.customers)
            lengths.append(ln)
            overlaps.append(_intra_route_overlap_proxy(inst, r.customers, cust_by_id))
    if lengths:
        total_len = max(1e-3, sum(lengths))
        mean_route_length_norm = float(np.mean(lengths)) / max(1.0, total_len / max(1, len(lengths)))
        length_std = float(np.std(lengths)) / max(1.0, np.mean(lengths))
        # Gini coefficient on route lengths
        arr = np.sort(np.array(lengths))
        cum = arr.cumsum()
        length_gini = float((2 * np.sum((np.arange(1, len(arr) + 1)) * arr) / (len(arr) * cum[-1])) - (len(arr) + 1) / len(arr))
        mean_overlap_proxy = float(np.mean(overlaps))
        max_overlap_proxy = float(np.max(overlaps))
        longest_dist_share = float(np.max(lengths)) / total_len
    else:
        mean_route_length_norm = length_std = length_gini = 0.0
        mean_overlap_proxy = max_overlap_proxy = longest_dist_share = 0.0

    # Block 16-21: customers per route
    sizes = [len(r.customers) for r in sol.routes if r.customers]
    if sizes:
        mean_sz = float(np.mean(sizes))
        std_sz = float(np.std(sizes))
        min_sz = float(np.min(sizes)) / max(1.0, mean_sz)
        max_sz = float(np.max(sizes)) / max(1.0, mean_sz)
        arr = np.sort(np.array(sizes))
        cum = arr.cumsum()
        size_gini = float((2 * np.sum((np.arange(1, len(arr) + 1)) * arr) / (len(arr) * cum[-1])) - (len(arr) + 1) / len(arr))
        singleton_frac = float(np.mean(np.array(sizes) == 1))
    else:
        mean_sz = std_sz = min_sz = max_sz = size_gini = singleton_frac = 0.0

    # Block 22-27: time windows
    day_len = max(1.0, float(inst.depot.due - inst.depot.ready))
    tw_widths = [(c.due - c.ready) / day_len for c in inst.customers]
    avg_tw_slack = float(np.mean(tw_widths))
    tw_late_norm = float(sol.metrics.get("tw_late_minutes", 0.0)) / day_len / max(1, len(sizes))
    early_wait_norm = float(sol.metrics.get("early_wait_minutes", 0.0)) / day_len / max(1, len(sizes))
    tightest_tw_frac = float(np.mean(np.array(tw_widths) < 0.1))
    # No arrival-time tracking in metrics; approximate with route_time / day_len.
    mean_arrive_vs_ready = float(sol.metrics.get("total_time_minutes", 0.0)) / day_len / max(1, len(sizes))
    mean_arrive_vs_due = mean_arrive_vs_ready  # same proxy

    # Block 28-31: cost decomposition
    e = type("E", (), {"wage_per_minute": 14.5 / 60, "cost_per_mile": 0.50,
                       "hard_late_penalty": 1000.0})()
    time_m = float(sol.metrics.get("total_time_minutes", 0.0))
    dist_m = float(sol.metrics.get("total_distance_miles", 0.0))
    wage_share = e.wage_per_minute * time_m / cost
    mile_share = e.cost_per_mile * dist_m / cost
    miss_share = e.hard_late_penalty * missed / cost
    late_share_cost = e.wage_per_minute * float(sol.metrics.get("tw_late_minutes", 0.0)) / cost

    feats = np.array([
        frac_served, miss_rate, miss_cost_share, late_share,
        n_routes_norm, mean_util, util_std, min_util, max_util, util_lt_05_frac,
        mean_route_length_norm, length_std, length_gini,
        mean_overlap_proxy, max_overlap_proxy, longest_dist_share,
        mean_sz / 10.0, std_sz / 5.0, min_sz, max_sz, size_gini, singleton_frac,
        avg_tw_slack, tw_late_norm, early_wait_norm, tightest_tw_frac,
        mean_arrive_vs_ready, mean_arrive_vs_due,
        wage_share, mile_share, miss_share, late_share_cost,
    ], dtype=np.float32)

    assert feats.shape == (FEATURE_DIM,), f"feature shape {feats.shape} != ({FEATURE_DIM},)"
    # Tame any inf/nan from degenerate solutions.
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    return torch.from_numpy(feats)
