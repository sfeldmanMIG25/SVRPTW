"""Extract a fixed-dim state vector from (Instance, Solution) for bandit / RL.

SPEC-3-OPSEL-01.  16 dims, normalised, designed so the bandit can learn
operator-class affinities (e.g. "merge_routes when route-count high",
"SISR when plateauing").
"""
from __future__ import annotations

import numpy as np

from svrptw.io import Instance
from svrptw.solvers.common.solution import Solution

_FEATURE_NAMES = (
    "n_norm",                  # N / 500
    "k_used_frac",             # routes_used / num_vehicles
    "mean_route_len_norm",     # mean route length / 20
    "std_route_len_norm",      # std route length / 20
    "min_route_len_norm",
    "max_route_len_norm",
    "missed_frac",             # missed / N
    "asym_score",              # already 0..1
    "cost_vs_greedy",          # current_cost / greedy_baseline
    "frac_short_routes",       # fraction of routes with <= 3 customers
    "frac_empty_vehicles",     # empty / num_vehicles
    "tw_tightness",            # mean(due-ready) / day length
    "depot_dist_mean_norm",    # mean depot-to-customer / max
    "time_used_frac",          # ops applied / budget proxy
    "plateaus",                # 0..1 — running counter of non-improving ops
    "asym_x_n",                # interaction: asym_score * n_norm
)


def featurize(inst: Instance, sol: Solution,
              greedy_cost: float | None = None,
              ops_applied: int = 0,
              plateaus_so_far: int = 0) -> np.ndarray:
    n = inst.num_customers
    K = inst.num_vehicles
    route_lens = np.array([len(r.customers) for r in sol.routes if r.customers])
    K_used = len(route_lens)

    if route_lens.size:
        mean_rl, std_rl, min_rl, max_rl = float(route_lens.mean()), float(route_lens.std()), int(route_lens.min()), int(route_lens.max())
        frac_short = float((route_lens <= 3).sum()) / len(route_lens)
    else:
        mean_rl = std_rl = 0.0
        min_rl = max_rl = 0
        frac_short = 0.0

    missed = sol.metrics.get("missed_deliveries", 0.0)

    tw_widths = np.array([c.due - c.ready for c in inst.customers])
    tw_tight = float(tw_widths.mean()) / max(1.0, float(inst.depot.due - inst.depot.ready))

    T = inst.travel_time
    depot_dist_mean = float(T[0, 1:].mean()) / max(1.0, float(T.max()))

    cost = sol.metrics.get("operational_cost", 0.0)
    cost_ratio = (cost / greedy_cost) if greedy_cost and greedy_cost > 0 else 1.0

    asym = float(getattr(inst, "asymmetry_score", 0.0))

    return np.array([
        n / 500.0,
        K_used / max(1, K),
        mean_rl / 20.0,
        std_rl / 20.0,
        min_rl / 20.0,
        max_rl / 20.0,
        float(missed) / max(1, n),
        asym,
        cost_ratio,
        frac_short,
        (K - K_used) / max(1, K),
        tw_tight,
        depot_dist_mean,
        ops_applied / 50.0,
        plateaus_so_far / 10.0,
        asym * n / 500.0,
    ], dtype=np.float64)


FEATURE_DIM = len(_FEATURE_NAMES)
