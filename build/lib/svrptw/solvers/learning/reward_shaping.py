"""Reward shaping per Phase D4 of the RL Roadmap (Human Reflection 10).

Three shaped reward terms compose with the base improvement-per-second
signal already used by the LinUCB bandit:

  - island_disposal: count of routes with <= 2 customers BEFORE minus AFTER
                     (positive when islands are removed).
  - isolated_stop_disposal: count of customers in solo routes (exactly
                            1 customer) BEFORE minus AFTER. Positive
                            when solo routes are absorbed into others.
  - leg_shrinkage: per-route, sum of edge lengths above the route's
                   median edge length. Sum across routes BEFORE minus
                   AFTER. Positive when above-median legs shrink.

All three are pure-Python, depend only on Solution and Instance
geometry. They are intentionally cheap (no full re-evaluation) so they
can run on every accepted move.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from svrptw.io import Instance
    from svrptw.solvers.common.solution import Solution

DEFAULT_COEFS: dict[str, float] = {
    "island_disposal": 8.0,
    "isolated_stop_disposal": 4.0,
    "leg_shrinkage": 0.05,
}


def _count_islands(sol: "Solution", max_customers: int = 2) -> int:
    """Number of routes with <= max_customers customers (and >= 1)."""
    n = 0
    for r in sol.routes:
        k = len(r.customers)
        if 1 <= k <= max_customers:
            n += 1
    return n


def _count_isolated_stops(sol: "Solution") -> int:
    """Number of customers sitting in solo routes (exactly 1 customer)."""
    return sum(1 for r in sol.routes if len(r.customers) == 1)


def _route_above_median_sum(inst: "Instance", customers: list[int]) -> float:
    """Sum of edge lengths above the median edge in this route.

    Edges include depot stitch: 0 -> c1, c1 -> c2, ..., ck -> 0.
    Empty routes contribute 0. A route with a single customer has only
    two edges (depot->c, c->depot); both are below or at the median, so
    above-median sum is 0 (median equals max → no edges strictly above).
    """
    if not customers:
        return 0.0
    D = inst.travel_dist
    seq = [0] + list(customers) + [0]
    edges: list[float] = [float(D[seq[i], seq[i + 1]])
                          for i in range(len(seq) - 1)]
    if not edges:
        return 0.0
    s = sorted(edges)
    m = len(s)
    if m % 2 == 1:
        med = s[m // 2]
    else:
        med = 0.5 * (s[m // 2 - 1] + s[m // 2])
    return float(sum(e for e in edges if e > med))


def _total_above_median(inst: "Instance", sol: "Solution") -> float:
    return float(sum(_route_above_median_sum(inst, r.customers)
                     for r in sol.routes if r.customers))


def shaped_reward_terms(
    inst: "Instance",
    sol_before: "Solution",
    sol_after: "Solution",
    settings=None,
) -> dict[str, float]:
    """Compute the three shaped-reward terms (Phase D4).

    Positive terms = improvement in the shaped signal. The `settings`
    parameter is accepted for forward-compat (the spec leaves room for
    settings-driven thresholds) but is currently unused.
    """
    del settings  # currently unused; reserved for future tuning hooks.
    isl_before = _count_islands(sol_before)
    isl_after = _count_islands(sol_after)
    iso_before = _count_isolated_stops(sol_before)
    iso_after = _count_isolated_stops(sol_after)
    leg_before = _total_above_median(inst, sol_before)
    leg_after = _total_above_median(inst, sol_after)
    return {
        "island_disposal": float(isl_before - isl_after),
        "isolated_stop_disposal": float(iso_before - iso_after),
        "leg_shrinkage": float(leg_before - leg_after),
    }


def total_shaped_reward(
    base_reward: float,
    terms: dict[str, float],
    coefs: dict[str, float] | None = None,
) -> float:
    """Combine base_reward with weighted shaped terms.

    `coefs=None` uses DEFAULT_COEFS. `coefs={}` (empty dict) is treated
    as "no shaping" → returns base_reward unchanged.
    """
    if coefs is None:
        coefs = DEFAULT_COEFS
    if not coefs:
        return float(base_reward)
    extra = sum(coefs.get(k, 0.0) * float(v) for k, v in terms.items())
    return float(base_reward) + float(extra)
