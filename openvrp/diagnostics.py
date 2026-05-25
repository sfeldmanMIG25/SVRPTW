"""SPEC-OPENVRP-08 — Diagnostics helpers and conditional operator
registration (SPEC-OPENVRP-04 §5, NON-OPTIONAL).

The search's operator pool is filtered at solve time to operators whose
target axis is active in the problem. Registering no-op operators
consumes search budget and measurably degrades result magnitude (the
fix recovered the loss and reduced wall in the research record). This
is correctness, not an optimization.
"""
from __future__ import annotations

from typing import Any

from openvrp.schema.input import Problem


# Full pool the research svrptw bandit registers (the 13-arm baseline plus
# the search-decided quality / cost-term operators added in Phase F / iter-6a).
ALL_OPERATORS: tuple[str, ...] = (
    "relocate",
    "swap",
    "two_opt",
    "two_opt_star",
    "sisr",
    "merge_routes",
    "split_route",
    "ejection_chain",
    "or_opt",
    "shift_start",        # peak/embargo windows
    "class_shift",        # mixed fleets / skills
    "depot_shift",        # multi-depot
    "recharge_insert",    # EV range (legacy single-resource)
    "replenish_insert",   # generalized multi-resource depot refill
    "pd_swap",            # PD pairs
    "crossings_unwind",   # quality axis
    "balance_loads",      # quality axis
    "tw_slack_raise",     # quality axis
)


def active_operator_pool(problem: Problem) -> list[str]:
    """Return the subset of ``ALL_OPERATORS`` whose axis is active in the
    problem. Exposed in ``SolveDiagnostics.operator_pool``.

    Rules (matches SPEC-OPENVRP-04 §5):
    - shift_start: present only if peak_windows non-empty OR any zone has
      a time-windowed embargo.
    - class_shift: present only if classes differ on cost/skills.
    - depot_shift: present only if >=2 depots.
    - recharge_insert: present only if any class has max_route_meters.
    - pd_swap: present only if any PD pair exists.
    - <quality>: present only if its quality_terms weight is > 0.
    """
    p = problem
    has_peak = bool(p.constraints.peak_windows)
    has_time_zone = any(z.active_window is not None for z in p.zones)
    classes_differ = (
        len({(c.cost_per_second, c.cost_per_meter, tuple(c.provides_skills))
             for c in p.fleet}) > 1
    )
    multi_depot = len(p.depots) > 1
    needs_recharge = any(c.max_route_meters is not None for c in p.fleet)
    # Generalized resource replenishment: any class declares consumption
    # OR any depot stocks resources (the replenish_insert operator runs
    # to consider depot-as-waypoint mid-route).
    needs_replenish = (any(c.consumes for c in p.fleet)
                       or any(d.resources for d in p.depots))
    has_pd = any(s.pickup_of or s.delivery_of for s in p.stops)
    qw = p.constraints.objective.quality_terms

    pool: list[str] = []
    always = ["relocate", "swap", "two_opt", "two_opt_star",
              "sisr", "merge_routes", "split_route", "ejection_chain", "or_opt"]
    pool.extend(always)
    if has_peak or has_time_zone:
        pool.append("shift_start")
    if classes_differ:
        pool.append("class_shift")
    if multi_depot:
        pool.append("depot_shift")
    if needs_recharge:
        pool.append("recharge_insert")
    if needs_replenish:
        pool.append("replenish_insert")
    if has_pd:
        pool.append("pd_swap")
    if qw.get("route_crossings", 0.0) > 0:
        pool.append("crossings_unwind")
    if qw.get("load_balance_cv", 0.0) > 0 or qw.get("load_balance_gini", 0.0) > 0:
        pool.append("balance_loads")
    if qw.get("time_window_slack", 0.0) > 0:
        pool.append("tw_slack_raise")
    return pool


def triangle_inequality_sample(time_matrix: Any, *, sample_size: int = 200,
                               tolerance: float = 0.05) -> list[str]:
    """Cheap sampled D16 check. Returns list of human-readable strings
    describing gross violations (>tolerance relative shortcut). Empty
    list ⇒ no violations sampled.

    Per SPEC-OPENVRP-06 §1 this is a warning, not a failure: the matrix
    is never silently 'fixed'.
    """
    import random
    import numpy as np
    T = np.asarray(time_matrix, dtype=np.float64)
    n = T.shape[0]
    if n < 3:
        return []
    rng = random.Random(0)
    out: list[str] = []
    for _ in range(sample_size):
        i, j, k = rng.sample(range(n), 3)
        if T[i, j] == 0 or np.isinf(T[i, k]) or np.isinf(T[k, j]):
            continue
        if T[i, k] + T[k, j] < T[i, j] * (1 - tolerance):
            out.append(
                f"D16: T[{i},{j}]={T[i,j]:.0f} > T[{i},{k}]+T[{k},{j}]={T[i,k]+T[k,j]:.0f}"
            )
            if len(out) > 10:
                break
    return out


__all__ = ["ALL_OPERATORS", "active_operator_pool", "triangle_inequality_sample"]
