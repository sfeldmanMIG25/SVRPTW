"""Solution dataclass + deterministic cost evaluator shared across solvers."""
from __future__ import annotations

from dataclasses import dataclass, field

from svrptw.config import Settings
from svrptw.io import Instance


@dataclass
class Route:
    """A single vehicle's route as a list of customer ids (1-indexed).
    Excludes the depot endpoints; the evaluator stitches those on."""
    customers: list[int] = field(default_factory=list)


@dataclass
class Solution:
    instance_id: str
    routes: list[Route]
    solver: str
    wall_clock_seconds: float
    budget_seconds: float
    feasible: bool
    metrics: dict[str, float] = field(default_factory=dict)
    git_sha: str = ""

    @property
    def num_vehicles_used(self) -> int:
        return sum(1 for r in self.routes if r.customers)

    def visited_customer_ids(self) -> set[int]:
        out: set[int] = set()
        for r in self.routes:
            out.update(r.customers)
        return out


def evaluate(inst: Instance, sol: Solution, settings: Settings) -> dict[str, float]:
    """Deterministic operational cost on the asymmetric travel-time matrix.
    Returns dict {operational_cost, total_distance_miles, total_time_minutes,
    missed_deliveries, hard_late_penalty_minutes}."""
    T = inst.travel_time
    D = inst.travel_dist
    served: set[int] = set()
    total_time = 0.0
    total_dist = 0.0
    total_late = 0.0
    total_early_wait = 0.0
    # Inner-loop hot path: reuse the shared per-instance cust dict cache.
    from svrptw.solvers.common.local_search import _cust_by_id
    cust_by_id = _cust_by_id(inst)

    for r in sol.routes:
        if not r.customers:
            continue
        # Stitch: depot(0) -> c1 -> c2 -> ... -> depot(0).  A customer whose
        # arrival is past its `due` is treated as a missed delivery (we still
        # count travel cost to and from it, but the customer isn't 'served').
        prev = 0
        clock = float(inst.depot.ready)
        for cid in r.customers:
            cust = cust_by_id[cid]
            tt = float(T[prev, cid])
            td = float(D[prev, cid])
            total_time += tt
            total_dist += td
            arrive = clock + tt
            if arrive < cust.ready:
                total_early_wait += (cust.ready - arrive)
                clock = cust.ready
            else:
                clock = arrive
            if clock > cust.due:
                total_late += (clock - cust.due)
                # Missed: skip service time, do not count as served.
            else:
                clock += cust.service
                served.add(cid)
            prev = cid
        # Return to depot
        total_time += float(T[prev, 0])
        total_dist += float(D[prev, 0])

    missed = inst.num_customers - len(served)
    e = settings.economics

    # Per-route capacity violation. Compute once; reused by both the
    # feasibility flag and the hard overload penalty. The original
    # evaluator silently accepted routes whose total demand exceeded
    # vehicle_capacity — LKH-3's wrapper exploited this and reported
    # cost wins on capacity-cheating solutions. Fix: penalise overload
    # at hard_late_penalty per unit-of-overflow-demand AND drop the
    # feasibility flag.
    cap = float(inst.vehicle_capacity) if inst.vehicle_capacity else float("inf")
    route_loads: list[float] = []
    total_overload = 0.0
    for r in sol.routes:
        if not r.customers:
            continue
        load = float(sum(cust_by_id[cid].demand for cid in r.customers))
        route_loads.append(load)
        if load > cap:
            total_overload += (load - cap)

    cost = (e.wage_per_minute * total_time
            + e.cost_per_mile * total_dist
            + e.early_wait_per_minute * total_early_wait
            + e.hard_late_penalty * missed
            + e.wage_per_minute * total_late
            + e.hard_late_penalty * total_overload)

    # SPEC-7-COST-01 — opt-in: under-utilised routes get an exponential
    # penalty; cross-route utilisation variance gets a symmetry penalty.
    # All gated on non-zero coefficients → bit-identical when disabled.
    if e.underutil_penalty_per_route > 0.0 or e.symmetry_penalty_coef > 0.0:
        utils = [min(1.0, ld / cap) for ld in route_loads] if route_loads else []
        if utils:
            if e.underutil_penalty_per_route > 0.0:
                target = e.underutil_target_util
                exp = e.underutil_exponent
                for u in utils:
                    shortfall = max(0.0, target - u)
                    cost += e.underutil_penalty_per_route * (shortfall ** exp)
            if e.symmetry_penalty_coef > 0.0 and len(utils) >= 2:
                mu = sum(utils) / len(utils)
                var = sum((u - mu) ** 2 for u in utils) / len(utils)
                cost += e.symmetry_penalty_coef * var

    # SPEC-7-COST-02 — per-route fixed cost. Pushes solvers toward
    # fewer routes when enabled. Gated on > 0 → bit-identical when off.
    if e.per_route_fixed_cost > 0.0:
        cost += e.per_route_fixed_cost * sol.num_vehicles_used

    return {
        "operational_cost": cost,
        "total_distance_miles": total_dist,
        "total_time_minutes": total_time,
        "missed_deliveries": float(missed),
        "tw_late_minutes": total_late,
        "early_wait_minutes": total_early_wait,
        "num_vehicles_used": float(sol.num_vehicles_used),
        "capacity_overload": total_overload,
        "feasible": float(missed == 0 and total_late == 0 and total_overload == 0),
    }
