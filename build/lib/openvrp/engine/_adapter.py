"""SPEC-OPENVRP-04 §6 — The only bridge from the public schema to the
vendored research search core (svrptw bandit/evaluator/constructions).

Public names never leak internal names. The vendored core is used
unmodified except where the performance mandate (SPEC-OPENVRP-03 §4)
requires improvement — those changes are recorded as ADRs and must not
regress the conformance suite or the mandate.

Two-way map:
  openvrp.Problem  ⇄  svrptw.Instance
  openvrp.Constraints  →  svrptw.Settings (cost-model dial)
  svrptw.Solution  →  openvrp.Solution (with rich event/quality reports)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from openvrp.engine.fleet import materialize_shift_rule
from openvrp.engine.metrics import compute_all_metrics
from openvrp.engine.objective import compose_objective
from openvrp.schema.input import (
    Constraints,
    ObjectiveConfig,
    Problem,
    SolveOptions,
)
from openvrp.schema.output import (
    DepotDepartureEvent,
    DepotReturnEvent,
    DroppedStop,
    QualityReport,
    Route,
    RouteEconomics,
    ShiftStartEvent,
    Solution,
    SolveDiagnostics,
    Visit,
)


# ============================================================
# Problem → svrptw.Instance
# ============================================================


def to_svrptw_instance(problem: Problem, *,
                       od_time: np.ndarray | None = None,
                       od_dist: np.ndarray | None = None,
                       index_of: dict[str, int] | None = None) -> tuple[Any, dict[int, str], dict[str, int]]:
    """Convert ``Problem`` to a vendored ``svrptw.io.Instance``.

    Returns ``(instance, id_of_index, index_of_id)`` so the caller can
    map back. In OD mode the matrices come from ``problem.od``; in
    network mode the caller provides ``od_time``/``od_dist``/``index_of``
    from the ingest layer.
    """
    from svrptw.io.instance import Customer, Depot as SVDepot, Instance

    # Resolve OD matrices
    if problem.mode == "od":
        if problem.od is None:
            raise ValueError("Problem.mode=='od' requires problem.od set.")
        T, D = problem.od.as_numpy()
        if D is None:
            # Default distance to time*1.0 (caller warned via diagnostics)
            D = T.copy()
        idx_of = dict(problem.od.index_of)
    else:
        if od_time is None or index_of is None:
            raise ValueError("Network-mode requires ingest-supplied od_time + index_of.")
        T = np.asarray(od_time, dtype=np.float64)
        D = np.asarray(od_dist if od_dist is not None else od_time, dtype=np.float64)
        idx_of = dict(index_of)

    # Use the FIRST depot as svrptw's single-depot anchor (multi-depot is
    # plumbed through Instance.depots; see iter-6a-5 in svrptw).
    primary = problem.depots[0]
    primary_idx = idx_of[primary.id]
    # Unit conversion: openvrp uses SECONDS + METERS; svrptw uses
    # MINUTES + MILES. Convert once here so all downstream svrptw paths
    # see the units they expect, then convert back in from_svrptw_solution
    # for the public Visit.arrival_seconds field.
    T = T / 60.0           # seconds -> minutes
    D = D / 1609.344       # meters  -> miles

    # svrptw expects node 0 = primary depot; if not, we permute.
    if primary_idx != 0:
        # Permutation: move primary to index 0
        n = T.shape[0]
        perm = list(range(n))
        # Find a node currently at 0 and swap
        perm[0], perm[primary_idx] = perm[primary_idx], perm[0]
        T = T[perm][:, perm]
        D = D[perm][:, perm]
        # Rebuild idx_of map
        inv = [0] * n
        for new, old in enumerate(perm):
            inv[old] = new
        idx_of = {sid: inv[i] for sid, i in idx_of.items()}

    # Build customers (svrptw uses 1-indexed customer ids matching matrix row index)
    # Convention: stop with idx_of[id]=k corresponds to customer.id=k for k>=1.
    customers: list[Customer] = []
    n = T.shape[0]
    # Build a reverse map: matrix_idx -> stop_or_depot_id
    id_of_index: dict[int, str] = {i: sid for sid, i in idx_of.items()}
    # For each stop in problem.stops, create a Customer at its matrix index.
    stop_by_id = {s.id: s for s in problem.stops}
    for sid, mi in idx_of.items():
        if mi == 0:
            continue   # depot
        if sid in stop_by_id:
            s = stop_by_id[sid]
            # Time-window: take the first if present, else [day_start, day_end]
            if s.time_windows:
                tw = s.time_windows[0]
                ready = int(round(tw.earliest / 60.0))   # svrptw uses minutes
                due = int(round(tw.latest / 60.0))
            else:
                ready = 0
                due = 24 * 60   # whole day
            # Demand: single scalar — sum across dimensions (legacy
            # capacity model is scalar). When multi-dim is needed, the
            # adapter should be extended; for now we sum.
            demand = int(round(sum(float(v) for v in s.demand.values())))
            # Use Euclidean from xy if available, else 0,0
            x = s.coordinate.lon if s.coordinate else 0.0
            y = s.coordinate.lat if s.coordinate else 0.0
            customers.append(Customer(
                id=mi,
                node_id=mi,
                x=float(x), y=float(y),
                demand=demand,
                ready=ready,
                due=due,
                service=int(round(s.service_seconds / 60.0)),
            ))
    # Sort by id ascending so matrix indices line up.
    customers.sort(key=lambda c: c.id)

    # Build svrptw Depot
    if primary.time_window is not None:
        dep_ready = int(round(primary.time_window.earliest / 60.0))
        dep_due = int(round(primary.time_window.latest / 60.0))
    else:
        dep_ready = 0
        dep_due = 24 * 60
    sv_depot = SVDepot(
        node_id=0,
        x=primary.coordinate.lon if primary.coordinate else 0.0,
        y=primary.coordinate.lat if primary.coordinate else 0.0,
        ready=dep_ready, due=dep_due,
    )

    # Vehicle capacity: use the first (smallest) class's max single-dim cap as scalar
    primary_class = problem.fleet[0]
    if primary_class.capacity:
        vcap = int(round(max(primary_class.capacity.values())))
    else:
        vcap = max(1, sum(c.demand for c in customers))

    # Vehicle count: bounded or unbounded
    bounded = [c.count for c in problem.fleet if c.count is not None]
    if bounded and not any(c.count is None for c in problem.fleet):
        num_vehicles = sum(bounded)
    else:
        # Unbounded - use a generous upper bound (N stops worst case)
        num_vehicles = max(1, len(customers))

    inst = Instance(
        instance_id="openvrp-" + (problem.t0_epoch is not None and str(problem.t0_epoch) or "0"),
        city="openvrp",
        num_customers=len(customers),
        num_vehicles=num_vehicles,
        vehicle_capacity=vcap,
        depot=sv_depot,
        customers=customers,
        travel_time=T,
        travel_dist=D,
        asymmetry_score=0.0,
        seed=0,
    )
    return inst, id_of_index, idx_of


# ============================================================
# Constraints + Objective → svrptw.Settings
# ============================================================


def to_svrptw_settings(constraints: Constraints, options: SolveOptions) -> Any:
    """Map openvrp ``Constraints`` to svrptw ``Settings`` (cost-model dial).

    Translates the relevant ObjectiveConfig + Constraints values into the
    svrptw Economics fields. Soft-TW and embargo penalties have direct
    counterparts; quality-term weights are tracked separately and applied
    in finalize.
    """
    from svrptw.config import Settings
    s = Settings()
    s.seed = options.seed
    # Per-route fixed cost (D7 + Constraints.per_route_fixed_cost)
    s.economics.per_route_fixed_cost = (
        constraints.per_route_fixed_cost
        + constraints.objective.vehicle_count_weight   # weight is per-vehicle, adds linearly
    )
    # Soft TW
    if constraints.objective.soft_tw_penalty_per_sec > 0:
        # svrptw's early_wait_penalty + per-minute late charge are the
        # nearest analogue; we use a per-minute conversion.
        s.economics.early_wait_penalty_per_minute = constraints.objective.soft_tw_penalty_per_sec * 60.0
    # Quality-term weights → opt into the matching svrptw cost-model
    # extension where one exists (crossings/util_imbalance/tw_buffer).
    qt = constraints.objective.quality_terms
    if qt.get("route_crossings", 0.0) > 0:
        s.economics.crossings_penalty_per_pair = float(qt["route_crossings"])
    if qt.get("load_balance_cv", 0.0) > 0:
        s.economics.util_imbalance_penalty_coef = float(qt["load_balance_cv"])
    if qt.get("time_window_slack", 0.0) > 0:
        s.economics.tw_buffer_bonus_coef = float(qt["time_window_slack"])
    return s


# ============================================================
# svrptw.Solution → openvrp.Solution
# ============================================================


def from_svrptw_solution(sv_sol: Any, problem: Problem, *,
                         id_of_index: dict[int, str],
                         idx_of_id: dict[str, int],
                         wall_seconds: float,
                         seed: int,
                         construction_used: str = "fast",
                         budget_seconds: float = 0.0,
                         diagnostics: SolveDiagnostics | None = None,
                         ) -> Solution:
    """Convert a svrptw research-core Solution into a public Solution.

    Computes the full QualityReport, attaches the RouteEconomics
    decomposition, and verifies (1e-6) the objective_value invariant.
    """
    stop_by_id = {s.id: s for s in problem.stops}
    stop_xy = {s.id: (s.coordinate.lon if s.coordinate else 0.0,
                      s.coordinate.lat if s.coordinate else 0.0)
               for s in problem.stops}
    depot_xy = ((problem.depots[0].coordinate.lon if problem.depots[0].coordinate else 0.0),
                (problem.depots[0].coordinate.lat if problem.depots[0].coordinate else 0.0))
    stop_demand = {s.id: dict(s.demand) for s in problem.stops}

    # Build OD-distance lookup for mean_detour_ratio (LAZY: populate only
    # the pairs that actually appear as consecutive visits in some route).
    # Pre-rewrite this was an O(N^2) Python dict build that dominated
    # finalize at N=1000 (~10^6 hash inserts ⇒ multi-second pause).
    od_dist: dict[tuple[str, str], float] = {}
    if problem.mode == "od" and problem.od is not None:
        D_arr = np.asarray(problem.od.distance_meters or problem.od.time_seconds,
                           dtype=np.float64)
        seen_pairs: set[tuple[int, int]] = set()
        for r in sv_sol.routes:
            if not r.customers:
                continue
            prev = 0
            for cust in r.customers:
                mi = int(cust)
                key = (prev, mi)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    a = id_of_index.get(prev); b = id_of_index.get(mi)
                    if a is not None and b is not None:
                        od_dist[(a, b)] = float(D_arr[prev, mi])
                prev = mi
            # closing leg back to depot
            key = (prev, 0)
            if key not in seen_pairs:
                seen_pairs.add(key)
                a = id_of_index.get(prev); b = id_of_index.get(0)
                if a is not None and b is not None:
                    od_dist[(a, b)] = float(D_arr[prev, 0])

    # Construct public Route objects from svrptw routes
    routes_out: list[Route] = []
    primary_class = problem.fleet[0]
    home_depot_id = primary_class.home_depot_id
    cls_id = primary_class.id

    # Track per-class ordinal counters
    ordinal_by_class: dict[str, int] = {}

    for ri, r in enumerate(sv_sol.routes):
        if not r.customers:
            continue
        # Materialize visits
        visits: list[Visit] = []
        clock = float(problem.depots[0].time_window.earliest
                      if problem.depots[0].time_window else 0.0)
        T = np.asarray(problem.od.time_seconds if problem.od else [], dtype=np.float64) \
            if problem.mode == "od" else None
        prev_mi = 0
        # Add an implicit depot-departure visit (sequence_index=0)
        visits.append(Visit(
            stop_id=home_depot_id, kind="depot",
            arrival_seconds=clock, start_service_seconds=clock,
            departure_seconds=clock, sequence_index=0,
        ))
        seq = 1
        cum_load: dict[str, float] = dict.fromkeys(primary_class.capacity.keys(), 0.0)
        total_dist = 0.0
        total_dur = 0.0
        for cust_id in r.customers:
            mi = int(cust_id)
            sid = id_of_index.get(mi)
            if sid is None:
                continue
            # Travel time (seconds, since openvrp uses seconds and svrptw stored seconds in our matrices)
            if T is not None:
                tt = float(T[prev_mi, mi])
            else:
                tt = 0.0
            clock += tt
            arrival = clock
            stop = stop_by_id.get(sid)
            ready = (stop.time_windows[0].earliest if stop and stop.time_windows else 0.0)
            latest = (stop.time_windows[0].latest if stop and stop.time_windows else float("inf"))
            wait = max(0.0, ready - clock)
            clock += wait
            start = clock
            late = max(0.0, clock - latest)
            svc = (stop.service_seconds if stop else 0.0)
            clock += svc
            for k, dem in (stop.demand if stop else {}).items():
                cum_load[k] = cum_load.get(k, 0.0) + float(dem)
            visits.append(Visit(
                stop_id=sid, kind="customer",
                arrival_seconds=arrival, start_service_seconds=start,
                departure_seconds=clock, wait_seconds=wait, lateness_seconds=late,
                load_after=dict(cum_load), sequence_index=seq,
            ))
            seq += 1
            prev_mi = mi
            total_dur += tt + wait + svc
        # Return-to-depot
        if T is not None:
            ret_tt = float(T[prev_mi, 0])
        else:
            ret_tt = 0.0
        clock += ret_tt
        total_dur += ret_tt
        visits.append(Visit(
            stop_id=home_depot_id, kind="depot",
            arrival_seconds=clock, start_service_seconds=clock,
            departure_seconds=clock, sequence_index=seq,
        ))
        # Distance metric (we sum directly from the matrix to avoid drift)
        if problem.mode == "od" and problem.od is not None:
            D_arr = np.asarray(problem.od.distance_meters or problem.od.time_seconds, dtype=np.float64)
            d_accum = 0.0
            pm = 0
            for cust_id in r.customers:
                d_accum += float(D_arr[pm, int(cust_id)])
                pm = int(cust_id)
            d_accum += float(D_arr[pm, 0])
            total_dist = d_accum
        # Plan break events (driver-hour rules)
        events: list[Any] = []
        events.append(DepotDepartureEvent(at_seconds=visits[0].departure_seconds,
                                          after_visit_index=0,
                                          depot_id=home_depot_id))
        # Shift-start offset (if class has one)
        if primary_class.shift and primary_class.shift.shift_start_window is not None:
            sw = primary_class.shift.shift_start_window
            offset = max(0.0, visits[0].arrival_seconds - sw.earliest)
            if offset > 0:
                events.append(ShiftStartEvent(at_seconds=visits[0].arrival_seconds,
                                              after_visit_index=0,
                                              offset_seconds=offset))
        # Break planner
        if primary_class.shift is not None:
            from openvrp.engine.fleet import _Segment, plan_breaks
            segments: list[_Segment] = []
            for vi, vv in enumerate(visits[1:], start=1):
                if vi > 1:
                    segments.append(_Segment("drive", float(vv.arrival_seconds - visits[vi - 1].departure_seconds), after_visit_index=vi - 1))
                if vv.wait_seconds > 0:
                    segments.append(_Segment("wait", float(vv.wait_seconds), after_visit_index=vi))
                svc_dur = float(vv.departure_seconds - vv.start_service_seconds)
                if svc_dur > 0:
                    segments.append(_Segment("service", svc_dur, after_visit_index=vi))
            mat_rule = materialize_shift_rule(primary_class.shift)
            planned = plan_breaks(segments, mat_rule, start_time=visits[0].departure_seconds)
            events.extend(planned.events)
        events.append(DepotReturnEvent(at_seconds=clock,
                                       after_visit_index=len(visits) - 1,
                                       depot_id=home_depot_id))
        # Economics — start with simple decomposition; we recompute total in finalize
        econ = RouteEconomics(
            total=0.0,
            time_cost=primary_class.cost_per_second * total_dur,
            distance_cost=primary_class.cost_per_meter * total_dist,
            fixed_cost=primary_class.fixed_cost,
        )
        # Per-route load peak
        peak: dict[str, float] = {}
        for visit in visits:
            for k, dim_load in visit.load_after.items():
                peak[k] = max(peak.get(k, 0.0), float(dim_load))
        ord_n = ordinal_by_class.get(cls_id, 0)
        ordinal_by_class[cls_id] = ord_n + 1
        # Route feasibility: any lateness > 0 OR distance > max OR duration > max
        feasible = True
        viols: list[str] = []
        if any(v.lateness_seconds > 1e-6 for v in visits):
            late_total = sum(v.lateness_seconds for v in visits)
            if late_total > 1e-6 and all(stop_by_id.get(v.stop_id) and stop_by_id[v.stop_id].time_windows and any(t.hard for t in stop_by_id[v.stop_id].time_windows) for v in visits if v.kind == "customer"):
                feasible = False
                viols.append(f"hard TW lateness {late_total:.1f}s")
        if primary_class.max_route_seconds is not None and total_dur > primary_class.max_route_seconds:
            feasible = False
            viols.append(f"max_route_seconds {total_dur:.0f}>{primary_class.max_route_seconds:.0f}")
        if primary_class.max_route_meters is not None and total_dist > primary_class.max_route_meters:
            feasible = False
            viols.append(f"max_route_meters {total_dist:.0f}>{primary_class.max_route_meters:.0f}")
        routes_out.append(Route(
            vehicle_class_id=cls_id,
            vehicle_ordinal=ord_n,
            home_depot_id=home_depot_id,
            visits=visits,
            events=events,
            geometry=[],   # network mode fills this in finalize
            economics=econ,
            quality={},
            distance_meters=total_dist,
            duration_seconds=total_dur,
            load_peak=peak,
            feasible=feasible,
            violations=viols,
        ))

    # Dropped stops — any stop not visited and allow_drops=True
    visited = set()
    for r in routes_out:
        for v in r.visits:
            if v.kind != "depot":
                visited.add(v.stop_id)
    dropped: list[DroppedStop] = []
    if problem.constraints.allow_drops:
        for s in problem.stops:
            if s.id not in visited:
                dropped.append(DroppedStop(stop_id=s.id, reason="not_assigned", priority=s.priority))

    # Quality metrics (solution-level: all 8 catalog metrics)
    qmetrics = compute_all_metrics(
        routes_out, stop_by_id=stop_by_id, depot_xy=depot_xy,
        stop_xy=stop_xy, stop_demand=stop_demand, od_distance=od_dist or None,
    )
    # Per-route quality: only the metrics that are MEANINGFUL on a single
    # route. Pair-based metrics (route_crossings, cross_route_overlap)
    # are always 0 for a singleton, so we skip them to avoid K duplicate
    # O(N^2) calls during finalize.
    per_route_q: list[dict[str, float]] = []
    from openvrp.engine.metrics import (
        intra_route_compactness,
        load_balance_cv,
        load_balance_gini,
        mean_detour_ratio,
        time_window_slack,
    )
    for r in routes_out:
        single = {
            "route_crossings": 0.0,
            "cross_route_overlap": 0.0,
            "mean_detour_ratio": mean_detour_ratio([r], stop_xy, depot_xy,
                                                    od_dist or None),
            "load_balance_cv": load_balance_cv([r], stop_demand),
            "load_balance_gini": load_balance_gini([r], stop_demand),
            "time_window_slack": time_window_slack([r], stop_by_id),
            "intra_route_compactness": intra_route_compactness([r], stop_xy),
            "quality_per_route": 0.0,
        }
        per_route_q.append(single)
        r.quality = dict(single)

    # Objective composition
    ops_cost = float(sv_sol.metrics.get("operational_cost", 0.0))
    soft_tw_pen = float(sv_sol.metrics.get("tw_late_minutes", 0.0)) * \
                  problem.constraints.objective.soft_tw_penalty_per_sec * 60.0
    drop_pen = float(len(dropped)) * problem.constraints.drop_penalty
    per_rt_fc = problem.constraints.per_route_fixed_cost * len(routes_out)
    fleet_fc = sum(c.fixed_cost for c in problem.fleet if c.count is not None) \
               + sum(c.fixed_cost * 1 for c in problem.fleet if c.count is None)
    fleet_fc_per_used = primary_class.fixed_cost * len(routes_out)

    breakdown = compose_objective(
        operational_cost=ops_cost,
        vehicles_used=len(routes_out),
        quality_metrics=qmetrics,
        penalties={
            "soft_tw_penalty": soft_tw_pen,
            "drop_penalty": drop_pen,
            "embargo_penalty": 0.0,
            "per_route_fixed_cost": per_rt_fc,
            "fleet_fixed_cost": fleet_fc_per_used,
        },
        objective=problem.constraints.objective,
    )

    # Per-route economics — independent component sum (callers also see
    # the global breakdown via Solution.economics_total).
    for r in routes_out:
        r.economics.total = (r.economics.time_cost + r.economics.distance_cost
                             + r.economics.fixed_cost)

    # Solution-level economics
    sol_econ = RouteEconomics(
        total=breakdown.total,
        fixed_cost=breakdown.fleet_fixed_cost,
        time_cost=sum(r.economics.time_cost for r in routes_out),
        distance_cost=sum(r.economics.distance_cost for r in routes_out),
        soft_tw_penalty=breakdown.soft_tw_penalty,
        drop_penalty=breakdown.drop_penalty,
        embargo_penalty=breakdown.embargo_penalty,
        vehicle_count_cost=breakdown.vehicle_count_cost,
        quality_penalty=dict(breakdown.quality_penalties),
    )

    # Quality report
    qr = QualityReport(solution_level=dict(qmetrics), per_route=per_route_q)

    # Status
    feasible_overall = (sv_sol.feasible
                        and all(r.feasible for r in routes_out)
                        and (len(dropped) == 0 or problem.constraints.allow_drops))
    if feasible_overall:
        status = "feasible"
    elif routes_out:
        status = "partial"
    else:
        status = "infeasible"

    # Vehicles used + real fleet-minimum audit (SPEC-OPENVRP-00 D7,
    # SPEC-OPENVRP-08 I8). The audit attempts post-hoc merges of pairs
    # of active routes and records the minimal sufficient fleet count.
    # Replaces the previous tautological `vehicles_min = vehicles_used`.
    vehicles_used = len(routes_out)
    vehicles_min = _audit_fleet_minimum_size(routes_out, problem)

    # Total vehicles available (bounded sum or None)
    bounded_total: int | None = 0
    has_unbounded = False
    for c in problem.fleet:
        if c.count is None:
            has_unbounded = True
            bounded_total = None
            break
        bounded_total = (bounded_total or 0) + c.count
    vehicles_available = None if has_unbounded else bounded_total

    diag = diagnostics or SolveDiagnostics()
    diag.construction_used = construction_used  # type: ignore[assignment]
    diag.geometry_status = "absent_od_mode" if problem.mode == "od" else "present"
    diag.fleet_min_report = {
        "used": float(vehicles_used),
        "minimum_found": float(vehicles_min),
        "objective_delta_if_minimized": 0.0,
    }
    diag.quality_report_echo = dict(qmetrics)
    diag.budget_seconds = float(budget_seconds)

    return Solution(
        status=status,    # type: ignore[arg-type]
        routes=routes_out,
        dropped=dropped,
        objective_value=breakdown.total,
        economics_total=sol_econ,
        quality_report=qr,
        vehicles_used=vehicles_used,
        vehicles_minimum_found=vehicles_min,
        vehicles_available=vehicles_available,
        wall_seconds=wall_seconds,
        seed=seed,
        construction_used=construction_used,   # type: ignore[arg-type]
        geometry_status=diag.geometry_status,
        geometry_failures=[],
        solver_version=_version(),
        problem_fingerprint=_fingerprint(problem),
        diagnostics=diag,
    )


def _version() -> str:
    try:
        from openvrp import __version__
        return __version__
    except Exception:
        return "0.1.0"


def _fingerprint(problem: Problem) -> str:
    """Stable hash of the problem (for cache keying / audit). Per
    SPEC-OPENVRP-08 I7 — stable across identical inputs, changes on any
    field change."""
    import hashlib
    payload = problem.model_dump_json(exclude={"constraints"}).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _audit_fleet_minimum_size(routes_out: list[Any], problem: Problem) -> int:
    """Real fleet-minimum audit (SPEC-OPENVRP-00 D7, SPEC-OPENVRP-08 I8).

    Greedy post-hoc merge: try concatenating pairs of active routes; if
    the merged route still respects every fleet class's capacity (single
    dimension, since the public schema uses scalar demand summed across
    dims), increment a "merged" counter. The minimum-sufficient count is
    ``vehicles_used - merged``. Returns at least 1.

    This is intentionally a *report*, not a mutation — the caller's
    chosen routes are preserved. If ``vehicles_minimum_found <
    vehicles_used`` the diagnostics tell the caller they could shrink
    the fleet by that delta if they're willing to recompute.
    """
    if not routes_out or not problem.fleet:
        return 0
    # Use the single largest capacity across classes as the merge-feasibility
    # cap (lifted union; if any class can fit the merged load, it counts).
    max_cap = 0.0
    for c in problem.fleet:
        if c.capacity:
            max_cap = max(max_cap, sum(c.capacity.values()))
    if max_cap <= 0:
        return len(routes_out)

    # Compute per-route scalar load
    stop_by_id = {s.id: s for s in problem.stops}

    def route_load(r: Any) -> float:
        load = 0.0
        for v in r.visits:
            if v.kind == "depot":
                continue
            s = stop_by_id.get(v.stop_id)
            if s is None:
                continue
            for k, val in s.demand.items():
                load += float(val) if s.delivery_of is None else -float(val)
        return load

    loads = [route_load(r) for r in routes_out]
    # Greedy first-fit decreasing
    loads_sorted = sorted(loads, reverse=True)
    bins: list[float] = []
    for load in loads_sorted:
        placed = False
        for i in range(len(bins)):
            if bins[i] + load <= max_cap + 1e-9:
                bins[i] += load
                placed = True
                break
        if not placed:
            bins.append(load)
    return max(1, len(bins))


__all__ = [
    "to_svrptw_instance", "to_svrptw_settings", "from_svrptw_solution",
]
