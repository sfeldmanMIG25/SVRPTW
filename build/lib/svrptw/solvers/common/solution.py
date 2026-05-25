"""Solution dataclass + deterministic cost evaluator shared across solvers."""
from __future__ import annotations

from dataclasses import dataclass, field

from svrptw.config import Settings
from svrptw.io import Instance


@dataclass
class Route:
    """A single vehicle's route as a list of customer ids (1-indexed).
    Excludes the depot endpoints; the evaluator stitches those on.

    iter-5w-bis: optional start_offset_minutes shifts the route's depot-start
    time from depot.ready to depot.ready + start_offset_minutes. Defaults to
    0.0 -> bit-identical to legacy single-shift-start behaviour. Used by the
    iter-5x peak_hour cost term: shifting a route's start out of peak windows
    can avoid the per-segment-of-time surcharge.
    """
    customers: list[int] = field(default_factory=list)
    start_offset_minutes: float = 0.0
    # iter-6a-7-bis: explicit vehicle-class override. -1 = auto (evaluator
    # picks smallest viable class per mixed_fleets). >= 0 = pin this route
    # to that class index (overrides auto-assignment). Lets the class_shift
    # operator upgrade routes to higher-skill / larger classes deliberately,
    # unblocking iter-6a-7 skills bandit-actionability.
    vehicle_class_idx: int = -1
    # iter-6a-5 multi_depot: which depot this route starts/ends at. Default 0
    # = the primary depot (inst.depot). When inst.depots is None, this is
    # ignored. When inst.depots is set, evaluator uses inst.depots[depot_idx]
    # for the route's depot xy + ready/due times.
    depot_idx: int = 0


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


# iter-6a-perf-2: per-route-pair crossings cache. Most operators mutate
# 1-2 routes per call; pairs that don't involve a mutated route hit cache.
# Module-level dict; bounded to ~16k entries via LRU-ish trim. Keys:
# (inst_id, tuple(route_a_customers), tuple(route_b_customers)). Each entry
# stores the integer crossings count for that ordered pair.
_CROSSINGS_PAIR_CACHE: dict[tuple, int] = {}
_CROSSINGS_PAIR_CACHE_MAX = 16384


def _crossings_count_cached(inst: Instance, sol: Solution) -> int:
    """Cached O(N^2) crossings tally, keyed per route-pair on customer-list
    tuples. Hit rate scales with operator locality (relocate/swap touch 1-2
    routes, so K-2 pairs are unchanged and hit cache).

    Performance optimization: segs are lazily built ONLY for routes that
    participate in at least one cache miss. On a fully-cached evaluate
    (identical sol re-evaluation), no segs are built at all.
    """
    from svrptw.metrics.quality import _segments_cross  # type: ignore
    # Pre-build cheap customer-tuples for all active routes (just iterates list of ints)
    route_tuples: list[tuple] = []
    active_routes: list = []
    for r in sol.routes:
        if not r.customers:
            continue
        route_tuples.append(tuple(r.customers))
        active_routes.append(r)

    inst_id = getattr(inst, "instance_id", "")
    n = len(route_tuples)
    # Lazy per-route segs: built on first miss, reused for subsequent misses
    route_segs_lazy: dict[int, list] = {}
    cust_xy = None  # built only if any miss
    depot_xy = None

    def _get_segs(route_idx: int):
        nonlocal cust_xy, depot_xy
        if route_idx in route_segs_lazy:
            return route_segs_lazy[route_idx]
        if cust_xy is None:
            cust_xy = {c.id: (c.x, c.y) for c in inst.customers}
            depot_xy = (inst.depot.x, inst.depot.y)
        r = active_routes[route_idx]
        path = [depot_xy] + [cust_xy[c] for c in r.customers if c in cust_xy] + [depot_xy]
        segs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        route_segs_lazy[route_idx] = segs
        return segs

    total = 0
    misses = 0
    for ai in range(n):
        ta = route_tuples[ai]
        for bj in range(ai + 1, n):
            tb = route_tuples[bj]
            # Canonical order key (smaller tuple first) so (A,B) and (B,A) share
            if ta <= tb:
                key = (inst_id, ta, tb)
            else:
                key = (inst_id, tb, ta)
            cached = _CROSSINGS_PAIR_CACHE.get(key)
            if cached is not None:
                total += cached
                continue
            # Miss: compute and store. Lazy-build segs only as needed.
            misses += 1
            segs_a = _get_segs(ai)
            segs_b = _get_segs(bj)
            pair_cross = 0
            for s1 in segs_a:
                for s2 in segs_b:
                    if _segments_cross(s1[0], s1[1], s2[0], s2[1]):
                        pair_cross += 1
            _CROSSINGS_PAIR_CACHE[key] = pair_cross
            total += pair_cross
    # Bound cache size; trim oldest ~25% if over cap (CPython dict iteration order = insertion)
    if len(_CROSSINGS_PAIR_CACHE) > _CROSSINGS_PAIR_CACHE_MAX:
        n_drop = _CROSSINGS_PAIR_CACHE_MAX // 4
        for k in list(_CROSSINGS_PAIR_CACHE.keys())[:n_drop]:
            del _CROSSINGS_PAIR_CACHE[k]
    return total


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

    # iter-5v: per-route duration tracking (depot-to-depot wall) for the
    # opt-in shift_overrun penalty. Always tracked (cheap list append per
    # route); only consumed when the gating coefs are both non-zero.
    route_durations: list[float] = []
    # iter-5x: per-route peak-window overlap minutes for the opt-in
    # peak-hour surcharge. Computed only when the gating fields are active;
    # otherwise list stays empty and the per-segment loop pays no overhead.
    e_preview = settings.economics
    _peak_active = (
        e_preview.peak_hour_wage_multiplier != 1.0
        and e_preview.peak_window_starts
        and e_preview.peak_window_ends
        and len(e_preview.peak_window_starts) == len(e_preview.peak_window_ends)
    )
    if _peak_active:
        _peak_windows = list(zip(
            e_preview.peak_window_starts, e_preview.peak_window_ends, strict=True
        ))
    else:
        _peak_windows = []
    route_peak_minutes: list[float] = []
    # iter-6a-1: per-route DRIVING-ONLY minutes (travel time, excluding
    # service and wait). Used by the break-violation cost term to model
    # EU 561 / US HOS rest rules. Always tracked (cheap); only consumed
    # when both gating coefs are non-zero.
    route_driving_minutes: list[float] = []
    # iter-6a-4: per-route total distance in miles. Used by EV range
    # penalty. Always tracked (cheap), only consumed when gating fires.
    route_distance_miles: list[float] = []
    # iter-6a-perf -- Phase F lite path. Inline accumulators for tw_buffer
    # mean and util_cv so Phase F coefs don't need to call score_solution()
    # (which is O(N^2) at N>=500 -- the 344x evaluate() slowdown root cause).
    # Only crossings still requires the standalone O(N^2) call below.
    _phase_f_lite = (
        e_preview.crossings_penalty_per_pair != 0.0
        or e_preview.util_imbalance_penalty_coef != 0.0
        or e_preview.tw_buffer_bonus_coef != 0.0
    )
    tw_buffer_sum_total = 0.0
    tw_buffer_n_stops = 0
    # iter-6a-2: count embargo-window violations across all customer arrivals.
    # Gated up-front so the per-arrival check is skipped when term is off.
    _embargo_active = (
        e_preview.embargo_violation_penalty_per_visit != 0.0
        and e_preview.embargo_window_starts
        and e_preview.embargo_window_ends
        and len(e_preview.embargo_window_starts) == len(e_preview.embargo_window_ends)
    )
    if _embargo_active:
        _embargo_windows = list(zip(
            e_preview.embargo_window_starts, e_preview.embargo_window_ends, strict=True
        ))
    else:
        _embargo_windows = []
    total_embargo_violations = 0

    def _in_embargo(t: float) -> bool:
        for ws, we in _embargo_windows:
            if ws <= t < we:
                return True
        return False

    def _peak_overlap(a: float, b: float) -> float:
        """Total minutes of [a,b] that fall inside any peak window."""
        if not _peak_windows or b <= a:
            return 0.0
        total = 0.0
        for ws, we in _peak_windows:
            lo = a if a > ws else float(ws)
            hi = b if b < we else float(we)
            if hi > lo:
                total += hi - lo
        return total

    for r in sol.routes:
        if not r.customers:
            continue
        # Stitch: depot(0) -> c1 -> c2 -> ... -> depot(0).  A customer whose
        # arrival is past its `due` is treated as a missed delivery (we still
        # count travel cost to and from it, but the customer isn't 'served').
        prev = 0
        # iter-6a-5 multi_depot: resolve which depot this route uses. If
        # inst.depots is set and depot_idx is in-range, use that; else fall
        # back to the legacy single inst.depot.
        # FIRST-CUT LIMITATION: travel_time matrix is still single-depot
        # indexed (node 0 = primary depot only). For depot_idx > 0 routes,
        # the depot-to-first-customer travel time is approximated from the
        # primary depot. To get TRUE multi-depot accurate travel, the
        # instance generator must produce one travel_time matrix per depot
        # (or a per-pair matrix). Tracked as a future-work item; the schema +
        # field plumbing is here so application code can use depot_idx for
        # ready/due time differences (different depot opening hours).
        _route_depot = inst.depot
        if inst.depots is not None and r.customers:
            _di = int(getattr(r, "depot_idx", 0))
            if 0 <= _di < len(inst.depots):
                _route_depot = inst.depots[_di]
        # iter-5w-bis: respect optional per-route start offset (default 0.0 ->
        # legacy single-shift-start). Routes that opt-in to a later start trade
        # off later customer arrivals (which may go late on tight TWs) against
        # avoiding peak-hour windows or other time-segment surcharges.
        depot_start = float(_route_depot.ready) + float(getattr(r, "start_offset_minutes", 0.0))
        clock = depot_start
        peak_in_route = 0.0
        driving_in_route = 0.0  # iter-6a-1: travel-only minutes per route
        dist_in_route = 0.0     # iter-6a-4: per-route distance in miles
        for cid in r.customers:
            cust = cust_by_id[cid]
            tt = float(T[prev, cid])
            td = float(D[prev, cid])
            total_time += tt
            driving_in_route += tt
            total_dist += td
            dist_in_route += td
            # Travel segment: [clock, clock + tt]
            if _peak_active:
                peak_in_route += _peak_overlap(clock, clock + tt)
            arrive = clock + tt
            if arrive < cust.ready:
                wait_dur = cust.ready - arrive
                total_early_wait += wait_dur
                # Wait at customer (no movement) -- charge wage but also peak
                # overlap so the bandit can prefer arriving INSIDE peak vs
                # waiting AT peak.
                if _peak_active:
                    peak_in_route += _peak_overlap(arrive, cust.ready)
                clock = cust.ready
            else:
                clock = arrive
            # iter-6a-2: check embargo on actual ARRIVAL/start-of-service clock.
            # If clock falls inside any embargo window, count one violation.
            if _embargo_active and _in_embargo(clock):
                total_embargo_violations += 1
            # iter-6a-perf: inline tw_buffer accumulation for Phase F lite path
            if _phase_f_lite:
                tw_w = max(1e-6, cust.due - cust.ready)
                margin = (cust.due - clock) / tw_w
                wait_pen = ((cust.ready - arrive) / tw_w) if arrive < cust.ready else 0.0
                buf = max(0.0, min(1.0, margin)) - 0.5 * wait_pen
                tw_buffer_sum_total += max(0.0, min(1.0, buf))
                tw_buffer_n_stops += 1
            if clock > cust.due:
                total_late += (clock - cust.due)
                # Missed: skip service time, do not count as served.
            else:
                # Service segment: [clock, clock + service]
                if _peak_active:
                    peak_in_route += _peak_overlap(clock, clock + cust.service)
                clock += cust.service
                served.add(cid)
            prev = cid
        # Return to depot
        ret_tt = float(T[prev, 0])
        ret_td = float(D[prev, 0])
        total_time += ret_tt
        driving_in_route += ret_tt
        total_dist += ret_td
        dist_in_route += ret_td
        if _peak_active:
            peak_in_route += _peak_overlap(clock, clock + ret_tt)
        clock += ret_tt
        route_durations.append(clock - depot_start)
        route_driving_minutes.append(driving_in_route)
        route_distance_miles.append(dist_in_route)
        if _peak_active:
            route_peak_minutes.append(peak_in_route)

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

    # iter-5v -- shift-overrun penalty. Per-route duration cap (depot-to-depot)
    # with linear penalty per minute over. PyVRP can't see this term; the
    # bandit can. Gated on BOTH coefs being non-zero -> bit-identical default.
    if (e.shift_max_minutes > 0.0
            and e.shift_overrun_penalty_per_min > 0.0
            and route_durations):
        cap_min = float(e.shift_max_minutes)
        coef = float(e.shift_overrun_penalty_per_min)
        for dur in route_durations:
            over = dur - cap_min
            if over > 0:
                cost += over * coef

    # iter-5x -- peak-hour wage surcharge. For each peak-overlap minute the
    # route accumulates, charge (multiplier - 1) * wage_per_minute on top of
    # the base wage already paid in `total_time`. PyVRP can't see this; the
    # bandit can shift start times / sequence to push activity OUT of peaks.
    if _peak_active and route_peak_minutes:
        mult = float(e.peak_hour_wage_multiplier)
        surcharge_per_min = (mult - 1.0) * e.wage_per_minute
        if surcharge_per_min != 0.0:
            cost += surcharge_per_min * sum(route_peak_minutes)

    # iter-5y -- cross-route driver-time-fairness penalty. coef * variance of
    # (wage_per_min * route_duration) across active routes. Reuses the
    # route_durations list already computed for iter-5v above. Step 0 check
    # (iter-5x recipe correction): cross-route operators (relocate / swap /
    # two_opt_star / merge_routes) exist in the bandit's arm set so the
    # bandit CAN rebalance route durations. Gated on coef != 0 ->
    # bit-identical default.
    if e.driver_time_variance_penalty_coef != 0.0 and len(route_durations) >= 2:
        wage = e.wage_per_minute
        wages = [wage * d for d in route_durations]
        mu = sum(wages) / len(wages)
        var = sum((w - mu) ** 2 for w in wages) / len(wages)
        cost += float(e.driver_time_variance_penalty_coef) * var

    # iter-6a-1 -- driver break violation (EU 561 / US HOS lite). Per-route
    # driving-only minutes > driving_max_minutes incurs linear penalty per
    # minute over. Step 0 check: split_route + merge_routes operators act
    # on route-driving-time directly -> bandit can break long routes apart
    # so each piece stays under the cap. Gated on BOTH coefs being non-zero
    # -> bit-identical default.
    if (e.driving_max_minutes > 0.0
            and e.break_violation_penalty_per_min > 0.0
            and route_driving_minutes):
        cap = float(e.driving_max_minutes)
        pen = float(e.break_violation_penalty_per_min)
        for drv in route_driving_minutes:
            over = drv - cap
            if over > 0:
                cost += over * pen

    # iter-6a-2 -- hard zones / embargo windows. Charge per-visit penalty for
    # any customer arrival that fell inside an embargo window during the loop.
    if _embargo_active and total_embargo_violations > 0:
        cost += float(e.embargo_violation_penalty_per_visit) * total_embargo_violations

    # iter-6a-4 -- EV range. Per-route total distance > vehicle_range_miles
    # incurs linear penalty per mile over. Gated on BOTH coefs non-zero ->
    # bit-identical default.
    if (e.vehicle_range_miles > 0.0
            and e.range_violation_penalty_per_mile > 0.0
            and route_distance_miles):
        rng = float(e.vehicle_range_miles)
        pen_mi = float(e.range_violation_penalty_per_mile)
        for dist_mi in route_distance_miles:
            over_mi = dist_mi - rng
            if over_mi > 0:
                cost += over_mi * pen_mi

    # iter-6a-3 -- mixed fleets. For each route, find the smallest viable
    # vehicle class (class_capacity >= route load); add that class's fixed +
    # per-mile premium. Gated on (all three tuples non-empty + same length +
    # any premium > 0). Uses route_loads (already computed for capacity
    # overload block) and route_distance_miles (iter-6a-4 plumbing).
    _classes_active = (
        len(e.vehicle_class_capacities) > 0
        and len(e.vehicle_class_capacities) == len(e.vehicle_class_fixed_premiums)
        and len(e.vehicle_class_capacities) == len(e.vehicle_class_per_mile_premiums)
        and (
            any(p != 0 for p in e.vehicle_class_fixed_premiums)
            or any(p != 0 for p in e.vehicle_class_per_mile_premiums)
        )
    )
    # iter-6a-6 -- pickup-and-delivery precedence pairs. Penalty per pair if
    # the pickup and delivery are in different routes OR delivery comes before
    # pickup in the same route. Built up here using a per-customer
    # route+position index for O(1) lookup per pair.
    if (e.pd_pairs_flat
            and e.pd_violation_penalty_per_pair != 0.0
            and len(e.pd_pairs_flat) % 2 == 0):
        # Build customer -> (route_idx, position_in_route) index. Last write
        # wins on duplicates (shouldn't happen in well-formed solutions).
        cust_route_pos: dict[int, tuple[int, int]] = {}
        for ri, r in enumerate(sol.routes):
            for pi, cid in enumerate(r.customers):
                cust_route_pos[cid] = (ri, pi)
        pen_per = float(e.pd_violation_penalty_per_pair)
        flat = e.pd_pairs_flat
        for i in range(0, len(flat), 2):
            pickup = int(flat[i])
            delivery = int(flat[i + 1])
            p = cust_route_pos.get(pickup)
            d = cust_route_pos.get(delivery)
            if p is None or d is None:
                # One of them unserved -- count as violation.
                cost += pen_per
                continue
            if p[0] != d[0] or p[1] >= d[1]:
                # Different routes OR delivery not strictly after pickup.
                cost += pen_per

    # iter-6a-8 -- minimum routes required (labor contract). Penalty per
    # missing route below min_routes_required. Bandit can split routes to
    # increase K and avoid penalty. Gated on BOTH non-zero -> bit-identical
    # default.
    if (e.min_routes_required > 0
            and e.under_min_routes_penalty_per_route > 0.0):
        used_k = int(sol.num_vehicles_used)
        shortage = max(0, int(e.min_routes_required) - used_k)
        if shortage > 0:
            cost += float(e.under_min_routes_penalty_per_route) * shortage

    # iter-6a-7 -- skills (driver-customer matching). Compute per-route assigned
    # class index, then for each customer check if class skill level >=
    # customer's required level. Penalty per mismatch. Reuses mixed_fleets
    # class-assignment logic; if mixed_fleets is off, treats all routes as
    # class 0 (the smallest). Gated on (non-empty pairs AND non-empty class
    # levels AND non-zero penalty).
    _skills_active = (
        len(e.customer_skill_levels_flat) > 0
        and len(e.customer_skill_levels_flat) % 2 == 0
        and len(e.vehicle_class_skill_levels) > 0
        and e.skill_mismatch_penalty_per_visit != 0.0
    )
    # Pre-build the route -> class_idx map if either mixed_fleets or skills
    # need it. Both use the same smallest-viable-class rule UNLESS the route
    # has an explicit vehicle_class_idx override (iter-6a-7-bis).
    route_class_idx: list[int] = []
    if (_classes_active or _skills_active) and route_loads:
        if _classes_active:
            classes_sorted = sorted(
                enumerate(zip(e.vehicle_class_capacities,
                              e.vehicle_class_fixed_premiums,
                              e.vehicle_class_per_mile_premiums, strict=True)),
                key=lambda kv: kv[1][0],
            )
        else:
            # Skills active but mixed_fleets off: assume single (largest)
            # vehicle class index 0.
            classes_sorted = [(0, (float("inf"), 0.0, 0.0))]
        # Walk active routes (parallel to route_loads). Use explicit
        # vehicle_class_idx if set (>=0), else smallest-viable.
        active_routes_for_class = [r for r in sol.routes if r.customers]
        for load, r in zip(route_loads, active_routes_for_class, strict=False):
            explicit = int(getattr(r, "vehicle_class_idx", -1))
            if explicit >= 0:
                # Honour the override (even if it's smaller than viable -- the
                # capacity-overload term in evaluate() will charge separately).
                route_class_idx.append(explicit)
                continue
            assigned = classes_sorted[-1][0]  # default to largest if nothing fits
            for orig_idx, (cap, _f, _pm) in classes_sorted:
                if load <= cap:
                    assigned = orig_idx
                    break
            route_class_idx.append(assigned)

    if _classes_active and route_loads and route_distance_miles and route_class_idx:
        # iter-6a-7-bis: use the route_class_idx map (which respects explicit
        # Route.vehicle_class_idx overrides). Index directly into the
        # original (unsorted) coefficient tuples.
        fixed_t = e.vehicle_class_fixed_premiums
        per_mi_t = e.vehicle_class_per_mile_premiums
        n_cls = len(fixed_t)
        for dist_mi, cls_idx in zip(route_distance_miles, route_class_idx, strict=False):
            if 0 <= cls_idx < n_cls:
                cost += float(fixed_t[cls_idx]) + float(per_mi_t[cls_idx]) * float(dist_mi)

    if _skills_active and route_class_idx:
        # Build customer_id -> required_skill_level dict from flat tuple.
        req = {}
        flat = e.customer_skill_levels_flat
        for i in range(0, len(flat), 2):
            req[int(flat[i])] = int(flat[i + 1])
        class_levels = list(e.vehicle_class_skill_levels)
        n_cls = len(class_levels)
        pen_per = float(e.skill_mismatch_penalty_per_visit)
        # Walk active routes (parallel to route_class_idx). Each customer with
        # required > assigned class level = 1 violation.
        active_routes = [r for r in sol.routes if r.customers]
        for r, cls_idx in zip(active_routes, route_class_idx, strict=False):
            provided = class_levels[cls_idx] if 0 <= cls_idx < n_cls else 0
            for cid in r.customers:
                needed = req.get(cid, 0)
                if needed > provided:
                    cost += pen_per

    # SPEC-F-COST-01 -- opt-in structural cost terms (iter-5l, Phase F).
    # iter-6a-perf REFACTOR: previous version called score_solution() once
    # whenever ANY of the three coefs was non-zero. score_solution() is O(N^2)
    # for inter_route_crossings + silhouette + hulls, causing a 344x evaluate()
    # slowdown at N=500 when Phase F is active. Lite path: compute util_cv
    # and tw_buffer_mean inline (already done in main loop via inline
    # accumulators); only call inter_route_crossings via a focused helper
    # when its coef is non-zero. Crossings still O(N^2) but skips
    # silhouette/hulls/intra_inter, saving ~50% of the original cost.
    if e.util_imbalance_penalty_coef != 0.0 and route_loads:
        cap_f = float(inst.vehicle_capacity) if inst.vehicle_capacity else 1.0
        utils = [min(1.0, ld / cap_f) for ld in route_loads]
        mu_u = sum(utils) / len(utils)
        var_u = sum((u - mu_u) ** 2 for u in utils) / len(utils)
        std_u = var_u ** 0.5
        util_cv = (std_u / mu_u) if mu_u > 0 else 0.0
        cost += e.util_imbalance_penalty_coef * util_cv
    if e.tw_buffer_bonus_coef != 0.0 and tw_buffer_n_stops > 0:
        mean_tw_buf = tw_buffer_sum_total / tw_buffer_n_stops
        cost += -1.0 * e.tw_buffer_bonus_coef * mean_tw_buf
    if e.crossings_penalty_per_pair != 0.0:
        cross = _crossings_count_cached(inst, sol)
        cost += e.crossings_penalty_per_pair * cross

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
