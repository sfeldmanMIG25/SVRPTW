"""Per-route + per-solution quality metrics for VRPTW.

All metrics are PURE FUNCTIONS of (Instance, Solution). No solver state,
no LLM calls, no I/O. They produce numeric scores that the unified
objective combines with VLM visual scores.

Metric families:
  Clustering:
    - mean_intra / mean_inter ratio (lower = better-clustered)
    - centroid_spread (avg distance from each customer to route centroid)
    - convex_hull_overlap_count (# pairs of routes whose hulls overlap)
    - silhouette-like score per route
  Balancing:
    - load_util_cv (coefficient of variation of route loads)
    - stops_per_route_cv
    - load_gini (Gini on loads)
  Geometric efficiency:
    - detour_ratio (route len / 2 * mean(depot-to-cust dist))
    - avg_self_crossings per route
    - inter_route_crossings (cross-route line intersections)
  Time-window:
    - mean_slack_min (avg due - arrival per stop)
    - tight_stops_frac (frac with <10% TW slack)
    - wait_to_service_ratio
  Route shape:
    - aspect_ratio per route (longest dim / shortest dim of bounding box)
    - mean turn-angle per route (sharpness)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from svrptw.io.instance import Instance
    from svrptw.solvers.common.solution import Solution


@dataclass
class RouteScore:
    route_idx: int
    n_customers: int
    load: float
    util: float
    centroid_x: float
    centroid_y: float
    centroid_spread: float          # mean distance from cust to route centroid
    intra_dist_mean: float          # mean pairwise dist within route
    detour_ratio: float             # route_len / 2*mean_depot_dist
    aspect_ratio: float             # bbox long/short
    self_crossings: int
    mean_turn_angle_rad: float
    tw_slack_min_mean: float
    tight_tw_frac: float            # frac of stops with <10% TW slack
    wait_total_min: float
    # SPEC-METRICS-02 -- TW buffer metrics (per user direction iter-5g)
    tw_buffer_score: float = 0.0    # 0..1, higher=better; rewards arriving
                                    # at TW open without waiting + with margin
                                    # before TW close
    early_arrival_frac: float = 0.0 # frac of stops arriving before ready (waste)
    late_arrival_frac: float = 0.0  # frac of stops arriving after due (violation)
    on_time_arrival_frac: float = 0.0  # frac arriving within [ready, due]


@dataclass
class SolutionQualityScore:
    """Per-solution scorecard. Higher = better for normalized metrics."""
    n_routes: int
    n_customers_served: int
    n_unrouted: int
    routes: list[RouteScore] = field(default_factory=list)
    # Clustering aggregates
    intra_inter_ratio: float = 0.0      # mean_intra / mean_inter   (lower = better)
    silhouette_like: float = 0.0        # in [-1, 1]; higher = better separation
    convex_hull_overlap_count: int = 0  # pairs of routes whose hulls overlap
    # Balancing aggregates
    load_util_cv: float = 0.0           # CV of utilizations across routes
    stops_per_route_cv: float = 0.0
    load_gini: float = 0.0              # 0 = perfectly balanced, 1 = max imbalance
    # Geometric efficiency
    inter_route_crossings: int = 0
    mean_detour_ratio: float = 0.0
    # Time-window
    mean_slack_min: float = 0.0
    tight_stops_frac: float = 0.0
    wait_to_service_ratio: float = 0.0
    # SPEC-METRICS-02 aggregates
    mean_tw_buffer_score: float = 0.0
    total_wait_min: float = 0.0
    on_time_frac: float = 0.0
    early_frac: float = 0.0
    late_frac: float = 0.0
    # Compound (lower is generally better, except where noted)
    quality_index: float = 0.0          # synthesized 0..1; higher = better
    # iter-5u: K-fair complement. quality_index is K-DEPENDENT (Spearman
    # rho with n_routes = +0.78 across 6 v1_large instances), so comparing
    # quality_index across solvers with different K is misleading -- a
    # solver that uses 2x more vehicles will naturally score higher
    # (each route covers fewer customers, fewer crossings, tighter clusters).
    # quality_per_route divides by K to neutralise this artifact. Use it
    # as the K-fair complement when comparing across solvers.
    quality_per_route: float = 0.0      # quality_index / max(n_routes, 1)


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _segments_cross(a1, a2, b1, b2) -> bool:
    """True if segment (a1,a2) crosses (b1,b2). Uses orientation tests."""
    def _ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    d1 = _ccw(b1, b2, a1)
    d2 = _ccw(b1, b2, a2)
    d3 = _ccw(a1, a2, b1)
    d4 = _ccw(a1, a2, b2)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _convex_hull_2d(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain. Returns CCW vertices, no duplicates."""
    pts = sorted(set(pts))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _hulls_overlap(h1: list[tuple[float, float]],
                    h2: list[tuple[float, float]]) -> bool:
    """Crude SAT on convex hulls. False if any separating axis exists."""
    if not h1 or not h2:
        return False
    for hull in (h1, h2):
        for i in range(len(hull)):
            edge = (hull[(i+1) % len(hull)][0] - hull[i][0],
                    hull[(i+1) % len(hull)][1] - hull[i][1])
            axis = (-edge[1], edge[0])  # normal
            proj1 = [(p[0]*axis[0] + p[1]*axis[1]) for p in h1]
            proj2 = [(p[0]*axis[0] + p[1]*axis[1]) for p in h2]
            if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                return False
    return True


def score_routes(inst: "Instance", sol: "Solution") -> list[RouteScore]:
    """Per-route quality scorecard (no aggregation)."""
    cust_xy = {c.id: (c.x, c.y) for c in inst.customers}
    cust_demand = {c.id: c.demand for c in inst.customers}
    cust_ready = {c.id: c.ready for c in inst.customers}
    cust_due = {c.id: c.due for c in inst.customers}
    cust_service = {c.id: getattr(c, "service", 10.0) for c in inst.customers}
    depot_xy = (inst.depot.x, inst.depot.y)
    cap = max(1.0, float(inst.vehicle_capacity))
    day_len = max(1.0, float(inst.depot.due - inst.depot.ready))

    out: list[RouteScore] = []
    for ridx, route in enumerate(sol.routes):
        if not route.customers:
            continue
        pts = [cust_xy[c] for c in route.customers]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        spread = sum(_euclid(p, (cx, cy)) for p in pts) / len(pts)
        # intra-route mean pairwise distance
        if len(pts) > 1:
            n = len(pts); dist_sum = 0.0; pair_count = 0
            for i in range(n):
                for j in range(i+1, n):
                    dist_sum += _euclid(pts[i], pts[j]); pair_count += 1
            intra_mean = dist_sum / pair_count
        else:
            intra_mean = 0.0
        # detour ratio
        path = [depot_xy] + pts + [depot_xy]
        route_len = sum(_euclid(path[i], path[i+1]) for i in range(len(path)-1))
        mean_depot_d = sum(_euclid(p, depot_xy) for p in pts) / max(1, len(pts))
        detour = route_len / max(1e-9, 2 * mean_depot_d)
        # bbox aspect ratio
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        w = max(xs) - min(xs); h = max(ys) - min(ys)
        if w + h > 0:
            aspect = max(w, h) / max(1e-9, min(w, h))
        else:
            aspect = 1.0
        # self-crossings (path segments that cross each other)
        crossings = 0
        segs = [(path[i], path[i+1]) for i in range(len(path)-1)]
        for i in range(len(segs)):
            for j in range(i+2, len(segs)):
                if abs(i - j) <= 1: continue
                if _segments_cross(segs[i][0], segs[i][1], segs[j][0], segs[j][1]):
                    crossings += 1
        # turn angles (sharper = bigger angle off-straight)
        turn_sum = 0.0; turn_n = 0
        for i in range(1, len(path) - 1):
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            n1 = math.hypot(*v1); n2 = math.hypot(*v2)
            if n1 < 1e-9 or n2 < 1e-9: continue
            cos_t = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
            cos_t = max(-1.0, min(1.0, cos_t))
            turn_sum += math.acos(cos_t); turn_n += 1
        mean_turn = (turn_sum / turn_n) if turn_n else 0.0
        # TW slack
        slacks = []; tight = 0
        for c in route.customers:
            tw = cust_due[c] - cust_ready[c]
            slacks.append(tw)
            if tw / day_len < 0.10: tight += 1
        slack_mean = (sum(slacks) / len(slacks)) if slacks else 0.0
        # SPEC-METRICS-02 -- TW buffer simulation along the route.
        # Travel times via inst.travel_time matrix (depot=0, customer i = id i).
        # Arrival = max(prev_depart + travel, ready[c])
        # Wait    = max(0, ready[c] - (prev_depart + travel))
        # Buffer  = (due[c] - arrival) / (due[c] - ready[c])  in [0,1] when on-time
        # Bias toward arriving CLOSE to ready without waiting AND with margin.
        T = inst.travel_time
        early = late = ontime = 0
        wait_total = 0.0
        buffer_scores = []   # higher=better per stop
        prev_node = 0
        cur_t = float(inst.depot.ready)
        for cid in route.customers:
            travel = float(T[prev_node, cid])
            arr_uncon = cur_t + travel
            ready_c = float(cust_ready[cid])
            due_c = float(cust_due[cid])
            tw_w = max(1e-6, due_c - ready_c)
            if arr_uncon < ready_c:
                wait = ready_c - arr_uncon
                arr = ready_c
                wait_total += wait
                early += 1
            else:
                wait = 0.0
                arr = arr_uncon
                if arr <= due_c:
                    ontime += 1
                else:
                    late += 1
            # buffer in [0,1]: 1 = arrived right at ready (full TW remaining,
            # no wait); 0 = arrived right at due (no margin); negative -> late.
            margin = (due_c - arr) / tw_w
            wait_penalty = (wait / tw_w) if wait > 0 else 0.0
            buf = max(0.0, min(1.0, margin)) - 0.5 * wait_penalty
            buffer_scores.append(max(0.0, min(1.0, buf)))
            cur_t = arr + float(cust_service[cid])
            prev_node = cid
        n_stops = max(1, len(route.customers))
        tw_buffer = (sum(buffer_scores) / n_stops) if buffer_scores else 0.0
        early_frac = early / n_stops
        late_frac = late / n_stops
        ontime_frac = ontime / n_stops
        load = float(sum(cust_demand.get(c, 0) for c in route.customers))
        out.append(RouteScore(
            route_idx=ridx, n_customers=len(route.customers),
            load=load, util=min(1.0, load/cap),
            centroid_x=cx, centroid_y=cy,
            centroid_spread=spread, intra_dist_mean=intra_mean,
            detour_ratio=detour, aspect_ratio=aspect,
            self_crossings=crossings, mean_turn_angle_rad=mean_turn,
            tw_slack_min_mean=slack_mean,
            tight_tw_frac=(tight / len(route.customers)) if route.customers else 0.0,
            wait_total_min=wait_total,
            tw_buffer_score=tw_buffer,
            early_arrival_frac=early_frac,
            late_arrival_frac=late_frac,
            on_time_arrival_frac=ontime_frac,
        ))
    return out


def _cv(xs: list[float]) -> float:
    if not xs: return 0.0
    m = sum(xs) / len(xs)
    if m == 0: return 0.0
    var = sum((x - m)**2 for x in xs) / len(xs)
    return math.sqrt(var) / m


def _gini(xs: list[float]) -> float:
    if not xs: return 0.0
    s = sorted(xs); n = len(s); total = sum(s)
    if total == 0: return 0.0
    cum = 0.0
    for i, v in enumerate(s, start=1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n


def score_solution(inst: "Instance", sol: "Solution") -> SolutionQualityScore:
    """Aggregate scorecard for one solution."""
    rs = score_routes(inst, sol)
    served = sol.visited_customer_ids()
    n_unrouted = inst.num_customers - len(served)

    # cluster aggregates
    intra_means = [r.intra_dist_mean for r in rs if r.n_customers > 1]
    centroids = [(r.centroid_x, r.centroid_y) for r in rs]
    inter_means = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            inter_means.append(_euclid(centroids[i], centroids[j]))
    intra_inter = ((sum(intra_means)/len(intra_means))
                   / (sum(inter_means)/len(inter_means))) if (intra_means and inter_means) else 0.0

    # Hull overlaps
    cust_xy = {c.id: (c.x, c.y) for c in inst.customers}
    hulls = [_convex_hull_2d([cust_xy[c] for c in inst.customers
                                if r.route_idx < len(sol.routes)
                                and c in sol.routes[r.route_idx].customers])
             for r in rs]
    overlap_count = 0
    for i in range(len(hulls)):
        for j in range(i+1, len(hulls)):
            if _hulls_overlap(hulls[i], hulls[j]):
                overlap_count += 1

    # Silhouette-like: per route, (b - a) / max(a, b) where
    #   a = mean intra-distance to own centroid
    #   b = mean distance to nearest other centroid
    sil_scores = []
    for r in rs:
        a = r.centroid_spread
        if not centroids or len(centroids) < 2:
            continue
        others = [c for c in centroids if c != (r.centroid_x, r.centroid_y)]
        b = min(_euclid((r.centroid_x, r.centroid_y), o) for o in others) if others else a
        denom = max(a, b)
        sil_scores.append((b - a) / denom if denom > 0 else 0.0)
    sil = sum(sil_scores)/len(sil_scores) if sil_scores else 0.0

    # Inter-route segment crossings
    cross = 0
    all_paths = []
    depot_xy = (inst.depot.x, inst.depot.y)
    for route in sol.routes:
        if not route.customers: continue
        path = [depot_xy] + [cust_xy[c] for c in route.customers] + [depot_xy]
        segs = [(path[i], path[i+1]) for i in range(len(path)-1)]
        all_paths.append(segs)
    for ai in range(len(all_paths)):
        for bj in range(ai+1, len(all_paths)):
            for s1 in all_paths[ai]:
                for s2 in all_paths[bj]:
                    if _segments_cross(s1[0], s1[1], s2[0], s2[1]):
                        cross += 1


    utils = [r.util for r in rs]
    stops = [r.n_customers for r in rs]
    loads = [r.load for r in rs]
    detours = [r.detour_ratio for r in rs]
    slacks = [r.tw_slack_min_mean for r in rs]
    tights = [r.tight_tw_frac * r.n_customers for r in rs]
    n_total = sum(stops)
    # TW buffer aggregates (per-route weighted by stop count)
    tw_buffers = [r.tw_buffer_score * r.n_customers for r in rs]
    waits = [r.wait_total_min for r in rs]
    earlies = [r.early_arrival_frac * r.n_customers for r in rs]
    lates = [r.late_arrival_frac * r.n_customers for r in rs]
    ontimes = [r.on_time_arrival_frac * r.n_customers for r in rs]
    out = SolutionQualityScore(
        n_routes=len(rs),
        n_customers_served=len(served),
        n_unrouted=n_unrouted,
        routes=rs,
        intra_inter_ratio=intra_inter,
        silhouette_like=sil,
        convex_hull_overlap_count=overlap_count,
        load_util_cv=_cv(utils),
        stops_per_route_cv=_cv([float(s) for s in stops]),
        load_gini=_gini(loads),
        inter_route_crossings=cross,
        mean_detour_ratio=(sum(detours)/len(detours)) if detours else 0.0,
        mean_slack_min=(sum(slacks)/len(slacks)) if slacks else 0.0,
        tight_stops_frac=(sum(tights)/n_total) if n_total > 0 else 0.0,
        wait_to_service_ratio=0.0,  # placeholder
        mean_tw_buffer_score=(sum(tw_buffers)/n_total) if n_total > 0 else 0.0,
        total_wait_min=sum(waits),
        on_time_frac=(sum(ontimes)/n_total) if n_total > 0 else 0.0,
        early_frac=(sum(earlies)/n_total) if n_total > 0 else 0.0,
        late_frac=(sum(lates)/n_total) if n_total > 0 else 0.0,
    )
    # Synthesized quality_index in [0, 1] -- weighted combination.
    # Interpretation: each component is normalized so that lower-is-better
    # raw-metrics map to higher contributions.
    parts = []
    parts.append(("silhouette_like", max(0.0, min(1.0, (out.silhouette_like + 1.0) / 2.0)), 0.20))
    parts.append(("intra_inter",     max(0.0, 1.0 - min(1.0, out.intra_inter_ratio)), 0.15))
    parts.append(("hull_overlap",    1.0 / (1.0 + out.convex_hull_overlap_count / max(1, out.n_routes)), 0.10))
    parts.append(("load_balance",    max(0.0, 1.0 - min(1.0, out.load_util_cv)), 0.15))
    parts.append(("inter_cross",     1.0 / (1.0 + out.inter_route_crossings / max(1, out.n_routes**2)), 0.15))
    parts.append(("detour",          max(0.0, 1.0 - min(1.0, (out.mean_detour_ratio - 1.0) / 2.0)), 0.10))
    parts.append(("tight_tws",       max(0.0, 1.0 - out.tight_stops_frac), 0.03))
    parts.append(("served_frac",     out.n_customers_served / max(1, inst.num_customers), 0.07))
    # SPEC-METRICS-02 -- TW buffer aggregates feed quality_index.
    # Reward arriving close to ready without waiting; penalize lateness hard.
    parts.append(("tw_buffer",       out.mean_tw_buffer_score, 0.07))
    parts.append(("on_time",         out.on_time_frac, 0.05))
    parts.append(("not_late",        max(0.0, 1.0 - out.late_frac * 4.0), 0.03))
    out.quality_index = min(1.0, sum(s * w for _, s, w in parts))
    # iter-5u: K-fair complement. quality_index is K-dependent (rho=+0.78);
    # quality_per_route normalises so comparisons across solvers with very
    # different K are not biased by route count.
    out.quality_per_route = out.quality_index / max(1, out.n_routes)
    return out
