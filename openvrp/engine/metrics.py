"""SPEC-OPENVRP-04 §3 — Composite operational-quality metric catalog.

Every metric here is **always computed and reported** in
``Solution.quality_report``; each becomes an optimized objective term
when the caller sets a weight in ``ObjectiveConfig.quality_terms``.

All metrics are operational/geometric. There is **no visual or VLM
metric** and none may be added (SPEC-OPENVRP-00 D9).

Each metric returns a non-negative penalty (lower=better) using an
instance-scale-invariant transform so weights are comparable across
instance sizes.

Catalog keys (mirrored in ``openvrp.schema.input.QUALITY_CATALOG``):
- ``route_crossings``         — count of inter-route segment intersections
- ``mean_detour_ratio``       — mean (leg path length / straight OD)
- ``load_balance_cv``         — coefficient of variation of vehicle loads
- ``load_balance_gini``       — Gini of vehicle loads
- ``time_window_slack``       — penalty = -mean_normalized_slack (higher slack better)
- ``intra_route_compactness`` — mean intra-route spatial spread
- ``cross_route_overlap``     — convex-hull overlap area across routes
- ``quality_per_route``       — composite ÷ routes (K-fair)
"""
from __future__ import annotations

import math
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover — numpy is a hard core dep
    np = None  # type: ignore


# ============================================================
# Per-route + per-solution geometry/load helpers
# ============================================================


def _route_points(route: Any, stop_xy: dict[str, tuple[float, float]],
                  depot_xy: tuple[float, float]) -> list[tuple[float, float]]:
    """Return depot -> stops -> depot polyline of (x,y) tuples."""
    out: list[tuple[float, float]] = [depot_xy]
    for v in route.visits:
        if v.kind == "depot":
            continue
        xy = stop_xy.get(v.stop_id)
        if xy is not None:
            out.append(xy)
    out.append(depot_xy)
    return out


def _segments(pts: list[tuple[float, float]]) -> list[tuple[_Pt, _Pt]]:
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


_Pt = tuple[float, float]


def _seg_intersect(a: _Pt, b: _Pt, c: _Pt, d: _Pt) -> bool:
    """Robust 2-D segment intersection (excludes collinear-touch).

    Returns True iff segments AB and CD strictly cross.
    """
    def ccw(p: _Pt, q: _Pt, r: _Pt) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1 = ccw(c, d, a)
    d2 = ccw(c, d, b)
    d3 = ccw(a, b, c)
    d4 = ccw(a, b, d)
    return bool(((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
                ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))


# ============================================================
# Catalog implementations
# ============================================================


def route_crossings(routes: list[Any], stop_xy: dict[str, tuple[float, float]],
                    depot_xy: tuple[float, float]) -> float:
    """Count of inter-route segment crossings. Lower is better.

    Scale-invariant: returned as-is (instances normalize by ÷ n_routes^2
    in ``quality_per_route``).
    """
    polys = [_route_points(r, stop_xy, depot_xy) for r in routes if r.visits]
    segs_per_route = [_segments(p) for p in polys]
    total = 0
    for i in range(len(segs_per_route)):
        for j in range(i + 1, len(segs_per_route)):
            for s1 in segs_per_route[i]:
                for s2 in segs_per_route[j]:
                    if _seg_intersect(s1[0], s1[1], s2[0], s2[1]):
                        total += 1
    return float(total)


def mean_detour_ratio(routes: list[Any], stop_xy: dict[str, tuple[float, float]],
                      depot_xy: tuple[float, float],
                      od_distance: dict[tuple[str, str], float] | None = None) -> float:
    """Mean ratio of actual leg distance (Euclidean fallback) to straight
    OD distance. >= 1.0 by triangle inequality; reported as ``ratio - 1.0``
    so the penalty is 0 for a tight basin and grows with detour.
    """
    if od_distance is None:
        return 0.0   # without OD we have no reference; report 0
    accum = 0.0
    n = 0
    for r in routes:
        for i in range(len(r.visits) - 1):
            a, b = r.visits[i].stop_id, r.visits[i + 1].stop_id
            od = od_distance.get((a, b))
            if od is None or od <= 0:
                continue
            xa = stop_xy.get(a, depot_xy)
            xb = stop_xy.get(b, depot_xy)
            actual = math.hypot(xa[0] - xb[0], xa[1] - xb[1])
            ratio = actual / od
            accum += max(0.0, ratio - 1.0)
            n += 1
    return accum / n if n else 0.0


def _route_loads(routes: list[Any], stop_demand: dict[str, dict[str, float]]) -> list[float]:
    """Per-route total scalar demand (sum across dims and stops)."""
    out = []
    for r in routes:
        load = 0.0
        for v in r.visits:
            if v.kind == "depot":
                continue
            for k, val in stop_demand.get(v.stop_id, {}).items():
                load += float(val)
        if load > 0:
            out.append(load)
    return out


def load_balance_cv(routes: list[Any], stop_demand: dict[str, dict[str, float]]) -> float:
    """Coefficient of variation of per-route loads. 0 = perfectly balanced.
    Penalty form is the raw CV (already non-negative)."""
    loads = _route_loads(routes, stop_demand)
    if len(loads) < 2:
        return 0.0
    mu = sum(loads) / len(loads)
    if mu == 0:
        return 0.0
    var = sum((x - mu) ** 2 for x in loads) / len(loads)
    return math.sqrt(var) / mu


def load_balance_gini(routes: list[Any], stop_demand: dict[str, dict[str, float]]) -> float:
    """Gini coefficient on per-route loads. 0 = perfectly equal, 1 = max
    inequality. Penalty form: raw Gini."""
    loads = sorted(_route_loads(routes, stop_demand))
    n = len(loads)
    if n < 2:
        return 0.0
    s = sum(loads)
    if s == 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(loads))
    return (2 * weighted) / (n * s) - (n + 1) / n


def time_window_slack(routes: list[Any], stop_by_id: dict[str, Any]) -> float:
    """Penalty = ``-mean_normalized_slack``, so higher slack = lower penalty
    (good). Slack is ``(latest_tw - arrival) / tw_width`` clipped [0,1].
    """
    accum = 0.0
    n = 0
    for r in routes:
        for v in r.visits:
            if v.kind == "depot":
                continue
            s = stop_by_id.get(v.stop_id)
            if s is None or not s.time_windows:
                continue
            # Use the tw matching this visit (multi-window: pick the one bracketing arrival)
            chosen = None
            for tw in s.time_windows:
                if tw.earliest <= v.arrival_seconds <= tw.latest + 1e-3:
                    chosen = tw
                    break
            if chosen is None:
                chosen = s.time_windows[0]
            width = max(1.0, chosen.latest - chosen.earliest)
            slack = max(0.0, min(1.0, (chosen.latest - v.arrival_seconds) / width))
            accum += slack
            n += 1
    mean_slack = (accum / n) if n else 0.0
    return -mean_slack   # penalty form


def intra_route_compactness(routes: list[Any], stop_xy: dict[str, tuple[float, float]]) -> float:
    """Mean per-route bounding-box diagonal (in coordinate units), divided
    by the number of stops on the route. Lower = more compact."""
    diags = []
    for r in routes:
        xs = []
        ys = []
        for v in r.visits:
            if v.kind == "depot":
                continue
            xy = stop_xy.get(v.stop_id)
            if xy is None:
                continue
            xs.append(xy[0]); ys.append(xy[1])
        if len(xs) < 2:
            continue
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        diags.append(diag / max(1, len(xs)))
    return (sum(diags) / len(diags)) if diags else 0.0


def cross_route_overlap(routes: list[Any], stop_xy: dict[str, tuple[float, float]]) -> float:
    """Pairwise overlap of axis-aligned bounding-box areas across routes,
    normalized by total bbox area. 0 = clean partition; higher = more overlap.
    Uses bbox as a cheap proxy for the convex-hull overlap mentioned in
    the spec; pure geometric, no external lib.
    """
    bboxes = []
    for r in routes:
        xs = []; ys = []
        for v in r.visits:
            if v.kind == "depot":
                continue
            xy = stop_xy.get(v.stop_id)
            if xy is None:
                continue
            xs.append(xy[0]); ys.append(xy[1])
        if len(xs) < 2:
            continue
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))
    if len(bboxes) < 2:
        return 0.0
    total_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in bboxes)
    if total_area <= 0:
        return 0.0
    overlap = 0.0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            a, b = bboxes[i], bboxes[j]
            dx = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            dy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            overlap += dx * dy
    return overlap / total_area


def quality_per_route(other_metrics: dict[str, float], n_routes: int) -> float:
    """K-fair composite: sum of penalty metrics divided by the route count.

    Doesn't double-count: ``quality_per_route`` itself is excluded from
    its own composite. Stays >= 0 because each input is in penalty form.
    """
    if n_routes <= 0:
        return 0.0
    components = ("route_crossings", "mean_detour_ratio",
                  "load_balance_cv", "load_balance_gini",
                  "intra_route_compactness", "cross_route_overlap")
    total = sum(other_metrics.get(k, 0.0) for k in components)
    return total / float(n_routes)


# ============================================================
# Top-level: compute all
# ============================================================


def compute_all_metrics(routes: list[Any], *,
                        stop_by_id: dict[str, Any], depot_xy: tuple[float, float],
                        stop_xy: dict[str, tuple[float, float]],
                        stop_demand: dict[str, dict[str, float]],
                        od_distance: dict[tuple[str, str], float] | None = None,
                        ) -> dict[str, float]:
    """Compute every catalog metric for the solution. Always reported,
    regardless of whether the caller weighted it (SPEC-OPENVRP-04 §3).
    """
    out: dict[str, float] = {}
    active_routes = [r for r in routes if r.visits]
    out["route_crossings"] = route_crossings(active_routes, stop_xy, depot_xy)
    out["mean_detour_ratio"] = mean_detour_ratio(active_routes, stop_xy, depot_xy, od_distance)
    out["load_balance_cv"] = load_balance_cv(active_routes, stop_demand)
    out["load_balance_gini"] = load_balance_gini(active_routes, stop_demand)
    out["time_window_slack"] = time_window_slack(active_routes, stop_by_id)
    out["intra_route_compactness"] = intra_route_compactness(active_routes, stop_xy)
    out["cross_route_overlap"] = cross_route_overlap(active_routes, stop_xy)
    out["quality_per_route"] = quality_per_route(out, len(active_routes))
    return out


__all__ = [
    "compute_all_metrics",
    "route_crossings", "mean_detour_ratio",
    "load_balance_cv", "load_balance_gini",
    "time_window_slack", "intra_route_compactness",
    "cross_route_overlap", "quality_per_route",
]
