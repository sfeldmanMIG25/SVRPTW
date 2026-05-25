"""SPEC-OPENVRP-05 — Serialization.

* JSON: lossless pydantic round-trip; ODMatrix nested-list (re)serialize.
* GeoJSON: RFC 7946 ``FeatureCollection`` for network-mode solutions.

A Euclidean line is **never** emitted as if it were a real network path
(SPEC-OPENVRP-00 D14). On OD-only input, ``solution_to_geojson`` raises
``GeometryUnavailable`` unless the caller supplies ``points=`` for an
explicitly flagged approximate fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openvrp.errors import GeometryUnavailable
from openvrp.schema.input import Coordinate, Problem
from openvrp.schema.output import Solution

# ============================================================
# JSON
# ============================================================


def problem_to_json(problem: Problem, path: str | None = None) -> str:
    body = problem.model_dump_json()
    if path is not None:
        Path(path).write_text(body, encoding="utf-8")
    return body


def problem_from_json(path: str | None = None, *, text: str | None = None) -> Problem:
    if (path is None) == (text is None):
        raise ValueError("Provide exactly one of path or text.")
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
    return Problem.model_validate_json(text or "")


def solution_to_json(sol: Solution, path: str | None = None) -> str:
    body = sol.model_dump_json()
    if path is not None:
        Path(path).write_text(body, encoding="utf-8")
    return body


def solution_from_json(path: str | None = None, *, text: str | None = None) -> Solution:
    if (path is None) == (text is None):
        raise ValueError("Provide exactly one of path or text.")
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
    return Solution.model_validate_json(text or "")


# ============================================================
# GeoJSON (RFC 7946)
# ============================================================


def solution_to_geojson(sol: Solution, *,
                        path: str | None = None,
                        include: dict[str, bool] | None = None,
                        points: dict[str, Coordinate] | None = None) -> dict[str, Any]:
    """Emit RFC 7946 FeatureCollection.

    Behavior:
    - network-mode solution (``geometry_status='present'``): one ``LineString``
      per Route from concatenated leg polylines; one ``Point`` per customer
      Visit; optional ``Point``s for break/zone/recharge events.
    - OD-only solution: raises ``GeometryUnavailable`` unless ``points=``
      provided, in which case approximate ``LineString``s are emitted with
      ``properties.geometry_approx=true``.
    """
    include = include or {}

    if sol.geometry_status in ("absent_od_mode", "absent_failed"):
        if points is None:
            raise GeometryUnavailable(
                "OD-only solution has no along-network geometry. "
                "Either solve a network-mode Problem, or pass "
                "points={stop_id: Coordinate} to emit explicitly-flagged "
                "approximate lines."
            )
        return _approx_geojson(sol, points=points, include=include, path=path)

    features: list[dict[str, Any]] = []
    # Stable order: routes by (class_id, ordinal)
    routes_sorted = sorted(sol.routes, key=lambda r: (r.vehicle_class_id, r.vehicle_ordinal))
    for r in routes_sorted:
        # Concatenate leg polylines
        coords: list[list[float]] = []
        for leg in r.geometry:
            for c in leg.polyline:
                if not coords or coords[-1] != [c.lon, c.lat]:
                    coords.append([c.lon, c.lat])
        if not coords:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "kind": "route",
                "vehicle_class_id": r.vehicle_class_id,
                "vehicle_ordinal": r.vehicle_ordinal,
                "home_depot_id": r.home_depot_id,
                "distance_meters": r.distance_meters,
                "duration_seconds": r.duration_seconds,
                "feasible": r.feasible,
                "violations": list(r.violations),
                "quality": dict(r.quality),
                "economics": r.economics.model_dump(),
            },
        })
        # Per-customer point features (visits)
        for v in sorted(r.visits, key=lambda v: v.sequence_index):
            if v.kind == "depot":
                continue
            # First leg endpoint as best fallback for the stop's coordinate
            point_coord = None
            for leg in r.geometry:
                if leg.from_stop_id == v.stop_id and leg.polyline:
                    point_coord = leg.polyline[0]
                    break
                if leg.to_stop_id == v.stop_id and leg.polyline:
                    point_coord = leg.polyline[-1]
                    break
            if point_coord is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [point_coord.lon, point_coord.lat]},
                "properties": {
                    "kind": v.kind,
                    "stop_id": v.stop_id,
                    "arrival_seconds": v.arrival_seconds,
                    "lateness_seconds": v.lateness_seconds,
                    "load_after": dict(v.load_after),
                    "sequence_index": v.sequence_index,
                    "vehicle_class_id": r.vehicle_class_id,
                    "vehicle_ordinal": r.vehicle_ordinal,
                },
            })
        # Event features (opt-in)
        if include.get("events"):
            for e in r.events:
                k = e.kind
                if k not in ("break", "zone_enter", "zone_exit", "recharge"):
                    continue
                # Anchor event to the after_visit_index's visit coord (best-effort)
                anchor_coord = None
                if e.after_visit_index is not None and r.geometry:
                    # use the endpoint of the matching leg
                    idx = max(0, min(e.after_visit_index, len(r.geometry) - 1))
                    if r.geometry[idx].polyline:
                        anchor_coord = r.geometry[idx].polyline[-1]
                if anchor_coord is None and r.geometry and r.geometry[0].polyline:
                    anchor_coord = r.geometry[0].polyline[0]
                if anchor_coord is None:
                    continue
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [anchor_coord.lon, anchor_coord.lat]},
                    "properties": {
                        "kind": "event",
                        "event_kind": k,
                        **e.model_dump(exclude={"kind"}),
                    },
                })
    # Dropped stop features
    for d in sol.dropped:
        # We don't have coords on the public Solution; the caller has them.
        # When points is provided, attach.
        if points and d.stop_id in points:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [points[d.stop_id].lon, points[d.stop_id].lat]},
                "properties": {
                    "kind": "dropped",
                    "stop_id": d.stop_id,
                    "reason": d.reason,
                    "priority": d.priority,
                    "dropped": True,
                },
            })

    fc = {
        "type": "FeatureCollection",
        "features": features,
    }
    if path is not None:
        Path(path).write_text(json.dumps(fc), encoding="utf-8")
    return fc


def _approx_geojson(sol: Solution, *, points: dict[str, Coordinate],
                    include: dict[str, bool], path: str | None) -> dict[str, Any]:
    """OD-mode fallback: emit straight LineStrings flagged geometry_approx."""
    features: list[dict[str, Any]] = []
    routes_sorted = sorted(sol.routes, key=lambda r: (r.vehicle_class_id, r.vehicle_ordinal))
    for r in routes_sorted:
        coords: list[list[float]] = []
        for v in sorted(r.visits, key=lambda v: v.sequence_index):
            pt = points.get(v.stop_id)
            if pt is None:
                continue
            coords.append([pt.lon, pt.lat])
        if len(coords) < 2:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "kind": "route",
                "vehicle_class_id": r.vehicle_class_id,
                "vehicle_ordinal": r.vehicle_ordinal,
                "home_depot_id": r.home_depot_id,
                "distance_meters": r.distance_meters,
                "duration_seconds": r.duration_seconds,
                "feasible": r.feasible,
                "violations": list(r.violations),
                "quality": dict(r.quality),
                "economics": r.economics.model_dump(),
                "geometry_approx": True,
            },
        })
        for v in sorted(r.visits, key=lambda v: v.sequence_index):
            if v.kind == "depot":
                continue
            pt = points.get(v.stop_id)
            if pt is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [pt.lon, pt.lat]},
                "properties": {
                    "kind": v.kind,
                    "stop_id": v.stop_id,
                    "arrival_seconds": v.arrival_seconds,
                    "lateness_seconds": v.lateness_seconds,
                    "load_after": dict(v.load_after),
                    "sequence_index": v.sequence_index,
                    "vehicle_class_id": r.vehicle_class_id,
                    "vehicle_ordinal": r.vehicle_ordinal,
                },
            })
    for d in sol.dropped:
        pt = points.get(d.stop_id)
        if pt is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [pt.lon, pt.lat]},
            "properties": {
                "kind": "dropped", "stop_id": d.stop_id, "reason": d.reason,
                "priority": d.priority, "dropped": True,
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    if path is not None:
        Path(path).write_text(json.dumps(fc), encoding="utf-8")
    return fc


__all__ = [
    "problem_to_json", "problem_from_json",
    "solution_to_json", "solution_from_json",
    "solution_to_geojson",
]
