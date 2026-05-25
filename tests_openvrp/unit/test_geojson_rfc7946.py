"""SPEC-OPENVRP-05 F2, F4, F6 — GeoJSON RFC 7946 conformance."""
from __future__ import annotations

import json

import numpy as np
import pytest

from openvrp import (
    Coordinate,
    Depot,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    solve_od,
)


def _make_solution_with_points():
    n = 5
    rng = np.random.default_rng(0)
    coords = [(0.0, 0.0)] + [(float(rng.uniform(-10, 10)),
                              float(rng.uniform(-10, 10))) for _ in range(n)]
    nn = len(coords)
    T = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i != j:
                T[i, j] = float(np.hypot(coords[i][0] - coords[j][0],
                                         coords[i][1] - coords[j][1]) * 60.0)
    depots = [Depot(id="D0", node_index=0,
                    coordinate=Coordinate(lon=coords[0][0], lat=coords[0][1]))]
    stops = [Stop(id=f"S{i}", node_index=i,
                  coordinate=Coordinate(lon=coords[i][0], lat=coords[i][1]),
                  demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, n + 1)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0")]
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=1.0, seed=0))
    points = {s.id: Coordinate(lon=coords[i + 1][0], lat=coords[i + 1][1])
              for i, s in enumerate(stops)}
    points["D0"] = Coordinate(lon=coords[0][0], lat=coords[0][1])
    return sol, points


def test_geojson_structure_rfc7946():
    """Must be a FeatureCollection, no top-level 'crs' member (RFC 7946)."""
    sol, points = _make_solution_with_points()
    fc = sol.to_geojson(points=points)
    assert fc["type"] == "FeatureCollection"
    assert "crs" not in fc, "RFC 7946 forbids a top-level 'crs' member"
    assert isinstance(fc["features"], list)


def test_geojson_features_have_valid_geometry():
    """Every feature has a geometry.type in {Point, LineString} with
    well-formed coordinates (numeric pairs, lon in [-180,180], lat in [-90,90])."""
    sol, points = _make_solution_with_points()
    fc = sol.to_geojson(points=points)
    for f in fc["features"]:
        assert f["type"] == "Feature"
        assert "geometry" in f and "properties" in f
        g = f["geometry"]
        assert g["type"] in ("Point", "LineString")
        coords = g["coordinates"]
        if g["type"] == "Point":
            assert isinstance(coords, list) and len(coords) == 2
            lon, lat = coords
            assert -180.0 <= lon <= 180.0
            assert -90.0 <= lat <= 90.0
        else:
            assert isinstance(coords, list) and len(coords) >= 2
            for pt in coords:
                lon, lat = pt[0], pt[1]
                assert -180.0 <= lon <= 180.0
                assert -90.0 <= lat <= 90.0


def test_geojson_double_serialization_byte_identical():
    """F6: feature ordering deterministic; two serializations byte-identical."""
    sol, points = _make_solution_with_points()
    a = json.dumps(sol.to_geojson(points=points), sort_keys=True)
    b = json.dumps(sol.to_geojson(points=points), sort_keys=True)
    assert a == b


def test_geojson_event_inclusion_doesnt_alter_route_features():
    """F7: opting in events adds Points but doesn't change route/stop features."""
    sol, points = _make_solution_with_points()
    fc_off = sol.to_geojson(points=points)
    fc_on = sol.to_geojson(points=points, include={"events": True})
    routes_off = [f for f in fc_off["features"] if f["properties"].get("kind") == "route"]
    routes_on = [f for f in fc_on["features"] if f["properties"].get("kind") == "route"]
    assert len(routes_off) == len(routes_on)
    # Approximate-line flag preserved
    for r in routes_off:
        assert r["properties"].get("geometry_approx") is True


def test_geojson_jsonschema_validates_if_available():
    """If jsonschema is installed (dev extra), validate against a minimal
    GeoJSON FeatureCollection schema. Skipped otherwise."""
    try:
        import jsonschema
    except ImportError:
        pytest.skip("jsonschema not installed (it's in [dev] extras)")
    sol, points = _make_solution_with_points()
    fc = sol.to_geojson(points=points)
    # Minimal RFC 7946 sketch (full GeoJSON schema is verbose; this covers the basics)
    schema = {
        "type": "object",
        "required": ["type", "features"],
        "properties": {
            "type": {"const": "FeatureCollection"},
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "geometry", "properties"],
                    "properties": {
                        "type": {"const": "Feature"},
                        "geometry": {
                            "type": "object",
                            "required": ["type", "coordinates"],
                            "properties": {
                                "type": {"enum": ["Point", "LineString", "Polygon"]},
                            },
                        },
                    },
                },
            },
        },
    }
    jsonschema.validate(fc, schema)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
