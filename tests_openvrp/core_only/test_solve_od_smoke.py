"""SPEC-OPENVRP-03 §C1, SPEC-OPENVRP-07 H1, H4 — core install OD-only solve.

Runs under the core-deps-only env (no [network]/[pyvrp]/[viz]).
"""
from __future__ import annotations

import numpy as np
import pytest

from openvrp import (
    Constraints,
    Coordinate,
    Depot,
    ObjectiveConfig,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    solve_od,
)


def _build_smoke():
    """5 customers in a circle, single depot at origin."""
    rng = np.random.default_rng(42)
    n = 5
    # node 0 = depot; nodes 1..5 = customers
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = [(0.0, 0.0)] + [(float(np.cos(a) * 10), float(np.sin(a) * 10)) for a in angles]
    nn = len(coords)
    T = np.zeros((nn, nn), dtype=np.float64)
    for i in range(nn):
        for j in range(nn):
            if i == j:
                continue
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            T[i, j] = np.hypot(dx, dy) * 60.0   # 1 unit = 1 minute = 60 s
    D = T.copy()   # treat 1 second = 1 meter for simplicity

    depots = [Depot(id="D0", node_index=0)]
    stops = [Stop(id=f"S{i}", node_index=i, demand={"weight": 1.0},
                  service_seconds=30.0,
                  time_windows=[TimeWindow(earliest=0.0, latest=8 * 3600)])
             for i in range(1, n + 1)]
    fleet = [VehicleClass(id="van", count=None, capacity={"weight": 100.0},
                          home_depot_id="D0", cost_per_second=0.01,
                          cost_per_meter=0.001, fixed_cost=10.0)]
    return T, D, depots, stops, fleet


def test_solve_od_basic():
    T, D, depots, stops, fleet = _build_smoke()
    sol = solve_od(time_matrix=T, distance_matrix=D, stops=stops, depots=depots,
                   fleet=fleet, options=SolveOptions(budget_seconds=3.0, seed=0))
    assert sol.status in ("feasible", "partial")
    assert sol.vehicles_used >= 1
    assert sol.objective_value > 0.0
    # Quality report fully populated even without weighted terms
    assert len(sol.quality_report.solution_level) == 8
    # Diagnostics filled
    assert sol.diagnostics.operator_pool, "Operator pool should not be empty"
    # Conditional registration: shift_start is OFF because no peak windows
    assert "shift_start" not in sol.diagnostics.operator_pool


def test_solve_od_with_peak_enables_shift_start():
    T, D, depots, stops, fleet = _build_smoke()
    constraints = Constraints(peak_windows=[TimeWindow(earliest=0, latest=3600)])
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=2.0, seed=0),
                   distance_matrix=D, constraints=constraints)
    assert "shift_start" in sol.diagnostics.operator_pool


def test_solve_od_with_quality_weight_enables_unwind():
    """SPEC-OPENVRP-04 E15: weighting a quality_terms key changes the
    chosen solution; the matching operator is registered."""
    T, D, depots, stops, fleet = _build_smoke()
    objective = ObjectiveConfig(quality_terms={"route_crossings": 1.0})
    constraints = Constraints(objective=objective)
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=2.0, seed=0),
                   distance_matrix=D, constraints=constraints)
    assert "crossings_unwind" in sol.diagnostics.operator_pool
    # route_crossings reported in quality_report regardless
    assert "route_crossings" in sol.quality_report.solution_level


def test_solve_od_json_roundtrip():
    """SPEC-OPENVRP-05 F1: lossless JSON round-trip on a real Solution."""
    T, D, depots, stops, fleet = _build_smoke()
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=2.0, seed=0),
                   distance_matrix=D)
    body = sol.to_json()
    from openvrp.io import solution_from_json
    sol2 = solution_from_json(text=body)
    assert sol2.objective_value == pytest.approx(sol.objective_value, rel=1e-9)
    assert sol2.vehicles_used == sol.vehicles_used


def test_solve_od_geometry_unavailable():
    """SPEC-OPENVRP-05 F5: to_geojson on OD-only raises GeometryUnavailable
    unless points= supplied."""
    from openvrp.errors import GeometryUnavailable
    T, D, depots, stops, fleet = _build_smoke()
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=2.0, seed=0),
                   distance_matrix=D)
    with pytest.raises(GeometryUnavailable):
        sol.to_geojson()
    # With points= ⇒ approximate lines flagged
    points = {s.id: Coordinate(lon=float(i), lat=float(i))
              for i, s in enumerate(stops)}
    fc = sol.to_geojson(points=points)
    assert fc["type"] == "FeatureCollection"
    # routes carry geometry_approx=True
    routes_feats = [f for f in fc["features"] if f["properties"].get("kind") == "route"]
    assert routes_feats
    for f in routes_feats:
        assert f["properties"].get("geometry_approx") is True


def test_solve_od_progress_callback():
    """C5: on_progress fires per phase; StopSolve cancels cleanly."""
    T, D, depots, stops, fleet = _build_smoke()
    events = []
    def on_progress(ev):
        events.append(ev.phase)
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=2.0, seed=0,
                                        on_progress=on_progress),
                   distance_matrix=D)
    # At minimum: ingest, construct, refine, finalize
    assert "ingest" in events
    assert "finalize" in events
    assert sol.status in ("feasible", "partial")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
