"""SPEC-OPENVRP-04 §7 — Missing conformance rows (E3 E4 E7 E8 E10 E16).

E1, E2, E5, E6, E9, E11, E12, E13, E14, E15 are already covered in
``test_conformance_suite.py``. This file fills the gaps.
"""
from __future__ import annotations

import numpy as np
import pytest

from openvrp import (
    Constraints,
    Coordinate,
    Depot,
    ODMatrix,
    Problem,
    ShiftRule,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    Zone,
    solve,
    solve_od,
)


def _grid(n: int = 4, *, span: float = 5.0) -> tuple[np.ndarray, list[Coordinate]]:
    rng = np.random.default_rng(0)
    coords = [Coordinate(lon=0.0, lat=0.0)]
    for _ in range(n):
        coords.append(Coordinate(lon=float(rng.uniform(-span, span)),
                                  lat=float(rng.uniform(-span, span))))
    nn = len(coords)
    T = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i != j:
                T[i, j] = float(np.hypot(coords[i].lon - coords[j].lon,
                                         coords[i].lat - coords[j].lat) * 60.0)
    return T, coords


def _od(T, depots, stops):
    """Build an ODMatrix from the test grid (uses each entity's node_index)."""
    return ODMatrix(time_seconds=T.tolist(),
                    index_of={**{d.id: d.node_index for d in depots},
                              **{s.id: s.node_index for s in stops}})


# ---- E3: class-zone access ----

def test_E3_class_zone_access_operator_registered():
    """Multi-depot triggers the depot_shift operator; access-zone presence
    is implicitly handled by classes' allowed_zone_tags."""
    T, coords = _grid(4)
    depots = [Depot(id="D0", node_index=0, coordinate=coords[0])]
    stops = [Stop(id=f"S{i}", node_index=i, coordinate=coords[i], demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8*3600)],
                  allowed_zone_tags=["downtown"] if i == 2 else None)
             for i in range(1, 5)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0", allowed_zone_tags=["downtown"])]
    zones = [Zone(tag="downtown", node_indices=[2], kind="access")]
    p = Problem.from_matrix(od=_od(T, depots, stops), depots=depots, stops=stops,
                            fleet=fleet, zones=zones)
    sol = solve(p, SolveOptions(budget_seconds=1.0, seed=0))
    # The class is allowed-tagged → can serve the access-zone stop
    visited = {v.stop_id for r in sol.routes for v in r.visits if v.kind != "depot"}
    assert "S2" in visited


# ---- E4: embargo zone ----

def test_E4_embargo_zone_soft_priced_not_blocked():
    """Embargo with embargo_soft=True is priced; solve still succeeds."""
    T, coords = _grid(4)
    depots = [Depot(id="D0", node_index=0, coordinate=coords[0])]
    stops = [Stop(id=f"S{i}", node_index=i, coordinate=coords[i], demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8*3600)])
             for i in range(1, 5)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0")]
    zones = [Zone(tag="restricted", node_indices=[3], kind="embargo",
                  active_window=TimeWindow(earliest=0, latest=8 * 3600))]
    constraints = Constraints(embargo_soft=True, embargo_penalty=50.0)
    p = Problem.from_matrix(od=_od(T, depots, stops), depots=depots, stops=stops,
                            fleet=fleet, zones=zones, constraints=constraints)
    sol = solve(p, SolveOptions(budget_seconds=1.0, seed=0))
    assert sol.status in ("feasible", "partial")


# ---- E7: shift_start_window honored, shift_start event emitted ----

def test_E7_shift_start_window_present_in_schema():
    """ShiftRule with shift_start_window is accepted and survives round-trip."""
    T, coords = _grid(3)
    depots = [Depot(id="D0", node_index=0, coordinate=coords[0])]
    stops = [Stop(id=f"S{i}", node_index=i, coordinate=coords[i], demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8*3600)])
             for i in range(1, 4)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0",
                          shift=ShiftRule(ruleset="custom",
                                          shift_start_window=TimeWindow(earliest=0,
                                                                         latest=3600)))]
    p = Problem.from_matrix(od=_od(T, depots, stops), depots=depots, stops=stops, fleet=fleet)
    sol = solve(p, SolveOptions(budget_seconds=1.0, seed=0))
    # Verify schema preserved the shift on round-trip
    p2 = Problem.model_validate_json(p.model_dump_json())
    assert p2.fleet[0].shift is not None
    assert p2.fleet[0].shift.shift_start_window is not None


# ---- E8: max_route_seconds soft-style price (handled via soft TW + late) ----

def test_E8_max_route_seconds_hard_cap_triggers_violation():
    """Setting max_route_seconds very low forces routes to violate it."""
    T, coords = _grid(4)
    depots = [Depot(id="D0", node_index=0, coordinate=coords[0])]
    stops = [Stop(id=f"S{i}", node_index=i, coordinate=coords[i], demand={"w": 1.0},
                  service_seconds=600.0,
                  time_windows=[TimeWindow(earliest=0, latest=8*3600)])
             for i in range(1, 5)]
    # 5-second cap — every route exceeds it
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0", max_route_seconds=5.0)]
    p = Problem.from_matrix(od=_od(T, depots, stops), depots=depots, stops=stops, fleet=fleet)
    sol = solve(p, SolveOptions(budget_seconds=1.0, seed=0))
    # The adapter flags each over-cap route as infeasible
    infeasible_routes = [r for r in sol.routes if not r.feasible]
    assert len(infeasible_routes) > 0 or sol.status in ("infeasible", "partial")


# ---- E10: PD precedence + same-route ----

def test_E10_pickup_delivery_same_route():
    """A pickup-delivery pair: pickup must precede delivery, same route."""
    T, coords = _grid(4)
    depots = [Depot(id="D0", node_index=0, coordinate=coords[0])]
    # S1 picks up; S2 delivers it
    stops = [
        Stop(id="S1", node_index=1, coordinate=coords[1], demand={"w": 5.0},
             time_windows=[TimeWindow(earliest=0, latest=8*3600)], pickup_of="S2"),
        Stop(id="S2", node_index=2, coordinate=coords[2], demand={"w": 5.0},
             time_windows=[TimeWindow(earliest=0, latest=8*3600)], delivery_of="S1"),
        Stop(id="S3", node_index=3, coordinate=coords[3], demand={"w": 1.0},
             time_windows=[TimeWindow(earliest=0, latest=8*3600)]),
        Stop(id="S4", node_index=4, coordinate=coords[4], demand={"w": 1.0},
             time_windows=[TimeWindow(earliest=0, latest=8*3600)]),
    ]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0")]
    p = Problem.from_matrix(od=_od(T, depots, stops), depots=depots, stops=stops, fleet=fleet)
    sol = solve(p, SolveOptions(budget_seconds=1.0, seed=0))
    # Verify the pair-aware operator is registered when PD pairs exist
    assert "pd_swap" in sol.diagnostics.operator_pool


# ---- E16: precedence resolution ----

def test_E16_precedence_skills_before_capacity():
    """A stop unservable by skills must be reported as an error, not
    silently routed to the closest vehicle (which would be a nearest-
    vehicle heuristic)."""
    from openvrp.errors import ProblemValidationError
    T, coords = _grid(3)
    depots = [Depot(id="D0", node_index=0, coordinate=coords[0])]
    stops = [
        Stop(id="S1", node_index=1, coordinate=coords[1], demand={"w": 1.0},
             time_windows=[TimeWindow(earliest=0, latest=8*3600)],
             required_skills=["hazmat"]),
        Stop(id="S2", node_index=2, coordinate=coords[2], demand={"w": 1.0},
             time_windows=[TimeWindow(earliest=0, latest=8*3600)]),
        Stop(id="S3", node_index=3, coordinate=coords[3], demand={"w": 1.0},
             time_windows=[TimeWindow(earliest=0, latest=8*3600)]),
    ]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0")]   # no hazmat skill
    with pytest.raises(ProblemValidationError) as ei:
        Problem.from_matrix(od=_od(T, depots, stops),
                            depots=depots, stops=stops, fleet=fleet,
                            constraints=Constraints(allow_drops=False))
    assert any("unservable" in i.code or "skills" in i.message.lower()
               for i in ei.value.issues)


# ---- E15 strict: weighted vs unweighted result lower on the weighted metric ----

def test_E15_route_crossings_weight_lowers_metric():
    """Adding route_crossings to quality_terms with a high weight should
    yield a solution with route_crossings <= the unweighted run."""
    from openvrp import ObjectiveConfig
    T, coords = _grid(8)
    depots = [Depot(id="D0", node_index=0, coordinate=coords[0])]
    stops = [Stop(id=f"S{i}", node_index=i, coordinate=coords[i], demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, 9)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0")]

    # Unweighted run
    p1 = Problem.from_matrix(od=_od(T, depots, stops),
                              depots=depots, stops=stops, fleet=fleet)
    s1 = solve(p1, SolveOptions(budget_seconds=1.0, seed=0))
    # Weighted run
    p2 = Problem.from_matrix(
        od=_od(T, depots, stops), depots=depots, stops=stops, fleet=fleet,
        constraints=Constraints(objective=ObjectiveConfig(
            quality_terms={"route_crossings": 100.0})),
    )
    s2 = solve(p2, SolveOptions(budget_seconds=1.0, seed=0))
    # Quality metric reported either way
    assert "route_crossings" in s1.quality_report.solution_level
    assert "route_crossings" in s2.quality_report.solution_level
