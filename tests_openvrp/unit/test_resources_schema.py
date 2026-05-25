"""SPEC-OPENVRP-01/02 — Generalized resource replenishment schema.

Multi-depot, multi-resource: depots stock resources; vehicle classes
consume resources per meter; ReplenishEvent records mid-route refills.

These tests validate the CONTRACT (schema round-trip + validators +
conditional operator registration). The native solver does not yet
construct replenishment-aware routes — that is a tracked completion
item against the locked schema.
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
    ReplenishEvent,
    Solution,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    solve_od,
)
from openvrp.errors import ProblemValidationError


def _grid(n: int = 4) -> tuple:
    rng = np.random.default_rng(0)
    coords = [(0.0, 0.0)]
    for _ in range(n):
        coords.append((float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10))))
    nn = len(coords)
    T = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i != j:
                T[i, j] = float(np.hypot(coords[i][0] - coords[j][0],
                                         coords[i][1] - coords[j][1]) * 60.0)
    return T, coords


# ============================================================
# Depot.resources field
# ============================================================


def test_depot_resources_default_empty_back_compat():
    """Default empty dict — preserves pre-resource behavior."""
    d = Depot(id="D0", node_index=0)
    assert d.resources == {}


def test_depot_resources_round_trip():
    """Per-resource stock survives JSON round-trip including None=unlimited."""
    d = Depot(id="D0", node_index=0,
              resources={"fuel_liters": 500.0, "spare_parts": None,
                         "water_liters": 1200.0})
    d2 = Depot.model_validate_json(d.model_dump_json())
    assert d == d2
    assert d2.resources["fuel_liters"] == 500.0
    assert d2.resources["spare_parts"] is None
    assert d2.resources["water_liters"] == 1200.0


# ============================================================
# VehicleClass.consumes + onboard_capacity
# ============================================================


def test_vehicle_class_consumes_round_trip():
    c = VehicleClass(id="van", count=None, capacity={"w": 100.0},
                     home_depot_id="D0",
                     consumes={"fuel_liters": 0.08},
                     onboard_capacity={"fuel_liters": 60.0})
    c2 = VehicleClass.model_validate_json(c.model_dump_json())
    assert c == c2
    assert c2.consumes["fuel_liters"] == 0.08
    assert c2.onboard_capacity["fuel_liters"] == 60.0


def test_vehicle_class_consumes_default_empty():
    c = VehicleClass(id="van", count=None, capacity={"w": 50.0},
                     home_depot_id="D0")
    assert c.consumes == {}
    assert c.onboard_capacity == {}


# ============================================================
# Validator: consumes-without-stocked warns
# ============================================================


def test_validator_warns_when_class_consumes_unstocked_resource():
    """A class that consumes 'fuel_liters' but no depot stocks it → warning."""
    T, _ = _grid(3)
    depots = [Depot(id="D0", node_index=0)]   # no resources stocked
    stops = [Stop(id=f"S{i}", node_index=i, demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, 4)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0",
                          consumes={"fuel_liters": 0.08})]
    od = ODMatrix(time_seconds=T.tolist(),
                  index_of={d.id: d.node_index for d in depots}
                  | {s.id: s.node_index for s in stops})
    p = Problem.from_matrix(od=od, depots=depots, stops=stops, fleet=fleet)
    warnings = p.input_warnings()
    assert any(w.code == "class.resource_unstocked" for w in warnings), \
        f"expected resource_unstocked warning; got {[w.code for w in warnings]}"


def test_validator_errors_when_onboard_capacity_zero_and_consumes():
    """Class consumes a resource but onboard_capacity for it = 0 → ERROR."""
    T, _ = _grid(3)
    depots = [Depot(id="D0", node_index=0, resources={"fuel_liters": 1000.0})]
    stops = [Stop(id=f"S{i}", node_index=i, demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, 4)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0",
                          consumes={"fuel_liters": 0.08},
                          onboard_capacity={"fuel_liters": 0.0})]
    od = ODMatrix(time_seconds=T.tolist(),
                  index_of={d.id: d.node_index for d in depots}
                  | {s.id: s.node_index for s in stops})
    with pytest.raises(ProblemValidationError) as ei:
        Problem.from_matrix(od=od, depots=depots, stops=stops, fleet=fleet)
    assert any(i.code == "class.onboard_zero" for i in ei.value.issues)


def test_validator_silent_when_resources_consistent():
    """Class consumes 'fuel' and a depot stocks 'fuel' → no warnings."""
    T, _ = _grid(3)
    depots = [Depot(id="D0", node_index=0,
                    resources={"fuel_liters": 1000.0})]
    stops = [Stop(id=f"S{i}", node_index=i, demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, 4)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0",
                          consumes={"fuel_liters": 0.08},
                          onboard_capacity={"fuel_liters": 60.0})]
    od = ODMatrix(time_seconds=T.tolist(),
                  index_of={d.id: d.node_index for d in depots}
                  | {s.id: s.node_index for s in stops})
    p = Problem.from_matrix(od=od, depots=depots, stops=stops, fleet=fleet)
    resource_warns = [w for w in p.input_warnings()
                      if "resource" in w.code or "onboard" in w.code]
    assert resource_warns == []


# ============================================================
# Conditional operator pool — replenish_insert registered
# ============================================================


def test_operator_pool_excludes_replenish_insert_when_no_resources():
    """Default problem with no resources → replenish_insert NOT registered."""
    T, _ = _grid(3)
    depots = [Depot(id="D0", node_index=0)]
    stops = [Stop(id=f"S{i}", node_index=i, demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, 4)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0")]
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=1.0, seed=0,
                                        construction="fast"))
    assert "replenish_insert" not in sol.diagnostics.operator_pool


def test_operator_pool_includes_replenish_insert_when_class_consumes():
    """Class declares consumption → replenish_insert IS registered."""
    T, _ = _grid(3)
    depots = [Depot(id="D0", node_index=0,
                    resources={"fuel_liters": 500.0})]
    stops = [Stop(id=f"S{i}", node_index=i, demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, 4)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0",
                          consumes={"fuel_liters": 0.08},
                          onboard_capacity={"fuel_liters": 60.0})]
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=1.0, seed=0,
                                        construction="fast"))
    assert "replenish_insert" in sol.diagnostics.operator_pool


def test_operator_pool_includes_replenish_insert_when_depot_stocks_resources():
    """Depot stocks resources (no class consumption) → still registered.

    A depot offering replenishment is a 'resource is in play' signal even
    if no current class declares consumption — future fleet changes might.
    """
    T, _ = _grid(3)
    depots = [Depot(id="D0", node_index=0,
                    resources={"spare_parts": None})]
    stops = [Stop(id=f"S{i}", node_index=i, demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, 4)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0")]
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=1.0, seed=0,
                                        construction="fast"))
    assert "replenish_insert" in sol.diagnostics.operator_pool


# ============================================================
# ReplenishEvent — discriminated-union round-trip
# ============================================================


def test_replenish_event_round_trip():
    """ReplenishEvent survives JSON round-trip via the discriminated union."""
    from openvrp.schema.output import Route, RouteEconomics
    e = ReplenishEvent(at_seconds=3600.0, after_visit_index=2,
                       at_depot_id="D1",
                       duration_seconds=600.0,
                       resources_replenished={"fuel_liters": 45.0,
                                              "water_liters": 12.0})
    r = Route(vehicle_class_id="van", vehicle_ordinal=0, home_depot_id="D0",
              economics=RouteEconomics(total=0.0), events=[e])
    r2 = Route.model_validate_json(r.model_dump_json())
    assert isinstance(r2.events[0], ReplenishEvent)
    assert r2.events[0].at_depot_id == "D1"
    assert r2.events[0].resources_replenished["fuel_liters"] == 45.0
    assert r2.events[0].resources_replenished["water_liters"] == 12.0


def test_replenish_event_distinct_from_legacy_recharge():
    """ReplenishEvent and RechargeEvent are distinct discriminated-union arms.

    Legacy RechargeEvent stays around for back-compat; new code should
    prefer ReplenishEvent.
    """
    from openvrp.schema.output import RechargeEvent, Route, RouteEconomics
    legacy = RechargeEvent(at_seconds=100.0, at_depot_id="D0",
                            duration_seconds=300.0,
                            energy_meters_restored=120_000.0)
    modern = ReplenishEvent(at_seconds=200.0, at_depot_id="D0",
                             resources_replenished={"fuel_liters": 50.0})
    r = Route(vehicle_class_id="van", vehicle_ordinal=0, home_depot_id="D0",
              economics=RouteEconomics(total=0.0),
              events=[legacy, modern])
    r2 = Route.model_validate_json(r.model_dump_json())
    assert isinstance(r2.events[0], RechargeEvent)
    assert isinstance(r2.events[1], ReplenishEvent)


# ============================================================
# Multi-depot multi-resource Problem round-trip (full coverage)
# ============================================================


def test_multi_depot_multi_resource_problem_round_trip():
    """A Problem with 2 depots × 2 resource types × 1 consuming class
    round-trips losslessly through JSON."""
    T, _ = _grid(4)
    depots = [
        Depot(id="hub_north", node_index=0, coordinate=Coordinate(lon=0, lat=0),
              resources={"fuel_liters": 1000.0, "spare_parts": None}),
        Depot(id="hub_south", node_index=0, coordinate=Coordinate(lon=0, lat=0),
              resources={"fuel_liters": 600.0}),   # no spare_parts here
    ]
    stops = [Stop(id=f"S{i}", node_index=i,
                  coordinate=Coordinate(lon=1, lat=1), demand={"w": 5.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, 5)]
    fleet = [VehicleClass(
        id="service_truck", count=None, capacity={"w": 100.0},
        home_depot_id="hub_north",
        consumes={"fuel_liters": 0.08, "spare_parts": 0.001},
        onboard_capacity={"fuel_liters": 60.0, "spare_parts": 20.0},
    )]
    od = ODMatrix(time_seconds=T.tolist(),
                  index_of={"hub_north": 0, "hub_south": 1,
                            **{s.id: s.node_index for s in stops}})
    # NOTE: two depots share node_index=0 in this synthetic test for
    # ODMatrix simplicity; the validator allows it because index_of
    # uniqueness is enforced by string key.
    depots[1] = depots[1].model_copy(update={"node_index": 1})
    p = Problem.from_matrix(od=od, depots=depots, stops=stops, fleet=fleet)
    body = p.model_dump_json()
    p2 = Problem.model_validate_json(body)
    assert p2.depots[0].resources["fuel_liters"] == 1000.0
    assert p2.depots[0].resources["spare_parts"] is None
    assert p2.depots[1].resources == {"fuel_liters": 600.0}
    assert p2.fleet[0].consumes["fuel_liters"] == 0.08
    assert p2.fleet[0].onboard_capacity["spare_parts"] == 20.0
