"""SPEC-OPENVRP-01 A10, SPEC-OPENVRP-05 F1 — full-coverage round-trip.

A fixture exercising every optional public field: PD pairs, multi-depot,
eu561 shift, zones, populated quality_terms, soft TW, peak windows,
priority, embargo soft + penalty.

Verifies T == T.model_validate_json(T.model_dump_json()) for both Problem
and Solution.
"""
from __future__ import annotations

import numpy as np

from openvrp import (
    Constraints,
    Coordinate,
    Depot,
    Network,
    ODMatrix,
    ObjectiveConfig,
    Problem,
    ShiftRule,
    SnapConfig,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    Zone,
    solve,
)


def _build_full_problem() -> Problem:
    n = 6
    rng = np.random.default_rng(0)
    coords = [(0.0, 0.0), (0.1, 0.0)]  # 2 depots
    coords.extend([(float(rng.uniform(-10, 10)),
                    float(rng.uniform(-10, 10))) for _ in range(n)])
    nn = len(coords)
    T = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i != j:
                T[i, j] = float(np.hypot(coords[i][0] - coords[j][0],
                                         coords[i][1] - coords[j][1]) * 600.0)
    od = ODMatrix(time_seconds=T.tolist(),
                  distance_meters=T.tolist(),
                  index_of={"D0": 0, "D1": 1,
                            **{f"S{i + 1}": 2 + i for i in range(n)}})
    depots = [
        Depot(id="D0", node_index=0, coordinate=Coordinate(lon=0, lat=0),
              time_window=TimeWindow(earliest=0, latest=24 * 3600)),
        Depot(id="D1", node_index=1, coordinate=Coordinate(lon=0.1, lat=0),
              time_window=TimeWindow(earliest=0, latest=24 * 3600)),
    ]
    stops = [
        Stop(id="S1", node_index=2, coordinate=Coordinate(lon=coords[2][0], lat=coords[2][1]),
             demand={"weight": 5.0}, service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0, latest=8 * 3600, hard=False)],
             required_skills=["std"], priority=2),
        Stop(id="S2", node_index=3, coordinate=Coordinate(lon=coords[3][0], lat=coords[3][1]),
             demand={"weight": 3.0}, service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0, latest=8 * 3600)],
             required_skills=["std"], pickup_of="S3"),
        Stop(id="S3", node_index=4, coordinate=Coordinate(lon=coords[4][0], lat=coords[4][1]),
             demand={"weight": 3.0}, service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0, latest=8 * 3600)],
             required_skills=["std"], delivery_of="S2"),
        Stop(id="S4", node_index=5, coordinate=Coordinate(lon=coords[5][0], lat=coords[5][1]),
             demand={"weight": 2.0}, service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0, latest=8 * 3600)],
             required_skills=["std"], allowed_zone_tags=["downtown"]),
        Stop(id="S5", node_index=6, coordinate=Coordinate(lon=coords[6][0], lat=coords[6][1]),
             demand={"weight": 1.0}, service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0, latest=8 * 3600)],
             required_skills=["std"]),
        Stop(id="S6", node_index=7, coordinate=Coordinate(lon=coords[7][0], lat=coords[7][1]),
             demand={"weight": 1.0}, service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0, latest=8 * 3600)],
             required_skills=["std"]),
    ]
    fleet = [
        VehicleClass(id="van", count=None, capacity={"weight": 50.0},
                     home_depot_id="D0", provides_skills=["std"],
                     allowed_zone_tags=["downtown"],
                     cost_per_second=0.005, cost_per_meter=0.001,
                     fixed_cost=20.0, peak_hour_cost_multiplier=1.5,
                     shift=ShiftRule(ruleset="eu561",
                                     shift_start_window=TimeWindow(earliest=0,
                                                                    latest=3600))),
        VehicleClass(id="ev", count=2, capacity={"weight": 30.0},
                     home_depot_id="D1", provides_skills=["std"],
                     max_route_meters=100000.0),
    ]
    zones = [
        Zone(tag="downtown", node_indices=[2, 5, 6], kind="access"),
        Zone(tag="park", node_indices=[7], kind="embargo",
             active_window=TimeWindow(earliest=0, latest=7200)),
    ]
    constraints = Constraints(
        drop_penalty=500.0, allow_drops=False,
        embargo_soft=True, embargo_penalty=100.0,
        per_route_fixed_cost=10.0,
        peak_windows=[TimeWindow(earliest=3600, latest=7200)],
        objective=ObjectiveConfig(
            operational_cost_weight=1.0,
            vehicle_count_weight=5.0,
            quality_terms={
                "route_crossings": 0.5,
                "load_balance_cv": 1.0,
                "time_window_slack": 0.1,
            },
            soft_tw_penalty_per_sec=0.02,
        ),
    )
    return Problem.from_matrix(od=od, depots=depots, stops=stops, fleet=fleet,
                                zones=zones, constraints=constraints,
                                t0_epoch=1700000000.0)


def test_problem_full_coverage_json_roundtrip():
    p = _build_full_problem()
    body = p.model_dump_json()
    p2 = Problem.model_validate_json(body)
    # pydantic equality goes by all fields — minus the validator-stashed
    # _input_warnings which is private. Compare via re-dump.
    assert p.model_dump_json() == p2.model_dump_json()


def test_solution_full_coverage_json_roundtrip():
    """After solving the full-coverage Problem, the Solution round-trips."""
    p = _build_full_problem()
    sol = solve(p, SolveOptions(budget_seconds=1.0, seed=0))
    body = sol.model_dump_json()
    from openvrp import Solution
    sol2 = Solution.model_validate_json(body)
    assert sol.model_dump_json() == sol2.model_dump_json()


def test_eu561_shift_rule_survives_roundtrip():
    p = _build_full_problem()
    body = p.model_dump_json()
    p2 = Problem.model_validate_json(body)
    # The eu561 ruleset is preserved on the first vehicle class
    assert p2.fleet[0].shift is not None
    assert p2.fleet[0].shift.ruleset == "eu561"


def test_pd_pair_survives_roundtrip():
    p = _build_full_problem()
    p2 = Problem.model_validate_json(p.model_dump_json())
    s2 = next(s for s in p2.stops if s.id == "S2")
    s3 = next(s for s in p2.stops if s.id == "S3")
    assert s2.pickup_of == "S3"
    assert s3.delivery_of == "S2"


def test_quality_terms_survive_roundtrip():
    p = _build_full_problem()
    p2 = Problem.model_validate_json(p.model_dump_json())
    qt = p2.constraints.objective.quality_terms
    assert "route_crossings" in qt
    assert qt["load_balance_cv"] == 1.0
