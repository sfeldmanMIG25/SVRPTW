"""Schema round-trip tests (SPEC-OPENVRP-01 A10, SPEC-OPENVRP-02 B7,
SPEC-OPENVRP-05 F1).

Every public type T must satisfy ``T == T.model_validate_json(T.model_dump_json())``.
"""
from __future__ import annotations

import pytest

from openvrp import (
    Constraints,
    Coordinate,
    Depot,
    Issue,
    Network,
    ObjectiveConfig,
    Problem,
    QUALITY_CATALOG,
    ShiftRule,
    Stop,
    TimeWindow,
    VehicleClass,
    Zone,
)
from openvrp.errors import ProblemValidationError
from openvrp.schema.output import (
    BreakEvent,
    DepotDepartureEvent,
    LegGeometry,
    QualityReport,
    Route,
    RouteEconomics,
    Solution,
    Visit,
)


def _basic_fleet() -> list[VehicleClass]:
    return [VehicleClass(id="van", count=None, capacity={"weight": 100.0},
                         home_depot_id="D0", provides_skills=[])]


def test_coordinate_roundtrip():
    c = Coordinate(lon=-74.0, lat=40.7)
    c2 = Coordinate.model_validate_json(c.model_dump_json())
    assert c == c2


def test_time_window_invariants():
    tw = TimeWindow(earliest=0, latest=3600, hard=True)
    tw2 = TimeWindow.model_validate_json(tw.model_dump_json())
    assert tw == tw2
    with pytest.raises(Exception):
        TimeWindow(earliest=100, latest=50)


def test_stop_roundtrip_full():
    s = Stop(id="ACME-1", coordinate=Coordinate(lon=0, lat=0),
             demand={"weight": 5.0},
             service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0, latest=3600)],
             required_skills=["hazmat"], priority=2)
    s2 = Stop.model_validate_json(s.model_dump_json())
    assert s == s2


def test_problem_validation_rejects_pd_self_reference():
    with pytest.raises(Exception):
        Stop(id="A", coordinate=Coordinate(lon=0, lat=0), pickup_of="A")


def test_problem_validation_missing_demand_key():
    """SPEC-OPENVRP-01 A3: stop.demand key must be in fleet capacity-dim union."""
    stops = [Stop(id="S1", coordinate=Coordinate(lon=0, lat=0),
                  demand={"unknown_dim": 1.0})]
    fleet = _basic_fleet()
    depots = [Depot(id="D0", coordinate=Coordinate(lon=0, lat=0))]
    with pytest.raises(ProblemValidationError) as ei:
        Problem(mode="network", network=Network(source="osm_place", osm_place="x"),
                depots=depots, stops=stops, fleet=fleet)
    assert any("demand_key_unknown" in i.code for i in ei.value.issues)


def test_problem_validation_unservable_stop_no_drops():
    """A5: a stop unservable by every class ⇒ error if allow_drops=False."""
    stops = [Stop(id="S1", coordinate=Coordinate(lon=0, lat=0),
                  demand={"weight": 9999.0})]
    fleet = _basic_fleet()
    depots = [Depot(id="D0", coordinate=Coordinate(lon=0, lat=0))]
    with pytest.raises(ProblemValidationError) as ei:
        Problem(mode="network", network=Network(source="osm_place", osm_place="x"),
                depots=depots, stops=stops, fleet=fleet)
    assert any(i.code == "stop.unservable" for i in ei.value.issues)


def test_problem_validation_unservable_stop_allow_drops_warns():
    stops = [Stop(id="S1", coordinate=Coordinate(lon=0, lat=0),
                  demand={"weight": 9999.0})]
    fleet = _basic_fleet()
    depots = [Depot(id="D0", coordinate=Coordinate(lon=0, lat=0))]
    p = Problem(mode="network", network=Network(source="osm_place", osm_place="x"),
                depots=depots, stops=stops, fleet=fleet,
                constraints=Constraints(allow_drops=True))
    warnings = p.input_warnings()
    assert any(w.code == "stop.unservable_droppable" for w in warnings)


def test_quality_terms_unknown_key_rejected():
    """A9: unknown quality_terms key is a validation error listing the catalog."""
    with pytest.raises(Exception) as ei:
        ObjectiveConfig(quality_terms={"VLM_score": 1.0})
    assert "VLM_score" in str(ei.value) or "catalog" in str(ei.value).lower()


def test_quality_catalog_known_keys_accepted():
    cfg = ObjectiveConfig(quality_terms={"route_crossings": 0.5,
                                          "load_balance_cv": 1.0})
    assert "route_crossings" in cfg.quality_terms


def test_shift_rule_eu561_materialization():
    """A6: ruleset='eu561' instantiates the full parameter set; explicit
    fields override individual parameters without discarding the rest."""
    from openvrp.engine.fleet import materialize_shift_rule
    rule = ShiftRule(ruleset="eu561")
    m = materialize_shift_rule(rule)
    # EU 561: 4.5h continuous drive, 45min break, 9h daily drive, etc.
    assert m.max_drive_seconds_before_break == 4.5 * 3600
    assert m.break_seconds == 45 * 60
    assert m.daily_drive_cap_seconds == 9 * 3600
    assert m.min_weekly_rest_seconds == 45 * 3600
    # Override break_seconds; rest of EU 561 structure preserved
    rule2 = ShiftRule(ruleset="eu561", break_seconds=30 * 60)
    m2 = materialize_shift_rule(rule2)
    assert m2.break_seconds == 30 * 60
    assert m2.daily_drive_cap_seconds == 9 * 3600
    assert m2.min_weekly_rest_seconds == 45 * 3600


def test_shift_rule_us_hos_materialization():
    """US HOS: 11h drive in 14h window, 30min break after 8h, 10h reset."""
    from openvrp.engine.fleet import materialize_shift_rule
    m = materialize_shift_rule(ShiftRule(ruleset="us_hos"))
    assert m.max_drive_seconds_before_break == 8 * 3600
    assert m.break_seconds == 30 * 60
    assert m.daily_drive_cap_seconds == 11 * 3600
    assert m.daily_onduty_cap_seconds == 14 * 3600


def test_route_economics_roundtrip():
    e = RouteEconomics(total=123.45, fixed_cost=10, time_cost=50,
                       distance_cost=20, quality_penalty={"route_crossings": 5.0})
    e2 = RouteEconomics.model_validate_json(e.model_dump_json())
    assert e == e2


def test_solution_roundtrip_minimal():
    sol = Solution(
        status="feasible",
        routes=[],
        objective_value=0.0,
        economics_total=RouteEconomics(total=0.0),
        quality_report=QualityReport(),
    )
    sol2 = Solution.model_validate_json(sol.model_dump_json())
    assert sol == sol2


def test_break_event_discriminator_roundtrip():
    """B7: Event discriminated union round-trips. Specifically the BreakEvent
    variant with rest_kind='split_break_segment' is preserved."""
    e = BreakEvent(at_seconds=3600, duration_seconds=900,
                   rule="eu561:split_break_segment",
                   rest_kind="split_break_segment", segment_index=0)
    # Embed in a Route to force the discriminated union path
    r = Route(vehicle_class_id="V", vehicle_ordinal=0, home_depot_id="D",
              economics=RouteEconomics(total=0.0), events=[e])
    r2 = Route.model_validate_json(r.model_dump_json())
    assert isinstance(r2.events[0], BreakEvent)
    assert r2.events[0].rest_kind == "split_break_segment"


def test_quality_catalog_size():
    assert len(QUALITY_CATALOG) == 8
    # No visual / VLM key (D9)
    for k in QUALITY_CATALOG:
        assert "vlm" not in k.lower() and "visual" not in k.lower()


def test_issue_carries_location():
    """SPEC-OPENVRP-08 §1: Issue.location uses stop/field-specific path."""
    issue = Issue(severity="error", code="stop.demand_key_unknown",
                  message="...", location="stops[id='ACME-12'].demand")
    assert "ACME-12" in issue.location


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
