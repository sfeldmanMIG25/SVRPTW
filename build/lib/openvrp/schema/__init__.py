"""Public schema for OpenVRP (SPEC-OPENVRP-01, -02).

These are the *only* types a caller constructs and inspects. Internal
solver structures live under ``openvrp.engine`` and are not exposed.
"""
from openvrp.schema.input import (
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
)
from openvrp.schema.output import (
    BreakEvent,
    DepotDepartureEvent,
    DepotReturnEvent,
    DroppedStop,
    Event,
    LegGeometry,
    ProgressEvent,
    QualityReport,
    RechargeEvent,
    Route,
    RouteEconomics,
    ShiftStartEvent,
    Solution,
    SolveDiagnostics,
    Visit,
    ZoneEnterEvent,
    ZoneExitEvent,
)

__all__ = [
    # Input
    "Coordinate", "TimeWindow", "Stop", "Depot", "VehicleClass", "ShiftRule",
    "Zone", "Network", "SnapConfig", "ODMatrix", "ObjectiveConfig",
    "Constraints", "SolveOptions", "Problem",
    # Output
    "Visit", "Event", "BreakEvent", "DepotDepartureEvent", "DepotReturnEvent",
    "ShiftStartEvent", "ZoneEnterEvent", "ZoneExitEvent", "RechargeEvent",
    "DroppedStop", "LegGeometry", "RouteEconomics", "QualityReport",
    "Route", "Solution", "SolveDiagnostics", "ProgressEvent",
]
