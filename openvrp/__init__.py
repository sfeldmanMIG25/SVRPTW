"""OpenVRP — pip-installable VRPTW solver with heterogeneous fleets,
full driver-hour rulesets, network-aware geometry, and MIT licensing.

```python
from openvrp import solve, solve_od, Problem, SolveOptions

# OD-mode (no extras needed):
sol = solve_od(time_matrix=T, stops=stops, depots=depots, fleet=fleet,
               options=SolveOptions(budget_seconds=10.0))

# Network-mode (requires `pip install openvrp[network]`):
problem = Problem.from_network(network=Network(source='osm_place',
                                                osm_place='Manhattan, New York'),
                               depots=depots, stops=stops, fleet=fleet)
sol = solve(problem, SolveOptions(budget_seconds=30.0))
sol.to_geojson('manhattan_routes.geojson')   # drops into QGIS / Leaflet
```

See the SPEC-OPENVRP-* documents in `WorldsFinestVRP/` for the
contracts behind every public type.
"""
from __future__ import annotations

__version__ = "0.1.0"

# ============================================================
# Public re-exports ONLY. Everything else is implementation detail.
# ============================================================
from openvrp.api import solve, solve_od
from openvrp.errors import (
    GeometryUnavailable,
    Issue,
    MissingExtra,
    OpenVRPError,
    ProblemValidationError,
    SolveAborted,
    StopSolve,
)
from openvrp.schema.input import (
    QUALITY_CATALOG,
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
    DropEvent,
    DroppedStop,
    Event,
    LegGeometry,
    ProgressEvent,
    QualityReport,
    RechargeEvent,
    ReplenishEvent,
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
    "__version__",
    # API
    "solve", "solve_od",
    # Errors
    "OpenVRPError", "ProblemValidationError", "MissingExtra",
    "GeometryUnavailable", "SolveAborted", "StopSolve", "Issue",
    # Schema input
    "Coordinate", "TimeWindow", "Stop", "Depot", "VehicleClass", "ShiftRule",
    "Zone", "Network", "SnapConfig", "ODMatrix", "ObjectiveConfig",
    "Constraints", "SolveOptions", "Problem", "QUALITY_CATALOG",
    # Schema output
    "Visit", "Event", "BreakEvent", "DepotDepartureEvent", "DepotReturnEvent",
    "ShiftStartEvent", "ZoneEnterEvent", "ZoneExitEvent", "RechargeEvent",
    "ReplenishEvent", "DropEvent", "DroppedStop", "LegGeometry", "RouteEconomics",
    "QualityReport", "Route", "Solution", "SolveDiagnostics", "ProgressEvent",
]
