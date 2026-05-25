# Schema reference

Auto-generated from the pydantic docstrings (SPEC-OPENVRP-09 J3).
Re-build by running `python -c "import openvrp; help(openvrp)"` or
calling pydantic's `model_json_schema()` per type.

## Public input types (SPEC-OPENVRP-01)

```python
from openvrp import (
    Coordinate, TimeWindow, Stop, Depot, VehicleClass, ShiftRule,
    Zone, Network, SnapConfig, ODMatrix, ObjectiveConfig,
    Constraints, SolveOptions, Problem, QUALITY_CATALOG,
)
```

| Type | Purpose |
|---|---|
| `Coordinate` | `(lon, lat)` EPSG:4326 |
| `TimeWindow` | Half-open window in seconds; `hard=False` makes it soft |
| `Stop` | Demand point with TW, demand, skills, PD link, priority |
| `Depot` | Class home base with optional open/close window |
| `VehicleClass` | Template; `count=None` ⇒ unbounded, solver minimizes |
| `ShiftRule` | EU 561 / US HOS / custom rulesets |
| `Zone` | Embargo or access zone, optionally time-windowed |
| `Network` | OSM place / bbox / graph file — `[network]` extra needed |
| `SnapConfig` | Caller-controlled snapping policy |
| `ODMatrix` | Pre-built OD; lossless JSON round-trip via nested lists |
| `ObjectiveConfig` | Caller's top-level lever over "best" |
| `Constraints` | Global toggles + penalties + objective |
| `SolveOptions` | Per-solve knobs (budget, seed, threads, on_progress) |
| `Problem` | Top-level — `Problem.from_matrix()` or `from_network()` |

## Public output types (SPEC-OPENVRP-02)

```python
from openvrp import (
    Visit, Event, BreakEvent, DepotDepartureEvent, DepotReturnEvent,
    ShiftStartEvent, ZoneEnterEvent, ZoneExitEvent, RechargeEvent,
    DropEvent, DroppedStop, LegGeometry, RouteEconomics, QualityReport,
    Route, Solution, SolveDiagnostics, ProgressEvent,
)
```

| Type | Purpose |
|---|---|
| `Visit` | Single stop arrival on a route |
| `Event` | Discriminated union of break / shift / zone / recharge / drop |
| `LegGeometry` | Real along-network polyline per route leg |
| `RouteEconomics` | Cost decomposition; enumerated + quality_penalty sums to total |
| `QualityReport` | Always-reported catalog metrics (solution-level + per-route) |
| `Route` | Vehicle ordinal + visits + events + geometry + economics |
| `Solution` | Top-level; `objective_value == economics_total.total` ±1e-6 |
| `SolveDiagnostics` | Operator pool, feasibility blockers, snap report, timings |
| `ProgressEvent` | Payload of `SolveOptions.on_progress` callback |

## Quality catalog (SPEC-OPENVRP-04 §3, D9)

```python
from openvrp import QUALITY_CATALOG
assert len(QUALITY_CATALOG) == 8
```

Setting an unknown key in `ObjectiveConfig.quality_terms` is a
validation error that lists the catalog. No visual/VLM metric exists.
