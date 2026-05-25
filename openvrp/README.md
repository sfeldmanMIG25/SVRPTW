# OpenVRP

A **pip-installable, dependency-honest** Python library for rich Vehicle
Routing Problems with Time Windows and heterogeneous fleets.

Give it an OD matrix (or a road network) plus a constraint configuration;
get back structured route objects with the logical plan, full event
timeline, cost decomposition, and (network path) real along-network geometry.

## Why

Esri/ArcGIS users stay on Esri for its UI and curated data; **OpenVRP
doesn't compete there**. OpenVRP competes by being:

- **Free, open, MIT-licensed** — no commercial-solver bill, no surprises
- **A library** — drop it into your data pipeline, not a separate app
- **Constraint-rich** — heterogeneous fleets, EU 561 / US HOS driver-hour
  rulesets, embargo zones, pickup-delivery, EV range, multi-depot,
  shift windows, peak-hour surcharges
- **Caller-configurable objective** — fold operational-quality metrics
  (load balance, time-window slack, route crossings) into optimization,
  not just into reports
- **Network-aware** — give it an OSM city name; get back real polylines
  that drop into QGIS/Leaflet via GeoJSON, no conversion

No VLM/visual scoring anywhere. Quality means measurable geometry and
balance, not what a model "thinks" looks pretty.

## Install

```bash
# Core: OD-only solving, full constraint catalog, JSON I/O
pip install openvrp

# Network-aware: snap to OSM, geometry reconstruction, GeoJSON
pip install openvrp[network]

# PyVRP accelerator (optional)
pip install openvrp[pyvrp]

# Everything
pip install openvrp[all]
```

## 60-second quickstart

```python
import numpy as np
from openvrp import solve_od, Depot, Stop, VehicleClass, TimeWindow, SolveOptions

# Build any OD matrix (nodes: [depot, stop_1, stop_2, ...])
n = 10
T = np.random.rand(n + 1, n + 1) * 1800  # 0..30 min travel
np.fill_diagonal(T, 0)

depots = [Depot(id="D0", node_index=0)]
stops = [
    Stop(id=f"S{i}", node_index=i, demand={"weight": 5.0},
         service_seconds=300,
         time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
    for i in range(1, n + 1)
]
fleet = [VehicleClass(id="van", count=None, capacity={"weight": 50.0},
                      home_depot_id="D0", cost_per_second=0.005,
                      cost_per_meter=0.001, fixed_cost=20.0)]

sol = solve_od(T, stops, depots, fleet,
               options=SolveOptions(budget_seconds=10.0, seed=0))

print(f"status={sol.status}")
print(f"vehicles_used={sol.vehicles_used}  objective=${sol.objective_value:.2f}")
for r in sol.routes:
    stops_seq = " -> ".join(v.stop_id for v in r.visits if v.kind != "depot")
    print(f"  {r.vehicle_class_id}#{r.vehicle_ordinal}: {stops_seq}")
```

## Network-aware (OSM)

```python
from openvrp import Problem, Network, Depot, Stop, VehicleClass, Coordinate, solve

problem = Problem.from_network(
    network=Network(source="osm_place", osm_place="Manhattan, New York, USA"),
    depots=[Depot(id="warehouse", coordinate=Coordinate(lon=-74.01, lat=40.72))],
    stops=[
        Stop(id=f"customer_{i}", coordinate=Coordinate(lon=-74.0 + i * 0.001, lat=40.72 + i * 0.001),
             demand={"weight": 1.0}, service_seconds=300,
             time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
        for i in range(20)
    ],
    fleet=[VehicleClass(id="van", count=None, capacity={"weight": 50.0},
                        home_depot_id="warehouse")],
)
sol = solve(problem, SolveOptions(budget_seconds=30.0))
sol.to_geojson("manhattan_routes.geojson", include={"events": True})
# Drop into QGIS / Leaflet / kepler.gl / deck.gl — no conversion
```

## Driver-hour rulesets (EU 561 / US HOS)

```python
from openvrp import VehicleClass, ShiftRule

# Full EU 561: 4.5h continuous drive then 45-min break (or 15+30 split);
# 9h daily drive, extendable to 10h ≤2/wk; 11h daily rest; 45h weekly rest.
fleet = [VehicleClass(
    id="long_haul",
    capacity={"weight": 200.0},
    home_depot_id="warehouse",
    shift=ShiftRule(ruleset="eu561"),
)]

# Full US HOS: 11h driving in 14h on-duty window; 30-min break after 8h;
# 10h off-duty reset; 60/70h in 7/8d; 34h restart.
us_fleet = [VehicleClass(
    id="us_truck",
    capacity={"weight": 200.0},
    home_depot_id="warehouse",
    shift=ShiftRule(ruleset="us_hos"),
)]
```

The solver inserts break/rest events at the latest feasible position; the
returned `Route.events` carries each break with its `rest_kind`
(`break`, `split_break_segment`, `daily_rest`, `reduced_daily_rest`,
`weekly_rest`) and the regulation parameter that forced it.

## Caller-configurable objective

Default is pure operational cost. Opt into quality terms:

```python
from openvrp import Constraints, ObjectiveConfig

constraints = Constraints(
    per_route_fixed_cost=50.0,   # $ per active vehicle-day
    objective=ObjectiveConfig(
        operational_cost_weight=1.0,
        vehicle_count_weight=0.0,             # intensify fleet shrinkage (always active)
        quality_terms={
            "route_crossings": 0.5,           # penalize visual chaos
            "load_balance_cv": 1.0,           # spread work evenly across drivers
            "time_window_slack": 0.1,         # reward staying ahead of TWs
        },
        soft_tw_penalty_per_sec=0.05,         # soft TW lateness pricing
    ),
)
```

The full catalog (8 metrics) — every one is computed and **always
reported** in `solution.quality_report`, regardless of weighting:

| key | meaning |
|---|---|
| `route_crossings` | inter-route segment intersections (lower better) |
| `mean_detour_ratio` | actual-leg ÷ straight OD (lower better) |
| `load_balance_cv` | CV of route loads (lower better) |
| `load_balance_gini` | Gini of route loads (lower better) |
| `time_window_slack` | -mean unused slack (higher slack better) |
| `intra_route_compactness` | mean intra-route spread (lower better) |
| `cross_route_overlap` | bbox overlap across routes (lower better) |
| `quality_per_route` | K-fair composite (higher better) |

There is **no visual / VLM metric** in this catalog and none can be
added (SPEC-OPENVRP-00 D9).

## Cancellation & progress

```python
from openvrp import StopSolve, ProgressEvent

def on_progress(ev: ProgressEvent) -> None:
    print(f"[{ev.phase}] {ev.elapsed_seconds:.1f}s  best=${ev.best_objective}")
    if ev.elapsed_seconds > 5.0:
        raise StopSolve()   # clean cancellation; returns best-so-far

sol = solve(problem, SolveOptions(budget_seconds=60.0, on_progress=on_progress))
```

## Determinism

Identical `Problem` + `seed` ⇒ identical stop sequence in every route
across runs, across `threads=1` vs `N`, and across repeated `fast`
constructions. Timings are a pure function of sequence + input.

## License

MIT. All hard dependencies of the core and `[network]` install are
permissive-compatible.
