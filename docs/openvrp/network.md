# Network tutorial

Take a road network → snap demand stops to nodes → solve → export
GeoJSON with real along-network polylines. Requires
`pip install openvrp[network]`.

```python
from openvrp import (
    Problem, Network, SnapConfig,
    Coordinate, Depot, Stop, VehicleClass, TimeWindow, SolveOptions, solve,
)

# 1. Network description — either an OSM place name, an OSM bbox, or
#    a local .graphml/.gpkg/.shp file.
network = Network(
    source="osm_place",
    osm_place="Manhattan, New York, USA",
    cache_dir="~/.cache/openvrp",          # cached after first fetch
    snapping=SnapConfig(mode="node",
                        max_snap_meters=250.0,   # default
                        strict=False),
)

# 2. Build the problem in network mode (stops use Coordinate, not node_index)
depots = [Depot(id="warehouse",
                coordinate=Coordinate(lon=-74.01, lat=40.72))]
stops = [
    Stop(id=f"customer_{i}",
         coordinate=Coordinate(lon=-74.0 + i * 0.001, lat=40.72 + i * 0.001),
         demand={"weight": 1.0}, service_seconds=300.0,
         time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
    for i in range(20)
]
fleet = [VehicleClass(id="van", count=None, capacity={"weight": 50.0},
                      home_depot_id="warehouse",
                      cost_per_second=0.005, cost_per_meter=0.001)]
problem = Problem.from_network(network=network, depots=depots, stops=stops,
                                fleet=fleet)

# 3. Solve. `ingest` fetches+caches the graph; `solve` runs construction
#    + refine; `finalize` traces per-leg shortest paths into LegGeometry.
sol = solve(problem, SolveOptions(budget_seconds=30.0, return_geometry=True))

# 4. Export to GeoJSON. Drops directly into QGIS, Leaflet, kepler.gl,
#    deck.gl, ArcGIS — no conversion. Optional events=True adds Points
#    for break/zone/recharge events.
sol.to_geojson("manhattan_routes.geojson", include={"events": True})

print(f"status={sol.status}  geometry={sol.geometry_status}")
print(f"snap_report (first 3): {sol.diagnostics.snap_report[:3]}")
```

## Notes

* **Cold graph load** happens before the solve timer starts — a fresh
  city fetch doesn't eat your construction budget.
* **Predecessors retained**: `scipy.sparse.csgraph.dijkstra(...,
  return_predecessors=True)` keeps the shortest-path tree so
  per-leg geometry is a path-traceback (O(path length)), not a
  re-search.
* **Triangle-inequality precondition** (D16): the OD layer assumes the
  triangle inequality holds across stops + depots. A cheap sampled check
  in `sol.diagnostics.triangle_violations` reports gross violations as
  a warning — never silently mutates the matrix, never fails the solve.
* **Snapping**: `strict=True` makes "stop > max_snap_meters from any
  node" a validation error; `strict=False` makes it a warning. Per-stop
  snap distance is recorded in `sol.diagnostics.snap_report`.
* **Geometry status**: `present` = every leg reconstructed; `partial` =
  some pairs disconnected (listed in `sol.geometry_failures`);
  `absent_failed` = full failure; `absent_od_mode` = ran in OD mode.
