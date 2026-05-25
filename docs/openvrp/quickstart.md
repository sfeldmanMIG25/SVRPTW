# Quickstart

Solve an OD-only constraint-rich problem in a handful of lines. **Core
install only** — no network or pyvrp extras needed.

```python
import numpy as np
from openvrp import (
    Depot, Stop, VehicleClass, TimeWindow, SolveOptions, solve_od,
)

# 1. Build any OD matrix. Index 0 is the depot; 1..N are stops.
n = 8
rng = np.random.default_rng(0)
coords = [(0.0, 0.0)] + [(float(rng.uniform(-10, 10)),
                          float(rng.uniform(-10, 10))) for _ in range(n)]
nn = len(coords)
T = np.zeros((nn, nn))
for i in range(nn):
    for j in range(nn):
        if i != j:
            T[i, j] = float(np.hypot(coords[i][0] - coords[j][0],
                                     coords[i][1] - coords[j][1]) * 60.0)

# 2. Describe the problem
depots = [Depot(id="warehouse", node_index=0)]
stops = [
    Stop(id=f"customer_{i}", node_index=i, demand={"weight": 5.0},
         service_seconds=300.0,
         time_windows=[TimeWindow(earliest=0.0, latest=8 * 3600)])
    for i in range(1, n + 1)
]
fleet = [VehicleClass(id="van", count=None, capacity={"weight": 50.0},
                     home_depot_id="warehouse",
                     cost_per_second=0.005, cost_per_meter=0.001,
                     fixed_cost=20.0)]

# 3. Solve
sol = solve_od(T, stops, depots, fleet,
               options=SolveOptions(budget_seconds=2.0, seed=0))

# 4. Inspect the result
print(f"status={sol.status}  K={sol.vehicles_used}  "
      f"objective=${sol.objective_value:.2f}")
for r in sol.routes:
    stops_seq = " -> ".join(v.stop_id for v in r.visits if v.kind != "depot")
    print(f"  {r.vehicle_class_id}#{r.vehicle_ordinal}: {stops_seq}")
```

Expected output: `status=feasible`, 1-2 vehicles used, every customer
served. The full event timeline lives in `sol.routes[0].events`; the
8-metric quality report in `sol.quality_report.solution_level`.

## What just happened

* `solve_od` is sugar — it builds an internal `Problem(mode='od')` and
  delegates to `solve(problem, options)`.
* The default `construction='auto'` picks PyVRP if `openvrp[pyvrp]` is
  installed, else the **native PyVRP-free fast solver** (a vectorized
  nearest-neighbor + merge_routes + 2-opt + cross-route relocate). The
  native solver scales to **N=1000 in single-digit seconds**.
* `sol` is a typed pydantic Solution. It round-trips through JSON
  losslessly (`sol.model_validate_json(sol.model_dump_json())`).
* `sol.quality_report.solution_level` reports all 8 catalog metrics
  even when none are weighted into the objective.
