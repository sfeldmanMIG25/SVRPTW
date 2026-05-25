# Heterogeneous fleet deep-dive

The mixed-fleet scenario commercial tools require tuning for. We build
a small-van + large-truck + refrigerated-unit fleet, run it, and read
the `RouteEconomics` decomposition to see why the solver chose its mix.

```python
import numpy as np
from openvrp import (
    Depot, Stop, VehicleClass, TimeWindow, SolveOptions, ObjectiveConfig,
    Constraints, Problem, ODMatrix, solve,
)

n = 10
rng = np.random.default_rng(0)
coords = [(0.0, 0.0)] + [(float(rng.uniform(-15, 15)),
                          float(rng.uniform(-15, 15))) for _ in range(n)]
nn = len(coords)
T = np.zeros((nn, nn))
for i in range(nn):
    for j in range(nn):
        if i != j:
            T[i, j] = float(np.hypot(coords[i][0] - coords[j][0],
                                     coords[i][1] - coords[j][1]) * 60.0)

depots = [Depot(id="W", node_index=0)]
stops = [
    # Some stops need cold-chain (refrigerated only)
    Stop(id=f"S{i}", node_index=i,
         demand={"weight": float(rng.integers(2, 20))},
         time_windows=[TimeWindow(earliest=0, latest=8 * 3600)],
         required_skills=["cold_chain"] if i % 4 == 0 else [])
    for i in range(1, n + 1)
]
fleet = [
    VehicleClass(id="van", count=None, capacity={"weight": 30.0},
                 home_depot_id="W",
                 cost_per_second=0.004, cost_per_meter=0.0008,
                 fixed_cost=15.0),
    VehicleClass(id="truck", count=None, capacity={"weight": 100.0},
                 home_depot_id="W",
                 cost_per_second=0.010, cost_per_meter=0.0015,
                 fixed_cost=60.0),
    VehicleClass(id="refrigerated", count=None, capacity={"weight": 80.0},
                 home_depot_id="W",
                 cost_per_second=0.015, cost_per_meter=0.0020,
                 fixed_cost=100.0,
                 provides_skills=["cold_chain"]),
]

od = ODMatrix(time_seconds=T.tolist(),
              index_of={**{d.id: d.node_index for d in depots},
                        **{s.id: s.node_index for s in stops}})
constraints = Constraints(
    per_route_fixed_cost=10.0,
    objective=ObjectiveConfig(vehicle_count_weight=20.0))

problem = Problem.from_matrix(od=od, depots=depots, stops=stops,
                              fleet=fleet, constraints=constraints)
sol = solve(problem, SolveOptions(budget_seconds=2.0, seed=0))

print(f"K_used={sol.vehicles_used}  K_min={sol.vehicles_minimum_found}  "
      f"obj=${sol.objective_value:.2f}")
print("Per-route economics:")
for r in sol.routes:
    e = r.economics
    print(f"  {r.vehicle_class_id}#{r.vehicle_ordinal}  fixed=${e.fixed_cost:.0f}  "
          f"time=${e.time_cost:.1f}  dist=${e.distance_cost:.2f}  total=${e.total:.2f}")
print(f"Fleet minimization audit:")
print(f"  used={sol.diagnostics.fleet_min_report['used']}  "
      f"min_found={sol.diagnostics.fleet_min_report['minimum_found']}")
```

## How to read the result

* The solver prefers the cheapest *eligible* class. Cold-chain stops
  force a refrigerated truck whenever they're on the same route.
* `vehicle_count_weight` intensifies the always-active fleet
  minimization (D7). Set it higher to trade some operational cost for
  one fewer vehicle.
* `fleet_min_report.minimum_found` is the smallest fleet a greedy
  post-hoc merge of the returned routes could land at. If
  `minimum_found < used`, the diagnostics name the weighted-objective
  reason for keeping extra vehicles (e.g. quality term, soft TW slack).
