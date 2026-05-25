"""60-second OpenVRP quickstart. Runs core-only (no network deps).

Output (run with `python examples_openvrp/quickstart.py`):
  status=feasible
  vehicles_used=2  objective=$XXX.XX
  ...
"""
from __future__ import annotations

import numpy as np

from openvrp import (
    Constraints,
    Depot,
    ObjectiveConfig,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    solve_od,
)


def main() -> None:
    rng = np.random.default_rng(42)
    n = 12
    # Build a deterministic OD: depot at index 0, stops 1..n at random angles
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = [(0.0, 0.0)] + [(float(np.cos(a) * 10), float(np.sin(a) * 10)) for a in angles]
    nn = len(coords)
    T = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i == j:
                continue
            T[i, j] = float(np.hypot(coords[i][0] - coords[j][0],
                                     coords[i][1] - coords[j][1]) * 120.0)
    D = T.copy()

    depots = [Depot(id="warehouse", node_index=0)]
    stops = [
        Stop(id=f"customer_{i}", node_index=i, demand={"weight": 5.0},
             service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0.0, latest=8 * 3600)])
        for i in range(1, n + 1)
    ]
    fleet = [VehicleClass(
        id="van", count=None, capacity={"weight": 50.0},
        home_depot_id="warehouse",
        cost_per_second=0.005, cost_per_meter=0.001, fixed_cost=20.0,
    )]
    constraints = Constraints(
        per_route_fixed_cost=10.0,
        objective=ObjectiveConfig(
            quality_terms={"route_crossings": 0.5, "load_balance_cv": 1.0},
        ),
    )

    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=5.0, seed=0),
                   distance_matrix=D, constraints=constraints)

    print(f"status={sol.status}")
    print(f"vehicles_used={sol.vehicles_used}  "
          f"vehicles_minimum_found={sol.vehicles_minimum_found}")
    print(f"objective=${sol.objective_value:.2f}  wall={sol.wall_seconds:.2f}s")
    print(f"operator_pool={sol.diagnostics.operator_pool}")
    print(f"quality:")
    for k, v in sol.quality_report.solution_level.items():
        print(f"  {k:30s} {v:.3f}")
    print("routes:")
    for r in sol.routes:
        stops_seq = " -> ".join(v.stop_id for v in r.visits if v.kind != "depot")
        print(f"  {r.vehicle_class_id}#{r.vehicle_ordinal}  "
              f"dist={r.distance_meters:.1f}m  dur={r.duration_seconds:.0f}s  "
              f"K={len(r.visits) - 2}  -> {stops_seq}")


if __name__ == "__main__":
    main()
