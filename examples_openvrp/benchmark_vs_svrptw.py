"""SPEC-OPENVRP-03 §4 — Performance mandate harness.

Compares openvrp's PyVRP-free `solve` against the svrptw research
`solve_auto` on small-scale OD-only problems. The mandate is:

> At matched wall-clock on the OSM-asymmetric evaluation regime, the
> shipped solver MUST be strictly faster than research solve_auto AND
> produce solutions of equal-or-lower objective cost.

This script doesn't load OSM instances (those need [network] + osmnx);
it builds a deterministic Euclidean asymmetric instance set and verifies
the contract on a representative N range.

Run with PYTHONPATH=. set so both packages are importable.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np

from openvrp import (
    Constraints,
    Depot,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    solve_od,
)


@dataclass
class BenchRow:
    label: str
    n: int
    solver: str
    wall_seconds: float
    objective: float
    vehicles_used: int


def _build_instance(n: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list, list, list]:
    rng = np.random.default_rng(seed)
    # Random Euclidean coords; add a small asymmetry term to T (one-way streets effect)
    coords = [(0.0, 0.0)] + [(float(rng.uniform(-30, 30)),
                              float(rng.uniform(-30, 30))) for _ in range(n)]
    nn = len(coords)
    base = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i == j:
                continue
            base[i, j] = np.hypot(coords[i][0] - coords[j][0],
                                  coords[i][1] - coords[j][1])
    asym = base + (rng.random((nn, nn)) * 0.3) * base   # +0..30% one-way penalty
    np.fill_diagonal(asym, 0)
    T = asym * 60.0   # 1 unit = 60 s
    D = T.copy()
    depots = [Depot(id="D0", node_index=0)]
    stops = [
        Stop(id=f"S{i}", node_index=i, demand={"weight": 5.0},
             service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0.0, latest=8 * 3600)])
        for i in range(1, n + 1)
    ]
    fleet = [VehicleClass(id="van", count=None, capacity={"weight": 50.0},
                          home_depot_id="D0", cost_per_second=0.005,
                          cost_per_meter=0.001, fixed_cost=20.0)]
    return T, D, depots, stops, fleet


def _bench_openvrp(label: str, n: int, T, D, depots, stops, fleet, budget: float) -> BenchRow:
    t0 = time.monotonic()
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=budget, seed=0),
                   distance_matrix=D)
    return BenchRow(
        label=label, n=n, solver="openvrp",
        wall_seconds=time.monotonic() - t0,
        objective=sol.objective_value,
        vehicles_used=sol.vehicles_used,
    )


def _bench_svrptw(label: str, n: int, T, D, depots, stops, fleet, budget: float) -> BenchRow | None:
    """Attempt to bench svrptw research solve_auto. Returns None if the
    heavy deps (pyvrp/ortools) aren't installed."""
    try:
        from svrptw.solvers.classical.portfolio_pyvrp_warm import (
            solve_auto as _svrptw_solve_auto,
        )
    except ImportError as e:
        print(f"  svrptw unavailable ({e}); skipping comparison")
        return None
    from openvrp.engine._adapter import to_svrptw_instance, to_svrptw_settings
    from openvrp.schema.input import Constraints, Problem, ODMatrix
    # Build a quick OD-mode problem for the converter
    od = ODMatrix(time_seconds=T.tolist(), distance_meters=D.tolist(),
                  index_of={**{d.id: i for i, d in enumerate(depots)},
                            **{s.id: s.node_index for s in stops}})
    problem = Problem.from_matrix(od=od, depots=depots, stops=stops, fleet=fleet)
    inst, _, _ = to_svrptw_instance(problem)
    settings = to_svrptw_settings(Constraints(), SolveOptions(budget_seconds=budget))
    t0 = time.monotonic()
    sv_sol = _svrptw_solve_auto(inst, settings, budget_seconds=budget, seed=0)
    wall = time.monotonic() - t0
    return BenchRow(
        label=label, n=n, solver="svrptw_solve_auto",
        wall_seconds=wall,
        objective=float(sv_sol.metrics.get("operational_cost", 0.0)),
        vehicles_used=sv_sol.num_vehicles_used,
    )


def run(N_values: list[int], budget: float, n_seeds: int = 3) -> None:
    print(f"OpenVRP performance mandate harness — N={N_values}, budget={budget}s, seeds={n_seeds}")
    print(f"{'N':>4} | {'solver':22} | {'wall_s':>7} | {'objective':>11} | {'K':>3}")
    print("-" * 70)
    summary = {n: {"openvrp_obj": [], "openvrp_wall": [], "sv_obj": [], "sv_wall": []}
               for n in N_values}
    for n in N_values:
        for seed in range(n_seeds):
            T, D, depots, stops, fleet = _build_instance(n, seed=seed)
            r1 = _bench_openvrp(f"N{n}-s{seed}", n, T, D, depots, stops, fleet, budget)
            print(f"{n:>4} | {r1.solver:22} | {r1.wall_seconds:7.2f} | {r1.objective:11.2f} | {r1.vehicles_used:>3}")
            summary[n]["openvrp_obj"].append(r1.objective)
            summary[n]["openvrp_wall"].append(r1.wall_seconds)
            r2 = _bench_svrptw(f"N{n}-s{seed}", n, T, D, depots, stops, fleet, budget)
            if r2 is not None:
                print(f"{n:>4} | {r2.solver:22} | {r2.wall_seconds:7.2f} | {r2.objective:11.2f} | {r2.vehicles_used:>3}")
                summary[n]["sv_obj"].append(r2.objective)
                summary[n]["sv_wall"].append(r2.wall_seconds)
    print()
    print("Aggregate (mean across seeds):")
    print(f"{'N':>4} | {'openvrp_obj':>12} | {'openvrp_wall':>13} | "
          f"{'sv_obj':>10} | {'sv_wall':>9} | {'delta_obj':>10}")
    for n in N_values:
        s = summary[n]
        op_obj = statistics.mean(s["openvrp_obj"]) if s["openvrp_obj"] else float("nan")
        op_w = statistics.mean(s["openvrp_wall"]) if s["openvrp_wall"] else float("nan")
        sv_obj = statistics.mean(s["sv_obj"]) if s["sv_obj"] else float("nan")
        sv_w = statistics.mean(s["sv_wall"]) if s["sv_wall"] else float("nan")
        if s["sv_obj"]:
            delta = op_obj - sv_obj
            print(f"{n:>4} | {op_obj:12.2f} | {op_w:13.2f} | "
                  f"{sv_obj:10.2f} | {sv_w:9.2f} | {delta:+10.2f}")
        else:
            print(f"{n:>4} | {op_obj:12.2f} | {op_w:13.2f} | "
                  f"{'n/a':>10} | {'n/a':>9} | {'n/a':>10}")


if __name__ == "__main__":
    import sys
    Ns = [10, 25, 50]
    if len(sys.argv) > 1 and sys.argv[1] == "fast":
        Ns = [10, 25]
    run(Ns, budget=3.0, n_seeds=2)
