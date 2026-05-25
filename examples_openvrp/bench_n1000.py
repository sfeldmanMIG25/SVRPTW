"""SPEC-OPENVRP-03 §4 — N=1000 perf check.

User requirement: N=1000+ in under 15 minutes producing decent solutions,
with a caller-set runtime knob. Default budget here is 60s; can be
overridden via `--budget`.

Runs the native PyVRP-free solver on a synthetic asymmetric instance and
reports wall time, status, vehicles used, objective, and timings
breakdown. The instance is deterministic given a seed.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from openvrp import (
    Depot,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    solve_od,
)


def build_instance(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, list, list, list]:
    rng = np.random.default_rng(seed)
    coords = [(0.0, 0.0)]
    for _ in range(n):
        coords.append((float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50))))
    nn = len(coords)
    # Build a (nn, nn) Euclidean travel-time matrix with asymmetry noise.
    # Vectorized: O(N^2) numpy, ~10ms at N=1001.
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    base = np.hypot(dx, dy)
    asym_factor = 1.0 + rng.random((nn, nn)) * 0.3
    T = (base * asym_factor) * 60.0   # seconds (1 unit ~ 1 min)
    np.fill_diagonal(T, 0)
    D = T.copy()   # meters proxy
    depots = [Depot(id="D0", node_index=0)]
    stops = [
        Stop(id=f"S{i}", node_index=i, demand={"weight": float(rng.integers(1, 6))},
             service_seconds=300.0,
             time_windows=[TimeWindow(earliest=0.0, latest=8 * 3600)])
        for i in range(1, n + 1)
    ]
    fleet = [VehicleClass(id="van", count=None, capacity={"weight": 100.0},
                          home_depot_id="D0",
                          cost_per_second=0.005, cost_per_meter=0.001,
                          fixed_cost=20.0)]
    return T, D, depots, stops, fleet


def run(n: int, budget: float, seed: int = 0,
        construction: str = "fast") -> None:
    print(f"\n=== N={n}  budget={budget}s  seed={seed}  construction={construction} ===")
    t0 = time.perf_counter()
    T, D, depots, stops, fleet = build_instance(n, seed)
    print(f"  built instance in {time.perf_counter() - t0:.2f}s")
    t1 = time.perf_counter()
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=budget, seed=seed,
                                        construction=construction),
                   distance_matrix=D)
    wall = time.perf_counter() - t1
    print(f"  solve wall: {wall:.2f}s")
    print(f"  status={sol.status}  K={sol.vehicles_used}  "
          f"K_min={sol.vehicles_minimum_found}")
    print(f"  objective=${sol.objective_value:.2f}")
    served = sum(len(r.visits) - 2 for r in sol.routes)
    print(f"  served={served}/{n}  feasibility={sol.diagnostics.feasibility_blockers or 'OK'}")
    print(f"  construction_used={sol.construction_used}")
    print(f"  timings={sol.diagnostics.timings}")
    quality = sol.quality_report.solution_level
    print(f"  quality:")
    for k, v in quality.items():
        print(f"    {k:30s} {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of customers")
    ap.add_argument("--budget", type=float, default=60.0, help="wall budget seconds")
    ap.add_argument("--seed", type=int, default=0, help="rng seed")
    ap.add_argument("--construction", choices=["auto", "fast", "pyvrp"],
                    default="fast",
                    help="solver path: 'fast'=native (sub-second at N=1000), "
                         "'pyvrp'=svrptw bandit (slower, higher quality), "
                         "'auto'=pyvrp if installed else fast")
    ap.add_argument("--full", action="store_true",
                    help="run a sweep N=100,500,1000")
    args = ap.parse_args()
    if args.full:
        for n in (100, 500, 1000):
            run(n, args.budget, args.seed, args.construction)
    else:
        run(args.n, args.budget, args.seed, args.construction)


if __name__ == "__main__":
    main()
