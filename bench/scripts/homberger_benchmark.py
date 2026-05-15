"""Homberger extended-Solomon benchmark: portfolio_pyvrp_warm vs PyVRP.

Scale selector via SIZE env var: SIZE=200 or SIZE=400.

Bench: warm@30 vs PyVRP@60 at N=200; warm@60 vs PyVRP@120 at N=400.
4-way parallel via bench.parallel.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from bench.parallel import map_instances


SIZE = int(os.environ.get("SIZE", "200"))
WARM_BUDGET = {200: 30.0, 400: 60.0}[SIZE]
PYVRP_BUDGET = {200: 60.0, 400: 120.0}[SIZE]


def solve_one(args):
    from svrptw.config import Settings
    from svrptw.io.solomon import load_solomon
    from svrptw.solvers.classical import portfolio_pyvrp_warm as ppw, pyvrp_solver as pv
    path = args
    inst = load_solomon(path)
    s = Settings()
    warm = ppw.solve(inst, s, budget_seconds=WARM_BUDGET)
    pyvrp = pv.solve(inst, s, budget_seconds=PYVRP_BUDGET)
    return {
        "name": inst.instance_id.replace("Solomon-", ""),
        "warm_cost": warm.metrics["operational_cost"],
        "warm_dist": warm.metrics["total_distance_miles"],
        "warm_routes": int(warm.metrics["num_vehicles_used"]),
        "pyvrp_cost": pyvrp.metrics["operational_cost"],
        "pyvrp_dist": pyvrp.metrics["total_distance_miles"],
        "pyvrp_routes": int(pyvrp.metrics["num_vehicles_used"]),
        "delta": pyvrp.metrics["operational_cost"] - warm.metrics["operational_cost"],
    }


def main() -> int:
    paths = sorted(Path(f"instances/homberger/{SIZE}").glob("*.txt"))
    # N=400 needs lower max_workers due to memory pressure (each worker
    # holds a 400×400 matrix + PyVRP+Numba state; 4× footprint OOMs the laptop).
    workers = 2 if SIZE >= 400 else 4
    print(f"Homberger N={SIZE}: {len(paths)} instances, warm@{WARM_BUDGET}s vs PyVRP@{PYVRP_BUDGET}s, {workers}-way parallel")
    rows = map_instances(solve_one, paths, max_workers=workers)
    Path(f"bench/runs/homberger_{SIZE}.json").write_text(json.dumps(rows, indent=2))

    valid = [r for r in rows if "delta" in r]
    n = len(valid)
    deltas = [r["delta"] for r in valid]
    wins = sum(1 for d in deltas if d > 0.01)
    losses = sum(1 for d in deltas if d < -0.01)
    print(f"\n[Homberger N={SIZE}] warm wins {wins}/{n}, pyvrp {losses}/{n}, mean delta {sum(deltas)/n:+.2f}")
    by = {}
    for r in valid:
        cls = "".join(c for c in r["name"][:3] if not c.isdigit() and c != "_")
        by.setdefault(cls, []).append(r["delta"])
    for cls in sorted(by):
        ds = by[cls]
        w = sum(1 for d in ds if d > 0.01)
        print(f"  {cls:<4} n={len(ds)} W={w} mean={sum(ds)/len(ds):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
