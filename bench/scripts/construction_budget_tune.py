"""Tune PyVRP construction budget for portfolio_pyvrp_warm.

Currently fixed at 5s of PyVRP construction + 25s of portfolio bandit
on a 30s total budget. Test {3s, 5s, 8s, 10s} construction splits to
find the optimal allocation.

Hypothesis: longer PyVRP construction = better starting point but less
bandit-refinement time. The 5s default may not be the sweet spot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as ppw


def main() -> int:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix"]
    paths = [Path(f"instances/v1/OSM-{c}-N200-I000.json") for c in cities]
    splits = (3.0, 5.0, 8.0, 10.0)
    rows = []
    for ip in paths:
        if not ip.exists(): continue
        inst = load_instance(str(ip))
        s = Settings()
        sys.stdout.write(f"\n{inst.instance_id}:\n")
        for cb in splits:
            sol = ppw.solve(inst, s, budget_seconds=30.0,
                             pyvrp_construction_budget=cb)
            rows.append({
                "instance_id": inst.instance_id,
                "construction_budget": cb,
                "cost": sol.metrics["operational_cost"],
                "wall": sol.wall_clock_seconds,
            })
            sys.stdout.write(f"  cb={cb:>4.1f}s  cost={rows[-1]['cost']:.1f}  wall={rows[-1]['wall']:.1f}s\n")
            sys.stdout.flush()

    # Write data FIRST.
    Path("bench/runs/construction_budget_tune.json").write_text(json.dumps(rows, indent=2))

    print("\n[summary] mean cost by construction budget:")
    print(f"  {'cb':>5} {'mean_cost':>10} {'mean_wall':>10}")
    by = {}
    for r in rows:
        by.setdefault(r["construction_budget"], []).append(r)
    for cb in splits:
        rs = by.get(cb, [])
        if not rs: continue
        mc = sum(r["cost"] for r in rs) / len(rs)
        mw = sum(r["wall"] for r in rs) / len(rs)
        print(f"  {cb:>5.1f} {mc:>10.1f} {mw:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
