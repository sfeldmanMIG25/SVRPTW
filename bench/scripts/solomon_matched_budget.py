"""Solomon matched-budget rematch: warm@30 vs PyVRP@30 (was warm@15 vs PyVRP@30).

Previous Solomon result: 29/56 (52%) with warm at half PyVRP's budget.
At matched budget (both 30s), does warm convincingly win, or does
PyVRP catch up?
"""
from __future__ import annotations

import json
from pathlib import Path

from bench.parallel import map_instances


def solve_one(args):
    from svrptw.config import Settings
    from svrptw.io.solomon import load_solomon
    from svrptw.solvers.classical import portfolio_pyvrp_warm as ppw, pyvrp_solver as pv
    path = args
    inst = load_solomon(path)
    s = Settings()
    warm = ppw.solve(inst, s, budget_seconds=30.0)  # matched budget
    pyvrp = pv.solve(inst, s, budget_seconds=30.0)
    return {
        "name": inst.instance_id.replace("Solomon-", ""),
        "warm_cost":  warm.metrics["operational_cost"],
        "warm_dist":  warm.metrics["total_distance_miles"],
        "warm_routes": int(warm.metrics["num_vehicles_used"]),
        "pyvrp_cost": pyvrp.metrics["operational_cost"],
        "pyvrp_dist": pyvrp.metrics["total_distance_miles"],
        "pyvrp_routes": int(pyvrp.metrics["num_vehicles_used"]),
        "delta": pyvrp.metrics["operational_cost"] - warm.metrics["operational_cost"],
    }


def main() -> int:
    paths = sorted(Path("instances/solomon").glob("*.txt"))
    print(f"Solomon matched-budget rematch: {len(paths)} instances, both at 30s, 4-way parallel")
    rows = map_instances(solve_one, paths, max_workers=4)
    Path("bench/runs/solomon_matched_budget.json").write_text(json.dumps(rows, indent=2))

    valid = [r for r in rows if "delta" in r]
    n = len(valid)
    deltas = [r["delta"] for r in valid]
    wins = sum(1 for d in deltas if d > 0.01)
    losses = sum(1 for d in deltas if d < -0.01)
    print(f"\n[matched 30s] warm wins {wins}/{n}, pyvrp wins {losses}/{n}, mean delta {sum(deltas)/n:+.2f}")
    by = {}
    for r in valid:
        cls = "".join(c for c in r["name"][:3] if not c.isdigit())
        by.setdefault(cls, []).append(r["delta"])
    for cls in sorted(by):
        ds = by[cls]
        w = sum(1 for d in ds if d > 0.01)
        print(f"  {cls:<4} n={len(ds)} W={w} mean={sum(ds)/len(ds):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
