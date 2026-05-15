"""v2 per-route fixed-cost smoke. Does route count drop when capacity isn't binding?

v1 result: route count flat at 11 across $0/$25/$100 (capacity floor binds).
v2 expectation: solvers should drop to 6-7 routes when $50+/route is charged,
since the capacity floor is 6.
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Economics, Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv, auction_gart as ag


def main() -> int:
    paths = sorted(Path("instances/v2").glob("OSM-*-N050-I000.json"))  # 1 per city
    rows = []
    for fixed in (0.0, 25.0, 100.0):
        s = Settings(economics=Economics(per_route_fixed_cost=fixed))
        for ip in paths:
            inst = load_instance(str(ip))
            sols = {
                "portfolio@8": pm.solve(inst, s, budget_seconds=8.0),
                "pyvrp@30":    pv.solve(inst, s, budget_seconds=30.0),
                "auction":     ag.solve(inst, s),
            }
            for name, sol in sols.items():
                rows.append({
                    "fixed_cost": fixed, "instance_id": inst.instance_id,
                    "solver": name,
                    "n_routes": int(sol.metrics["num_vehicles_used"]),
                    "cost": sol.metrics["operational_cost"],
                    "feasible": bool(sol.metrics.get("feasible", True)),
                })

    print(f"\n[summary] mean route count by (solver, fixed_cost):")
    print(f"  {'solver':<13} {'fixed=0':>8} {'fixed=25':>9} {'fixed=100':>10}")
    by = {}
    for r in rows:
        by.setdefault((r["solver"], r["fixed_cost"]), []).append(r["n_routes"])
    for solver in ("portfolio@8", "pyvrp@30", "auction"):
        vals = [sum(by.get((solver, f), [0])) / max(1, len(by.get((solver, f), [0])))
                for f in (0.0, 25.0, 100.0)]
        print(f"  {solver:<13} {vals[0]:>8.2f} {vals[1]:>9.2f} {vals[2]:>10.2f}")

    Path("bench/runs/v2_per_route_fixed_smoke.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
