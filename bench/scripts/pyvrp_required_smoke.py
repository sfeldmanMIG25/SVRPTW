"""PyVRP smoke after required=True fix.

Run 2 instances per city × 8 cities = 16 N=50 instances at 30s budget.
Report mean cost and whether any customer was dropped.
"""
from __future__ import annotations

import time
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import pyvrp_solver as pv


def main() -> int:
    pattern = "OSM-*-N050-I00[01].json"
    paths = sorted(Path("instances/v1").glob(pattern))
    s = Settings()
    rows = []
    for ip in paths:
        inst = load_instance(str(ip))
        t0 = time.perf_counter()
        sol = pv.solve(inst, s, budget_seconds=30.0)
        dt = time.perf_counter() - t0
        m = sol.metrics
        rows.append((inst.instance_id, m["operational_cost"], m["missed_deliveries"],
                     m.get("capacity_overload", 0), m["feasible"], sol.num_vehicles_used, dt))
        print(f"  {inst.instance_id:<32} cost={m['operational_cost']:8.1f} "
              f"miss={int(m['missed_deliveries']):2d} overload={int(m.get('capacity_overload',0))} "
              f"feas={int(m['feasible'])} veh={int(sol.num_vehicles_used):2d} ({dt:.1f}s)")
    n = max(len(rows), 1)
    mc = sum(r[1] for r in rows) / n
    mm = sum(r[2] for r in rows) / n
    mv = sum(r[5] for r in rows) / n
    print(f"\nmean cost={mc:.1f}  mean miss={mm:.2f}  mean veh={mv:.2f}  n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
