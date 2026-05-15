"""SPEC-7-COST-02 smoke: does per-route fixed cost push solvers to use fewer routes?

Run portfolio, pyvrp, auction_gart on 5 N=50 Manhattan instances at three
fixed-cost levels: 0, 25, 100. Record route count, distance, total cost.
A successful intervention should drop route count monotonically with
fixed-cost without exploding distance.

If route count drops without much distance penalty, the operators
already had route-reducing moves available — they just weren't motivated.
If route count flatlines, we need new route-merge operators (e.g. the
route-pair-zip seed from the council spec).
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Economics, Settings, TimeBudget
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv, auction_gart as ag


def _settings_with(fixed: float) -> Settings:
    return Settings(
        economics=Economics(per_route_fixed_cost=fixed),
        time=TimeBudget(),
    )


def main() -> int:
    paths = sorted(Path("instances/v1").glob("OSM-Manhattan-N050-I*.json"))[:5]
    rows: list[dict] = []
    for fixed in (0.0, 25.0, 100.0):
        s = _settings_with(fixed)
        for ip in paths:
            inst = load_instance(str(ip))
            sols = {
                "portfolio@8": pm.solve(inst, s, budget_seconds=8.0),
                "pyvrp@30":    pv.solve(inst, s, budget_seconds=30.0),
                "auction":     ag.solve(inst, s),
            }
            for name, sol in sols.items():
                rows.append({
                    "fixed_cost": fixed,
                    "instance_id": inst.instance_id,
                    "solver": name,
                    "operational_cost": sol.metrics["operational_cost"],
                    "num_vehicles_used": int(sol.metrics["num_vehicles_used"]),
                    "total_distance_miles": sol.metrics["total_distance_miles"],
                    "feasible": bool(sol.metrics.get("feasible", True)),
                    "capacity_overload": float(sol.metrics.get("capacity_overload", 0.0)),
                })
            print(f"  fixed={fixed:>5.1f}  {inst.instance_id}  "
                  f"port={sols['portfolio@8'].num_vehicles_used}  "
                  f"pyvrp={sols['pyvrp@30'].num_vehicles_used}  "
                  f"auct={sols['auction'].num_vehicles_used}")

    out = Path("bench/runs/per_route_fixed_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    # Aggregate per solver per fixed level: mean route count, mean dist.
    print("\n[summary] mean route count by (solver, fixed_cost):")
    print(f"  {'solver':<13} {'fixed=0':>8} {'fixed=25':>9} {'fixed=100':>10}")
    by = {}
    for r in rows:
        by.setdefault((r["solver"], r["fixed_cost"]), []).append(r["num_vehicles_used"])
    for solver in ("portfolio@8", "pyvrp@30", "auction"):
        vals = [sum(by.get((solver, f), [0])) / max(1, len(by.get((solver, f), [0])))
                for f in (0.0, 25.0, 100.0)]
        print(f"  {solver:<13} {vals[0]:>8.2f} {vals[1]:>9.2f} {vals[2]:>10.2f}")

    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
