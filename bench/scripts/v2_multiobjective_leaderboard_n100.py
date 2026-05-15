"""v2 multi-objective leaderboard at N=100 - confirms the headline scales.

At N=50, portfolio swept PyVRP 40-0 at $100/route, 113-7 overall.
This bench checks whether the structural advantage persists at N=100
where wall-budget pressure rises and solver dynamics differ.
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Economics, Settings, TimeBudget
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv, \
    auction_gart as ag


def _s(fixed: float) -> Settings:
    return Settings(economics=Economics(per_route_fixed_cost=fixed), time=TimeBudget())


def main() -> int:
    paths = sorted(Path("instances/v2").glob("OSM-*-N100-I*.json"))
    print(f"v2 N=100 leaderboard on {len(paths)} instances × 3 cost-levels × 3 solvers")
    rows: list[dict] = []
    for fixed in (0.0, 25.0, 100.0):
        s = _s(fixed)
        for ip in paths:
            inst = load_instance(str(ip))
            sols = {
                "portfolio@15": pm.solve(inst, s, budget_seconds=15.0),
                "pyvrp@45":     pv.solve(inst, s, budget_seconds=45.0),
                "auction":      ag.solve(inst, s),
            }
            for name, sol in sols.items():
                rows.append({
                    "fixed_cost": fixed, "instance_id": inst.instance_id,
                    "solver": name,
                    "operational_cost": sol.metrics["operational_cost"],
                    "num_vehicles_used": int(sol.metrics["num_vehicles_used"]),
                    "feasible": bool(sol.metrics.get("feasible", True)),
                })
            print(f"  ${fixed:>5.0f}  {inst.instance_id:<32}  "
                  f"port={sols['portfolio@15'].metrics['operational_cost']:.1f}/"
                  f"{sols['portfolio@15'].num_vehicles_used}r  "
                  f"pyvrp={sols['pyvrp@45'].metrics['operational_cost']:.1f}/"
                  f"{sols['pyvrp@45'].num_vehicles_used}r")

    print("\n[summary] portfolio vs pyvrp wins on operational_cost (N=100):")
    print(f"  {'fixed':>5} {'port wins':>10} {'pyvrp wins':>10} {'mean port-pyvrp':>16}")
    by = {(r["instance_id"], r["fixed_cost"], r["solver"]): r["operational_cost"] for r in rows}
    iids = sorted({r["instance_id"] for r in rows})
    for fixed in (0.0, 25.0, 100.0):
        port_w = pv_w = 0
        deltas = []
        for iid in iids:
            p = by.get((iid, fixed, "portfolio@15"))
            v = by.get((iid, fixed, "pyvrp@45"))
            if p is None or v is None: continue
            deltas.append(v - p)
            if p < v - 0.01: port_w += 1
            elif v < p - 0.01: pv_w += 1
        print(f"  ${fixed:>4.0f} {port_w:>10} {pv_w:>10} {sum(deltas)/max(1,len(deltas)):>+16.2f}")
    Path("bench/runs/v2_multiobjective_leaderboard_n100.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
