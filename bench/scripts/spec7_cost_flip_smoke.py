"""SPEC-7-COST-01 validation smoke.

For each of 10 Manhattan N=50 instances, run the 4 reference solvers
twice — once with zero-coef (legacy) cost and once with the SPEC-7
defaults — and compare rankings.

Predicts: LKH-3's relative position improves (it builds fewer fuller
routes, so the underutil penalty hurts it the least).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from svrptw.config import Settings
from svrptw.config.schema import Economics, Settings as SettingsT
from svrptw.io import load_instance
from svrptw.solvers.classical import (
    auction_gart as auction_gart_mod,
    greedy as greedy_mod,
    lkh3 as lkh3_mod,
    portfolio as portfolio_mod,
)
from svrptw.solvers.common.solution import evaluate


def _solve(solver: str, inst, s: SettingsT):
    if solver == "greedy":
        return greedy_mod.solve(inst, s)
    if solver == "auction_gart":
        return auction_gart_mod.solve(inst, s)
    if solver == "lkh3@1":
        return lkh3_mod.solve(inst, s, budget_seconds=1.0)
    if solver == "portfolio@10":
        return portfolio_mod.solve(inst, s, budget_seconds=10.0)
    raise ValueError(solver)


def main() -> int:
    instances = sorted(Path("instances/v1").glob("OSM-Manhattan-N050-I*.json"))[:10]
    solvers = ["greedy", "auction_gart", "lkh3@1", "portfolio@10"]

    s_legacy = Settings()  # zero coefs by default
    s_new = Settings()
    s_new.economics = Economics(
        wage_per_hour=14.50, cost_per_mile=0.50, hard_late_penalty=1000.0,
        underutil_penalty_per_route=40.0,
        underutil_exponent=2.0,
        underutil_target_util=0.70,
        symmetry_penalty_coef=80.0,
    )

    rows = []
    for ip in instances:
        inst = load_instance(str(ip))
        for solver in solvers:
            t0 = time.perf_counter()
            sol = _solve(solver, inst, s_legacy)
            dt = time.perf_counter() - t0
            # Re-evaluate the same solution under the new cost model.
            new_metrics = evaluate(inst, sol, s_new)
            rows.append({
                "instance_id": inst.instance_id,
                "solver": solver,
                "cost_legacy": sol.metrics["operational_cost"],
                "cost_new": new_metrics["operational_cost"],
                "penalty_delta": new_metrics["operational_cost"] - sol.metrics["operational_cost"],
                "num_vehicles": int(sol.num_vehicles_used),
                "wall_s": dt,
            })
            print(f"  {inst.instance_id} {solver:<14} "
                  f"legacy={sol.metrics['operational_cost']:7.1f} "
                  f"new={new_metrics['operational_cost']:7.1f} "
                  f"d={new_metrics['operational_cost'] - sol.metrics['operational_cost']:+7.1f} "
                  f"vh={int(sol.num_vehicles_used)} ({dt:.1f}s)")

    out_path = Path("bench/runs/spec7_cost_flip_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_path} ({len(rows)} rows)")

    # Summary by solver.
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "legacy": 0.0, "new": 0.0, "delta": 0.0, "vh": 0})
    for r in rows:
        a = agg[r["solver"]]
        a["n"] += 1
        a["legacy"] += r["cost_legacy"]
        a["new"] += r["cost_new"]
        a["delta"] += r["penalty_delta"]
        a["vh"] += r["num_vehicles"]
    print(f"\n{'solver':<14} {'mean legacy':>12} {'mean new':>10} {'mean d':>9} {'mean veh':>9}")
    for solver, a in sorted(agg.items(), key=lambda x: x[1]["legacy"] / max(x[1]["n"], 1)):
        n = max(a["n"], 1)
        print(f"{solver:<14} {a['legacy']/n:>12.1f} {a['new']/n:>10.1f} "
              f"{a['delta']/n:>+9.1f} {a['vh']/n:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
