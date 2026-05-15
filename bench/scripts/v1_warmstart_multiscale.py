"""Multi-scale characterization of portfolio_pyvrp_warm on v1.

For N in {50, 100, 200, 500}: 8 instances (one per city × 1 rep) at
budgets {N=50: 10s, N=100: 15s, N=200: 30s, N=500: 60s} where the
warm variant matches that total budget with PyVRP construction +
portfolio bandit refinement.

The full scaling table is the publishable artifact: at which scales
does the warmstart fix dominate, and what's the gain?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, portfolio_pyvrp_warm as ppw, \
    pyvrp_solver as pv


BUDGETS = {50: 10.0, 100: 15.0, 200: 30.0, 500: 60.0}
PYVRP_BASELINE_BUDGET = {50: 30.0, 100: 45.0, 200: 60.0, 500: 120.0}


def main() -> int:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix",
              "Charleston", "Austin", "Pittsburgh", "Cambridge"]
    rows = []
    for N in (50, 100, 200, 500):
        budget = BUDGETS[N]
        pyvrp_budget = PYVRP_BASELINE_BUDGET[N]
        for c in cities:
            ip = Path(f"instances/v1/OSM-{c}-N{N:03d}-I000.json")
            if not ip.exists():
                continue
            inst = load_instance(str(ip))
            s = Settings()
            pyvrp_ref = pv.solve(inst, s, budget_seconds=pyvrp_budget)
            port_v    = pm.solve(inst, s, budget_seconds=budget)
            warm_v    = ppw.solve(inst, s, budget_seconds=budget)
            rows.append({
                "N": N, "instance_id": inst.instance_id,
                "budget_seconds": budget,
                "pyvrp_baseline_budget": pyvrp_budget,
                "pyvrp_baseline_cost": pyvrp_ref.metrics["operational_cost"],
                "port_vanilla_cost":    port_v.metrics["operational_cost"],
                "port_warm_cost":       warm_v.metrics["operational_cost"],
            })
            sys.stdout.write(
                f"  N={N:>3} {inst.instance_id:<32}  "
                f"pyvrp{int(pyvrp_budget)}={rows[-1]['pyvrp_baseline_cost']:.1f}  "
                f"port{int(budget)}={rows[-1]['port_vanilla_cost']:.1f}  "
                f"warm{int(budget)}={rows[-1]['port_warm_cost']:.1f}\n"
            )
            sys.stdout.flush()

    out = Path("bench/runs/v1_warmstart_multiscale.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")

    print("\n[summary] warm vs pyvrp_baseline (positive = warm beats pyvrp):")
    print(f"  {'N':>3} {'inst':>4} {'warm wins':>10} {'pyvrp wins':>11} {'mean delta':>+12}")
    for N in (50, 100, 200, 500):
        rs = [r for r in rows if r["N"] == N]
        n = len(rs)
        if n == 0: continue
        deltas = [r["pyvrp_baseline_cost"] - r["port_warm_cost"] for r in rs]
        w = sum(1 for d in deltas if d > 1.0)
        l = sum(1 for d in deltas if d < -1.0)
        print(f"  {N:>3} {n:>4} {w:>10} {l:>11} {sum(deltas)/n:>+12.2f}")

    print("\n[summary] warm vs port_vanilla (positive = warm beats vanilla, same budget):")
    print(f"  {'N':>3} {'inst':>4} {'warm wins':>10} {'vanilla wins':>13} {'mean delta':>+12}")
    for N in (50, 100, 200, 500):
        rs = [r for r in rows if r["N"] == N]
        n = len(rs)
        if n == 0: continue
        deltas = [r["port_vanilla_cost"] - r["port_warm_cost"] for r in rs]
        w = sum(1 for d in deltas if d > 1.0)
        l = sum(1 for d in deltas if d < -1.0)
        print(f"  {N:>3} {n:>4} {w:>10} {l:>13} {sum(deltas)/n:>+12.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
