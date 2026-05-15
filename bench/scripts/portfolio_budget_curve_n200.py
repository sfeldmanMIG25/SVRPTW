"""Diagnostic: portfolio at N=200, $0/route, multiple budgets.

If portfolio loses to PyVRP at N=200/$0 (-$78 mean delta), the cause
is either (a) budget too small for the bandit to explore enough, or
(b) the bandit hits a plateau and stops improving. Test by running
portfolio at {15s, 30s, 60s, 120s} on the same 6 instances and
plotting the cost curve.

If 120s catches up to PyVRP@60s, the fix is budget (or bandit
parameters that let it explore longer). If 120s plateaus near 30s
cost, the fix is operator-pool gaps - the bandit is stuck.
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv


def main() -> int:
    # 6 representative v2 N=200 instances (one per city, ish).
    paths = sorted(Path("instances/v2").glob("OSM-*-N200-I000.json"))[:6]
    print(f"diagnostic: portfolio budget curve on {len(paths)} N=200 instances")
    rows = []
    for ip in paths:
        inst = load_instance(str(ip))
        s = Settings()
        pyvrp60 = pv.solve(inst, s, budget_seconds=60.0)
        baseline = pyvrp60.metrics["operational_cost"]
        print(f"\n  {inst.instance_id}: pyvrp@60={baseline:.1f}")
        for budget in (15, 30, 60, 120):
            sol = pm.solve(inst, s, budget_seconds=float(budget))
            cost = sol.metrics["operational_cost"]
            delta = baseline - cost
            print(f"    portfolio@{budget:>3}s = {cost:.1f}  delta_vs_pyvrp60 = {delta:+.1f}")
            rows.append({
                "instance_id": inst.instance_id,
                "budget_seconds": budget,
                "portfolio_cost": cost,
                "pyvrp60_cost": baseline,
                "delta": delta,
            })

    print("\n[summary] mean portfolio cost by budget (lower better):")
    print(f"  {'budget':>8} {'mean cost':>10} {'mean delta vs pyvrp@60':>24}")
    by_b = {}
    for r in rows:
        by_b.setdefault(r["budget_seconds"], []).append((r["portfolio_cost"], r["delta"]))
    for b in (15, 30, 60, 120):
        if b not in by_b: continue
        costs, deltas = zip(*by_b[b])
        print(f"  {b:>8} {sum(costs)/len(costs):>10.1f} {sum(deltas)/len(deltas):>+24.1f}")

    Path("bench/runs/portfolio_budget_curve_n200.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
