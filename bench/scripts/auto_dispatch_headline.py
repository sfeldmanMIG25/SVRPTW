"""Final headline: solve_auto vs benchmarks across all 4 N's.

Auto-dispatch picks the right variant for each scale:
  N<100: vanilla portfolio
  100<=N<300: warm (PyVRP+bandit fusion)
  N>=300: vanilla portfolio

Bench at each scale's natural budget vs PyVRP at 2x budget.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import pyvrp_solver as pv
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto

BUDGETS = {50: 10.0, 100: 15.0, 200: 30.0, 500: 60.0}
PYVRP_BUDGETS = {50: 30.0, 100: 45.0, 200: 60.0, 500: 120.0}


def main() -> int:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix",
              "Charleston", "Austin", "Pittsburgh", "Cambridge"]
    rows = []
    for N in (50, 100, 200, 500):
        b = BUDGETS[N]; pb = PYVRP_BUDGETS[N]
        for c in cities:
            ip = Path(f"instances/v1/OSM-{c}-N{N:03d}-I000.json")
            if not ip.exists(): continue
            inst = load_instance(str(ip))
            s = Settings()
            auto_sol  = solve_auto(inst, s, budget_seconds=b)
            pyvrp_sol = pv.solve(inst, s, budget_seconds=pb)
            rows.append({
                "N": N, "instance_id": inst.instance_id,
                "auto_budget": b, "pyvrp_budget": pb,
                "auto_dispatched_to": auto_sol.solver,
                "auto_cost": auto_sol.metrics["operational_cost"],
                "pyvrp_cost": pyvrp_sol.metrics["operational_cost"],
                "delta": pyvrp_sol.metrics["operational_cost"] - auto_sol.metrics["operational_cost"],
            })
            sys.stdout.write(f"  N={N:>3} {inst.instance_id:<32}  "
                             f"auto={rows[-1]['auto_cost']:.1f} ({auto_sol.solver})  "
                             f"pyvrp={rows[-1]['pyvrp_cost']:.1f}  "
                             f"delta={rows[-1]['delta']:+.1f}\n")
            sys.stdout.flush()

    Path("bench/runs/auto_dispatch_headline.json").write_text(json.dumps(rows, indent=2))

    print("\n[headline] solve_auto vs PyVRP@2x-budget:")
    print(f"  {'N':>3} {'auto_b':>8} {'pyvrp_b':>9} {'wins':>5} {'losses':>7} {'mean delta':>12}")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows: by[r["N"]].append(r)
    for N in (50, 100, 200, 500):
        rs = by.get(N, [])
        if not rs: continue
        n = len(rs)
        deltas = [r["delta"] for r in rs]
        wins = sum(1 for d in deltas if d > 1.0)
        losses = sum(1 for d in deltas if d < -1.0)
        mean = sum(deltas) / n
        print(f"  {N:>3} {BUDGETS[N]:>8.0f} {PYVRP_BUDGETS[N]:>9.0f} {wins:>5} {losses:>7} {mean:>+12.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
