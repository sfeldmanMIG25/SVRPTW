"""OOD multi-scale validation: held-out instances at N=100 and N=500.

N=200 OOD already validated (16/16 wins, +$112 mean). Now confirm
the architectural claim holds at N=100 and N=500 with the same
held-out protocol: I=003/I=004 across 8 cities.

For each N, solve_auto picks the right variant per the refined recipe:
  N=100: portfolio_pyvrp_warm (100 ≤ N < 300)
  N=500: vanilla portfolio (N ≥ 300, bandit-sufficient)

Bench at each scale's natural budget vs PyVRP at 2x.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import pyvrp_solver as pv
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto

BUDGETS = {100: 15.0, 500: 60.0}
PYVRP_BUDGETS = {100: 45.0, 500: 120.0}


def main() -> int:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix",
              "Charleston", "Austin", "Pittsburgh", "Cambridge"]
    reps = ["I003", "I004"]
    rows = []
    for N in (100, 500):
        b = BUDGETS[N]; pb = PYVRP_BUDGETS[N]
        for c in cities:
            for r in reps:
                ip = Path(f"instances/v1/OSM-{c}-N{N:03d}-{r}.json")
                if not ip.exists(): continue
                inst = load_instance(str(ip))
                s = Settings()
                auto_sol  = solve_auto(inst, s, budget_seconds=b)
                pyvrp_sol = pv.solve(inst, s, budget_seconds=pb)
                rows.append({
                    "N": N, "instance_id": inst.instance_id,
                    "auto_dispatched_to": auto_sol.solver,
                    "auto_cost":  auto_sol.metrics["operational_cost"],
                    "pyvrp_cost": pyvrp_sol.metrics["operational_cost"],
                    "delta": pyvrp_sol.metrics["operational_cost"] - auto_sol.metrics["operational_cost"],
                })
                sys.stdout.write(f"  N={N:>3} {inst.instance_id:<32}  "
                                 f"auto={rows[-1]['auto_cost']:.1f}  "
                                 f"pyvrp={rows[-1]['pyvrp_cost']:.1f}  "
                                 f"delta={rows[-1]['delta']:+.1f}\n")
                sys.stdout.flush()

    Path("bench/runs/v1_ood_multiscale.json").write_text(json.dumps(rows, indent=2))

    print("\n[OOD headline by scale]")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows: by[r["N"]].append(r)
    for N in sorted(by):
        rs = by[N]; n = len(rs)
        deltas = [r["delta"] for r in rs]
        w = sum(1 for d in deltas if d > 1.0)
        l = sum(1 for d in deltas if d < -1.0)
        disp = ", ".join(set(r["auto_dispatched_to"] for r in rs))
        print(f"  N={N:>3} n={n}  W={w} L={l}  mean={sum(deltas)/n:+.2f}  "
              f"dispatched_to={disp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
