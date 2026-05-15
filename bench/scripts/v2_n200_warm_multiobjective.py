"""v2 N=200 multi-objective leaderboard using portfolio_pyvrp_warm.

Original v2 N=200 leaderboard (with vanilla portfolio):
  $0:   port 6/24,  pyvrp 18/24  (PyVRP wins)
  $25:  port 10/24, pyvrp 14/24
  $100: port 19/24, pyvrp 5/24

With the warmstart fix shipped this session, re-run to see the
combined effect: PyVRP-construction warmstart + bandit + multi-
objective advantage at the same 30s budget vs PyVRP@60.
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Economics, Settings, TimeBudget
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as ppw, pyvrp_solver as pv


def _s(fixed): return Settings(economics=Economics(per_route_fixed_cost=fixed), time=TimeBudget())


def main() -> int:
    paths = sorted(Path("instances/v2").glob("OSM-*-N200-I*.json"))
    print(f"v2 N=200 multi-objective with warmstart: {len(paths)} instances x 3 cost levels")
    rows = []
    for fixed in (0.0, 25.0, 100.0):
        s = _s(fixed)
        for ip in paths:
            inst = load_instance(str(ip))
            pyvrp60 = pv.solve(inst, s, budget_seconds=60.0)
            warm30  = ppw.solve(inst, s, budget_seconds=30.0)
            rows.append({
                "fixed_cost": fixed, "instance_id": inst.instance_id,
                "pyvrp60": pyvrp60.metrics["operational_cost"],
                "warm30":  warm30.metrics["operational_cost"],
                "delta":   pyvrp60.metrics["operational_cost"] - warm30.metrics["operational_cost"],
            })
            print(f"  ${fixed:>5.0f}  {inst.instance_id:<32}  "
                  f"pyvrp60={rows[-1]['pyvrp60']:.1f}  "
                  f"warm30={rows[-1]['warm30']:.1f}  "
                  f"delta={rows[-1]['delta']:+.1f}")

    print("\n[summary] portfolio_pyvrp_warm@30 vs pyvrp@60 (v2 N=200):")
    print(f"  {'fixed':>5} {'warm wins':>10} {'pyvrp wins':>11} {'mean delta':>12}")
    by = {(r["instance_id"], r["fixed_cost"]): r for r in rows}
    iids = sorted({r["instance_id"] for r in rows})
    for fixed in (0.0, 25.0, 100.0):
        w = l = 0
        deltas = []
        for iid in iids:
            r = by.get((iid, fixed))
            if r is None: continue
            deltas.append(r["delta"])
            if r["delta"] > 0.01: w += 1
            elif r["delta"] < -0.01: l += 1
        print(f"  ${fixed:>4.0f} {w:>10} {l:>11} {sum(deltas)/max(1,len(deltas)):>+12.2f}")
    Path("bench/runs/v2_n200_warm_multiobjective.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
