"""Diagnostic: does PyVRP-warmstart help portfolio at N=200/$0?

Hypothesis: at large N, auction_gart's construction is worse than
PyVRP's. Portfolio's bandit then starts from a worse local basin
and can't climb out in 30s. Use the initial_solution= hook to seed
the bandit from PyVRP@5s (just construction phase) and see if the
final cost matches or exceeds vanilla portfolio.
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv


def main() -> int:
    paths = sorted(Path("instances/v2").glob("OSM-*-N200-I000.json"))[:6]
    print(f"PyVRP-warmstart experiment on {len(paths)} N=200 instances")
    rows = []
    for ip in paths:
        inst = load_instance(str(ip))
        s = Settings()
        # Reference: PyVRP@60s alone (the baseline portfolio loses to at $0)
        pyvrp60 = pv.solve(inst, s, budget_seconds=60.0)
        # Baseline: vanilla portfolio@30 (auction warmstart)
        port30 = pm.solve(inst, s, budget_seconds=30.0)
        # Experiment: PyVRP@5 warmstart + portfolio bandit for 25s remaining
        warm = pv.solve(inst, s, budget_seconds=5.0)
        portWarm = pm.solve(inst, s, budget_seconds=25.0, initial_solution=warm)

        rows.append({
            "instance_id": inst.instance_id,
            "pyvrp60": pyvrp60.metrics["operational_cost"],
            "port30_vanilla": port30.metrics["operational_cost"],
            "port30_pyvrp_warm": portWarm.metrics["operational_cost"],
        })
        print(f"  {inst.instance_id:<32} pyvrp60={rows[-1]['pyvrp60']:.1f} "
              f"port30={rows[-1]['port30_vanilla']:.1f} "
              f"port_pyvrp_warm={rows[-1]['port30_pyvrp_warm']:.1f}")

    print("\n[summary]")
    vs_pyvrp_vanilla = [r["pyvrp60"] - r["port30_vanilla"] for r in rows]
    vs_pyvrp_warm    = [r["pyvrp60"] - r["port30_pyvrp_warm"] for r in rows]
    print(f"  mean (pyvrp60 - port30_vanilla):       {sum(vs_pyvrp_vanilla)/len(rows):+.2f}")
    print(f"  mean (pyvrp60 - port30_pyvrp_warm):    {sum(vs_pyvrp_warm)/len(rows):+.2f}")
    print(f"  mean (port30_vanilla - port_pyvrp_warm): "
          f"{sum((r['port30_vanilla'] - r['port30_pyvrp_warm']) for r in rows) / len(rows):+.2f}")

    Path("bench/runs/portfolio_pyvrp_warmstart_n200.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
