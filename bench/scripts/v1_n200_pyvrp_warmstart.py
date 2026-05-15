"""Validate portfolio_pyvrp_warm on v1 N=200 (the original benchmark).

v2 N=200 result: warmstart variant wins 87.5% of instances at $0/route.
Does the architectural win carry over to v1 (capacity_buffer 1.4, the
'extreme' setting from the original SVRPTW research codebase)?

8 cities x 5 reps = 40 instances. portfolio_warm vs pyvrp@60 head-to-head.
Same 30s total budget for the warm variant (5s pyvrp construction +
25s portfolio bandit).
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import pyvrp_solver as pv, portfolio as pm, \
    portfolio_pyvrp_warm as ppw


def main() -> int:
    paths = sorted(Path("instances/v1").glob("OSM-*-N200-I*.json"))
    print(f"v1 N=200 portfolio_pyvrp_warm validation: {len(paths)} instances")
    rows = []
    for ip in paths:
        inst = load_instance(str(ip))
        s = Settings()
        # Three variants compared on the same instance.
        pyvrp60 = pv.solve(inst, s, budget_seconds=60.0)
        port30  = pm.solve(inst, s, budget_seconds=30.0)
        warm30  = ppw.solve(inst, s, budget_seconds=30.0)
        rows.append({
            "instance_id": inst.instance_id,
            "pyvrp60": pyvrp60.metrics["operational_cost"],
            "port30":  port30.metrics["operational_cost"],
            "warm30":  warm30.metrics["operational_cost"],
        })
        print(f"  {inst.instance_id:<32}  pyvrp60={rows[-1]['pyvrp60']:.1f}  "
              f"port30={rows[-1]['port30']:.1f}  warm30={rows[-1]['warm30']:.1f}")

    print("\n[summary] head-to-head wins (positive delta = our variant beats baseline):")
    print(f"  {'comparison':<35} {'wins':>5} {'losses':>7} {'ties':>5} {'mean delta':>+12}")
    pairs = [
        ("portfolio_warm vs pyvrp60", lambda r: r["pyvrp60"] - r["warm30"]),
        ("portfolio_warm vs vanilla portfolio", lambda r: r["port30"] - r["warm30"]),
        ("vanilla portfolio vs pyvrp60", lambda r: r["pyvrp60"] - r["port30"]),
    ]
    n = len(rows)
    for label, fn in pairs:
        deltas = [fn(r) for r in rows]
        wins   = sum(1 for d in deltas if d > 1.0)
        losses = sum(1 for d in deltas if d < -1.0)
        ties   = n - wins - losses
        print(f"  {label:<35} {wins:>5} {losses:>7} {ties:>5} {sum(deltas)/n:>+12.2f}")

    Path("bench/runs/v1_n200_pyvrp_warmstart.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
