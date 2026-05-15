"""v1 N=500 with tuned portfolio_pyvrp_warm defaults.

Previous N=500 result (old defaults cb=5, plateaus=6): warm wins
5/8 vs vanilla (+$75), warm wins 7/8 vs pyvrp@120 (+$151).
With tuned defaults (cb=8s, plateaus=20), do these numbers improve?

Bound: 5 N=500 instances x 3 variants (pyvrp@120, vanilla@60, warm@60).
Budget per solve: ~60-120s. Total ~25-30 min.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, portfolio_pyvrp_warm as ppw, \
    pyvrp_solver as pv


def main() -> int:
    paths = sorted(Path("instances/v1").glob("OSM-Manhattan-N500-I*.json"))[:5]
    print(f"v1 N=500 with tuned defaults: {len(paths)} instances")
    rows = []
    for ip in paths:
        inst = load_instance(str(ip))
        s = Settings()
        pyvrp120 = pv.solve(inst, s, budget_seconds=120.0)
        port60   = pm.solve(inst, s, budget_seconds=60.0)
        warm60   = ppw.solve(inst, s, budget_seconds=60.0)  # uses tuned defaults
        rows.append({
            "instance_id": inst.instance_id,
            "pyvrp120": pyvrp120.metrics["operational_cost"],
            "port60":   port60.metrics["operational_cost"],
            "warm60":   warm60.metrics["operational_cost"],
        })
        sys.stdout.write(f"  {inst.instance_id:<32}  pyvrp120={rows[-1]['pyvrp120']:.1f}  "
                         f"port60={rows[-1]['port60']:.1f}  warm60={rows[-1]['warm60']:.1f}\n")
        sys.stdout.flush()

    Path("bench/runs/v1_n500_tuned_warmstart.json").write_text(json.dumps(rows, indent=2))

    print("\n[summary] head-to-head wins:")
    n = len(rows)
    for label, fn in [
        ("warm60 vs pyvrp120", lambda r: r["pyvrp120"] - r["warm60"]),
        ("warm60 vs port60 vanilla", lambda r: r["port60"] - r["warm60"]),
        ("port60 vanilla vs pyvrp120", lambda r: r["pyvrp120"] - r["port60"]),
    ]:
        deltas = [fn(r) for r in rows]
        w = sum(1 for d in deltas if d > 1.0)
        l = sum(1 for d in deltas if d < -1.0)
        mean = sum(deltas) / n
        print(f"  {label:<35}: W{w}/L{l} mean={mean:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
