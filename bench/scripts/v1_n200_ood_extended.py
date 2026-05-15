"""Extended v1 N=200 OOD: I=001/I=002 across 8 cities (16 more held-out instances).

Combined with the existing I=003/I=004 16-instance OOD bench, this brings
N=200 OOD coverage to 32 instances — much stronger statistical power.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as ppw, pyvrp_solver as pv


def main() -> int:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix",
              "Charleston", "Austin", "Pittsburgh", "Cambridge"]
    reps = ["I001", "I002"]
    paths = []
    for c in cities:
        for r in reps:
            ip = Path(f"instances/v1/OSM-{c}-N200-{r}.json")
            if ip.exists(): paths.append(ip)

    print(f"v1 N=200 OOD extended (I001/I002): {len(paths)} instances")
    rows = []
    for ip in paths:
        inst = load_instance(str(ip))
        s = Settings()
        pyvrp60 = pv.solve(inst, s, budget_seconds=60.0)
        warm30  = ppw.solve(inst, s, budget_seconds=30.0)
        rows.append({
            "instance_id": inst.instance_id,
            "pyvrp60_cost": pyvrp60.metrics["operational_cost"],
            "warm30_cost":  warm30.metrics["operational_cost"],
            "delta": pyvrp60.metrics["operational_cost"] - warm30.metrics["operational_cost"],
        })
        sys.stdout.write(f"  {inst.instance_id:<32} pyvrp60={rows[-1]['pyvrp60_cost']:.1f} "
                         f"warm30={rows[-1]['warm30_cost']:.1f} delta={rows[-1]['delta']:+.1f}\n")
        sys.stdout.flush()
    Path("bench/runs/v1_n200_ood_extended.json").write_text(json.dumps(rows, indent=2))
    n = len(rows); deltas = [r["delta"] for r in rows]
    w = sum(1 for d in deltas if d > 1.0); l = sum(1 for d in deltas if d < -1.0)
    print(f"\n[v1 N=200 OOD extended] warm wins {w}/{n}, pyvrp {l}/{n}, mean {sum(deltas)/n:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
