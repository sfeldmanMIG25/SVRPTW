"""v2 OOD: same held-out protocol on v2 (looser capacity regime).

v1 N=200 OOD: 16/16 wins. Does it survive at v2's capacity_buffer 2.5?
Tests architectural-claim robustness across instance regimes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as ppw, pyvrp_solver as pv


def main() -> int:
    # v2 has up to I=002 per city × N=200 (3 per_pair). All instances are
    # actually held out from cb-tuning (which used v1 only). Use all 24.
    paths = sorted(Path("instances/v2").glob("OSM-*-N200-I*.json"))
    print(f"v2 N=200 OOD: {len(paths)} instances")
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
    Path("bench/runs/v2_n200_ood_warmstart.json").write_text(json.dumps(rows, indent=2))
    n = len(rows); deltas = [r["delta"] for r in rows]
    w = sum(1 for d in deltas if d > 1.0); l = sum(1 for d in deltas if d < -1.0)
    print(f"\n[v2 OOD] warm30 wins {w}/{n}, pyvrp wins {l}/{n}, mean delta {sum(deltas)/n:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
