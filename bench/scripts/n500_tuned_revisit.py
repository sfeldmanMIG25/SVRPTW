"""Re-test N=500 warm-vs-vanilla with TUNED defaults (cb=8, plateaus=20).

The earlier N=500 test used cb=5/plateaus=6 (old defaults) and showed
warm vs vanilla essentially tied. With cb=8/plateaus=20 (current
tuned defaults), does warm now beat vanilla at N=500, OR does the
N-recipe's 'vanilla at N>=300' verdict still hold?

8 v1 N=500 instances (1 per city), parallel-bench helper.
"""
from __future__ import annotations

import json
from pathlib import Path

from bench.parallel import map_instances


def solve_one(args):
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio as pm, portfolio_pyvrp_warm as ppw, \
        pyvrp_solver as pv
    ip = args
    inst = load_instance(str(ip))
    s = Settings()
    pyvrp120 = pv.solve(inst, s, budget_seconds=120.0)
    port60   = pm.solve(inst, s, budget_seconds=60.0)
    warm60   = ppw.solve(inst, s, budget_seconds=60.0)  # uses tuned defaults
    return {
        "instance_id": inst.instance_id,
        "pyvrp120": pyvrp120.metrics["operational_cost"],
        "port60":   port60.metrics["operational_cost"],
        "warm60":   warm60.metrics["operational_cost"],
        "warm_vs_vanilla_delta": port60.metrics["operational_cost"] - warm60.metrics["operational_cost"],
        "warm_vs_pyvrp_delta":   pyvrp120.metrics["operational_cost"] - warm60.metrics["operational_cost"],
    }


def main() -> int:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix",
              "Charleston", "Austin", "Pittsburgh", "Cambridge"]
    paths = [Path(f"instances/v1/OSM-{c}-N500-I000.json") for c in cities]
    paths = [p for p in paths if p.exists()]
    print(f"N=500 tuned-defaults revisit: {len(paths)} instances, 4-way parallel")
    rows = map_instances(solve_one, paths, max_workers=4)
    Path("bench/runs/n500_tuned_revisit.json").write_text(json.dumps(rows, indent=2))

    n = len(rows)
    d_vv = [r["warm_vs_vanilla_delta"] for r in rows if "warm_vs_vanilla_delta" in r]
    d_vp = [r["warm_vs_pyvrp_delta"] for r in rows if "warm_vs_pyvrp_delta" in r]
    if d_vv:
        w_vv = sum(1 for d in d_vv if d > 1.0); l_vv = sum(1 for d in d_vv if d < -1.0)
        print(f"\n[summary @N=500 tuned]")
        print(f"  warm vs vanilla:  W={w_vv}/L={l_vv}  mean delta = {sum(d_vv)/len(d_vv):+.2f}")
        w_vp = sum(1 for d in d_vp if d > 1.0); l_vp = sum(1 for d in d_vp if d < -1.0)
        print(f"  warm vs pyvrp120: W={w_vp}/L={l_vp}  mean delta = {sum(d_vp)/len(d_vp):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
