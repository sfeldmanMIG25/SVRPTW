"""Run portfolio_pyvrp_warm vs PyVRP on the 56 Solomon N=100 benchmark instances.

Uses the parallel-bench helper per PARALLEL_GUIDANCE.md mandate.
Each worker loads its own instance and runs both solvers; main thread
reduces results and reports head-to-head + per-class breakdown.

Published Solomon optima are in distance units (Solomon's speed=1
convention). Our cost = wage_per_min * total_dist + cost_per_mile *
total_dist = total_dist * (wage/60 + cost_per_mile). For default
Settings: 0.7417 * total_dist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.parallel import map_instances


# Solomon classes (C/R/RC × 1/2). 1xx = short scheduling horizon (tight TW),
# 2xx = long horizon (loose TW). The 1xx series is the harder benchmark.
INSTANCES = sorted(Path("instances/solomon").glob("*.txt"))


def solve_one(args):
    """Worker: solve one Solomon instance with both solvers, return result row."""
    import time
    from svrptw.config import Settings
    from svrptw.io.solomon import load_solomon
    from svrptw.solvers.classical import portfolio_pyvrp_warm as ppw, pyvrp_solver as pv
    path = args
    inst = load_solomon(path)
    s = Settings()
    t0 = time.perf_counter()
    warm = ppw.solve(inst, s, budget_seconds=15.0)
    warm_wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    pyvrp = pv.solve(inst, s, budget_seconds=30.0)
    pyvrp_wall = time.perf_counter() - t0
    return {
        "name": inst.instance_id.replace("Solomon-", ""),
        "warm_cost":  warm.metrics["operational_cost"],
        "warm_dist":  warm.metrics["total_distance_miles"],
        "warm_routes": int(warm.metrics["num_vehicles_used"]),
        "warm_feasible": bool(warm.metrics["feasible"]),
        "warm_wall_s": warm_wall,
        "pyvrp_cost":  pyvrp.metrics["operational_cost"],
        "pyvrp_dist":  pyvrp.metrics["total_distance_miles"],
        "pyvrp_routes": int(pyvrp.metrics["num_vehicles_used"]),
        "pyvrp_feasible": bool(pyvrp.metrics["feasible"]),
        "pyvrp_wall_s": pyvrp_wall,
        "delta": pyvrp.metrics["operational_cost"] - warm.metrics["operational_cost"],
    }


def main() -> int:
    print(f"Solomon benchmark: {len(INSTANCES)} N=100 instances, "
          f"warm@15s vs PyVRP@30s, max_workers=4")
    rows = map_instances(solve_one, INSTANCES, max_workers=4)

    # Write JSON FIRST (PARALLEL_GUIDANCE.md rule).
    Path("bench/runs/solomon_benchmark.json").write_text(json.dumps(rows, indent=2))

    print("\n[per-instance]")
    for r in sorted(rows, key=lambda x: x.get("name", "")):
        if "_failed" in r and r["_failed"]:
            sys.stdout.write(f"  {r.get('task','?')}: FAILED {r.get('_error','')}\n")
            continue
        sys.stdout.write(
            f"  {r['name']:<8} warm={r['warm_dist']:>7.1f}/{r['warm_routes']:>2}r  "
            f"pyvrp={r['pyvrp_dist']:>7.1f}/{r['pyvrp_routes']:>2}r  "
            f"Δcost={r['delta']:+8.2f}\n"
        )

    valid = [r for r in rows if "name" in r]
    n = len(valid)
    deltas = [r["delta"] for r in valid]
    wins = sum(1 for d in deltas if d > 0.01)
    losses = sum(1 for d in deltas if d < -0.01)
    print(f"\n[summary] warm@15 vs PyVRP@30 on Solomon N=100:")
    print(f"  n={n}  warm wins={wins}  PyVRP wins={losses}  ties={n-wins-losses}")
    print(f"  mean delta = {sum(deltas)/n:+.2f}")
    # Per-class breakdown.
    by_class = {}
    for r in valid:
        cls = "".join(c for c in r["name"][:3] if not c.isdigit())  # C, R, RC
        by_class.setdefault(cls, []).append(r["delta"])
    print("\n[per-class]")
    for cls in sorted(by_class):
        ds = by_class[cls]
        w = sum(1 for d in ds if d > 0.01)
        print(f"  {cls:<4} n={len(ds)} W={w} mean={sum(ds)/len(ds):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
