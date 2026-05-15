"""OOD validation: portfolio_pyvrp_warm vs PyVRP@60 on instances NEVER seen in tuning.

Held-out: I=003, I=004 across all 8 cities (the tuning used I=000 only,
and construction_budget_tune used only 4 of the 8 cities). 16 truly
held-out N=200 instances. If warm still wins 75%+ here, the
architectural claim survives selection-bias scrutiny.
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
    reps = ["I003", "I004"]   # NEVER seen in any tuning step
    paths = []
    for c in cities:
        for r in reps:
            ip = Path(f"instances/v1/OSM-{c}-N200-{r}.json")
            if ip.exists():
                paths.append(ip)
    print(f"OOD held-out bench: {len(paths)} N=200 instances (I003/I004, 8 cities)")

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
            "delta":        pyvrp60.metrics["operational_cost"] - warm30.metrics["operational_cost"],
        })
        sys.stdout.write(f"  {inst.instance_id:<32}  pyvrp60={rows[-1]['pyvrp60_cost']:.1f}  "
                         f"warm30={rows[-1]['warm30_cost']:.1f}  delta={rows[-1]['delta']:+.1f}\n")
        sys.stdout.flush()

    # Write JSON FIRST (lesson from earlier bugs).
    Path("bench/runs/v1_n200_ood_warmstart.json").write_text(json.dumps(rows, indent=2))

    print("\n[OOD headline]")
    n = len(rows)
    deltas = [r["delta"] for r in rows]
    wins = sum(1 for d in deltas if d > 1.0)
    losses = sum(1 for d in deltas if d < -1.0)
    ties = n - wins - losses
    mean = sum(deltas) / n
    print(f"  warm30 wins: {wins}/{n} ({wins/n*100:.1f}%)")
    print(f"  pyvrp wins:  {losses}/{n} ({losses/n*100:.1f}%)")
    print(f"  ties:        {ties}/{n}")
    print(f"  mean delta:  {mean:+.2f}")

    # Per-city breakdown to detect any one-city dominance.
    from collections import defaultdict
    by_city = defaultdict(list)
    for r in rows:
        city = r["instance_id"].split("-")[1]
        by_city[city].append(r["delta"])
    print("\n[per-city mean delta]")
    for city in sorted(by_city):
        ds = by_city[city]
        w = sum(1 for d in ds if d > 1.0)
        print(f"  {city:<15} n={len(ds)}  W={w}  mean={sum(ds)/len(ds):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
