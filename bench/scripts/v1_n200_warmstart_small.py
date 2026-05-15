"""Smaller v1 N=200 warmstart bench: 8 instances (one per city), capture stdout robustly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, portfolio_pyvrp_warm as ppw, pyvrp_solver as pv


def main() -> int:
    # One instance per city.
    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix",
              "Charleston", "Austin", "Pittsburgh", "Cambridge"]
    paths = [Path(f"instances/v1/OSM-{c}-N200-I000.json") for c in cities]
    rows = []
    for ip in paths:
        if not ip.exists():
            continue
        inst = load_instance(str(ip))
        s = Settings()
        pyvrp60 = pv.solve(inst, s, budget_seconds=60.0)
        port30  = pm.solve(inst, s, budget_seconds=30.0)
        warm30  = ppw.solve(inst, s, budget_seconds=30.0)
        rows.append({
            "instance_id": inst.instance_id,
            "pyvrp60": pyvrp60.metrics["operational_cost"],
            "port30":  port30.metrics["operational_cost"],
            "warm30":  warm30.metrics["operational_cost"],
        })
        sys.stdout.write(f"  {inst.instance_id:<32}  pyvrp60={rows[-1]['pyvrp60']:.1f}  "
                         f"port30={rows[-1]['port30']:.1f}  warm30={rows[-1]['warm30']:.1f}\n")
        sys.stdout.flush()

    out = Path("bench/runs/v1_n200_warmstart_small.json")
    out.write_text(json.dumps(rows, indent=2))
    sys.stdout.write(f"\nwrote {out}\n")
    sys.stdout.flush()

    print("\n[summary]")
    n = len(rows)
    pairs = [
        ("warm30 vs pyvrp60", lambda r: r["pyvrp60"] - r["warm30"]),
        ("warm30 vs port30 vanilla", lambda r: r["port30"] - r["warm30"]),
        ("port30 vanilla vs pyvrp60", lambda r: r["pyvrp60"] - r["port30"]),
    ]
    for label, fn in pairs:
        deltas = [fn(r) for r in rows]
        wins = sum(1 for d in deltas if d > 1.0)
        losses = sum(1 for d in deltas if d < -1.0)
        mean = sum(deltas) / max(1, n)
        print(f"  {label:<32}: wins={wins}/{n}  losses={losses}/{n}  mean={mean:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
