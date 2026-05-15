"""Fusion vs portfolio-alone — small-N to medium-N.

POMO literature claims neural construction wins more at larger N. Test
that on v1 N=50/100/200 with budget=10s each.
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import fusion as fm
from svrptw.solvers.classical import portfolio as pm


def main() -> int:
    settings = Settings()
    rows: list[dict] = []
    for N in (50, 100, 200):
        ips = sorted(Path("instances/v1").glob(f"OSM-Manhattan-N{N:03d}-I*.json"))[:3]
        for ip in ips:
            inst = load_instance(str(ip))
            pf = pm.solve(inst, settings, budget_seconds=10.0)
            fu = fm.solve(inst, settings, budget_seconds=10.0, K_pomo=8, M_topstarts=3)
            rows.append({
                "instance_id": inst.instance_id, "N": N,
                "portfolio_cost": pf.metrics["operational_cost"],
                "fusion_cost":    fu.metrics["operational_cost"],
                "delta":          pf.metrics["operational_cost"] - fu.metrics["operational_cost"],
                "portfolio_wall": pf.wall_clock_seconds,
                "fusion_wall":    fu.wall_clock_seconds,
            })
            print(f"  {inst.instance_id:<32} port={pf.metrics['operational_cost']:.1f} "
                  f"fusion={fu.metrics['operational_cost']:.1f} delta={rows[-1]['delta']:+.2f}")

    print("\n[summary] mean delta (portfolio - fusion; positive = fusion wins):")
    print(f"  {'N':>5} {'n':>3} {'delta':>10}")
    for N in (50, 100, 200):
        ds = [r["delta"] for r in rows if r["N"] == N]
        if ds:
            print(f"  {N:>5} {len(ds):>3} {sum(ds)/len(ds):>+10.2f}")
    Path("bench/runs/v1_fusion_vs_portfolio.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
