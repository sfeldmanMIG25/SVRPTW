"""Re-run v2 N=200 multi-objective leaderboard with portfolio_pyvrp_warm.

Portfolio with PyVRP@5s warmstart + bandit@25s beat vanilla
portfolio@30s by $117/instance and beat PyVRP@60s by $35/instance on
the diagnostic. This bench validates that the new variant flips the
N=200/$0 leaderboard from loss (PyVRP wins 18/24) to win.

All 24 v2 N=200 instances x 3 cost levels.
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Economics, Settings, TimeBudget
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv


def _s(fixed): return Settings(economics=Economics(per_route_fixed_cost=fixed), time=TimeBudget())


def main() -> int:
    paths = sorted(Path("instances/v2").glob("OSM-*-N200-I*.json"))
    print(f"v2 N=200 leaderboard with PyVRP-warmstart: {len(paths)} instances x 3 cost levels")
    rows = []
    for fixed in (0.0, 25.0, 100.0):
        s = _s(fixed)
        for ip in paths:
            inst = load_instance(str(ip))
            # Reference: PyVRP@60 alone
            pyvrp60 = pv.solve(inst, s, budget_seconds=60.0)
            # Fused: PyVRP@5 warmstart + portfolio bandit @25s = 30s total
            warm = pv.solve(inst, s, budget_seconds=5.0)
            port_warm = pm.solve(inst, s, budget_seconds=25.0, initial_solution=warm)
            rows.append({
                "fixed_cost": fixed, "instance_id": inst.instance_id,
                "pyvrp60_cost": pyvrp60.metrics["operational_cost"],
                "port_pyvrp_warm_cost": port_warm.metrics["operational_cost"],
                "delta": pyvrp60.metrics["operational_cost"] - port_warm.metrics["operational_cost"],
            })
            print(f"  ${fixed:>5.0f}  {inst.instance_id:<32}  "
                  f"pyvrp60={rows[-1]['pyvrp60_cost']:.1f}  "
                  f"port_warm={rows[-1]['port_pyvrp_warm_cost']:.1f}  "
                  f"delta={rows[-1]['delta']:+.1f}")

    print("\n[summary] port_pyvrp_warm vs pyvrp60 (N=200):")
    print(f"  {'fixed':>5} {'port_warm wins':>15} {'pyvrp wins':>11} {'mean delta':>12}")
    by = {(r["instance_id"], r["fixed_cost"]): r for r in rows}
    iids = sorted({r["instance_id"] for r in rows})
    for fixed in (0.0, 25.0, 100.0):
        pw = pvw = 0
        deltas = []
        for iid in iids:
            r = by.get((iid, fixed))
            if r is None: continue
            deltas.append(r["delta"])
            if r["delta"] > 0.01: pw += 1
            elif r["delta"] < -0.01: pvw += 1
        mean = sum(deltas) / max(1, len(deltas))
        print(f"  ${fixed:>4.0f} {pw:>15} {pvw:>11} {mean:>+12.2f}")

    Path("bench/runs/v2_n200_pyvrp_warmstart_leaderboard.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
