"""v2 multi-objective leaderboard: portfolio vs pyvrp vs lkh3 vs auction
under per_route_fixed_cost ∈ {$0, $25, $100}.

Hypothesis (from earlier v2 smoke): portfolio's bandit responds to
per_route_fixed_cost (drops 6.25 → 6.00 routes at $100), but PyVRP
ignores it (flat at 7.75 routes). So at $100/route on v2:
  portfolio_cost = transit + wages + 6.00 × $100 = transit + $600
  pyvrp_cost     = transit + wages + 7.75 × $100 = transit + $775
Even if PyVRP's transit is slightly lower, $175 extra fixed cost
swings the leaderboard. This is the structural-advantage finding.

Bench: 40 v2 instances × 3 fixed-cost levels × 4 solvers = 480 solves.
Bound by pyvrp@30 (40 × 3 × 30 = 60 min). Total ~75 min in background.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from svrptw.config import Economics, Settings, TimeBudget
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv, \
    auction_gart as ag


def _settings_with(fixed: float) -> Settings:
    return Settings(
        economics=Economics(per_route_fixed_cost=fixed),
        time=TimeBudget(),
    )


def main() -> int:
    paths = sorted(Path("instances/v2").glob("*.json"))
    paths = [p for p in paths if "manifest" not in p.name]
    print(f"v2 leaderboard on {len(paths)} instances × 3 cost-levels × 3 solvers")
    rows: list[dict] = []
    for fixed in (0.0, 25.0, 100.0):
        s = _settings_with(fixed)
        for ip in paths:
            inst = load_instance(str(ip))
            t0 = time.perf_counter()
            sols = {
                "portfolio@10": pm.solve(inst, s, budget_seconds=10.0),
                "pyvrp@30":     pv.solve(inst, s, budget_seconds=30.0),
                "auction":      ag.solve(inst, s),
            }
            for name, sol in sols.items():
                rows.append({
                    "fixed_cost": fixed,
                    "instance_id": inst.instance_id,
                    "solver": name,
                    "operational_cost": sol.metrics["operational_cost"],
                    "num_vehicles_used": int(sol.metrics["num_vehicles_used"]),
                    "feasible": bool(sol.metrics.get("feasible", True)),
                    "capacity_overload": float(sol.metrics.get("capacity_overload", 0.0)),
                })
            print(f"  fixed=${fixed:>5.0f}  {inst.instance_id:<32}  "
                  f"port={sols['portfolio@10'].metrics['operational_cost']:.1f}/"
                  f"{sols['portfolio@10'].num_vehicles_used}r  "
                  f"pyvrp={sols['pyvrp@30'].metrics['operational_cost']:.1f}/"
                  f"{sols['pyvrp@30'].num_vehicles_used}r  "
                  f"auct={sols['auction'].metrics['operational_cost']:.1f}/"
                  f"{sols['auction'].num_vehicles_used}r")

    # Aggregate.
    print("\n[summary] portfolio vs pyvrp head-to-head wins on operational_cost:")
    print(f"  {'fixed':>5}  {'port wins':>10} {'pyvrp wins':>10} {'mean port-pyvrp':>16}")
    by = {(r["instance_id"], r["fixed_cost"], r["solver"]): r["operational_cost"] for r in rows}
    iids = sorted({r["instance_id"] for r in rows})
    for fixed in (0.0, 25.0, 100.0):
        port_wins = pyvrp_wins = 0
        deltas = []
        for iid in iids:
            p_cost = by.get((iid, fixed, "portfolio@10"))
            v_cost = by.get((iid, fixed, "pyvrp@30"))
            if p_cost is None or v_cost is None:
                continue
            deltas.append(v_cost - p_cost)
            if p_cost < v_cost - 0.01: port_wins += 1
            elif v_cost < p_cost - 0.01: pyvrp_wins += 1
        mean = sum(deltas) / max(1, len(deltas))
        print(f"  ${fixed:>4.0f}  {port_wins:>10} {pyvrp_wins:>10} {mean:>+16.2f}")

    Path("bench/runs/v2_multiobjective_leaderboard.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote bench/runs/v2_multiobjective_leaderboard.json ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
