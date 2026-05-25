"""Quick post-hoc analysis of wholesale_v1large_full9.json.

Fires the moment the wholesale comparison drops its JSON. Computes:
- per-solver mean cost / quality / wall on v1_large
- ranked leaderboard sorted by mean unified score
- per-instance solve_auto vs each public solver delta
- verdict: does solve_auto win cost vs OR-Tools and LKH-3 in particular?

Usage:
  PYTHONPATH=. python bench/scripts/summarize_wholesale_full9.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_PATH = "bench/runs/wholesale_v1large_full9.json"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not Path(path).exists():
        print(f"NOT_READY {path} does not exist yet")
        return 1
    rows = json.loads(Path(path).read_text())
    rows = [r for r in rows if not r.get("_failed")]
    if not rows:
        print(f"EMPTY {path} has no successful rows")
        return 1

    by_solver: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_solver[r["solver"]].append(r)

    # Per-solver leaderboard
    print("=== wholesale leaderboard (mean across all instances) ===")
    print(f"  {'solver':22s} {'n':>3s} {'cost':>10s} {'q_idx':>7s} "
          f"{'cross':>6s} {'wall_p50':>9s} {'feasible':>9s}")
    leaderboard = []
    for solver, rs in by_solver.items():
        n = len(rs)
        mc = sum(r["operational_cost"] for r in rs) / n
        mq = sum(r["quality_index"] for r in rs) / n
        mx = sum(r.get("inter_route_crossings", 0) for r in rs) / n
        walls = sorted(r["wall_clock_s"] for r in rs)
        p50 = walls[len(walls) // 2]
        feas = sum(1 for r in rs if r.get("feasible", True)) / n
        leaderboard.append((solver, n, mc, mq, mx, p50, feas))
    leaderboard.sort(key=lambda x: x[2])
    for solver, n, mc, mq, mx, p50, feas in leaderboard:
        print(f"  {solver:22s} {n:>3d} {mc:>10.1f} {mq:>7.3f} "
              f"{mx:>6.0f} {p50:>9.1f}s {feas:>9.1%}")

    # Per-instance head-to-head: solve_auto vs each public solver
    if "solve_auto" in by_solver:
        print("\n=== solve_auto vs each public solver, per-instance cost delta ===")
        sa_by_inst = {r["instance_id"]: r for r in by_solver["solve_auto"]}
        public = [s for s in ["pyvrp", "ortools", "lkh3"] if s in by_solver]
        for s in public:
            print(f"\n  vs {s}:")
            print(f"    {'instance':32s} {'solve_auto':>10s} {s:>10s} {'delta':>9s}")
            other = {r["instance_id"]: r for r in by_solver[s]}
            n_win = 0
            n_tot = 0
            d_sum = 0.0
            for iid in sorted(sa_by_inst):
                if iid not in other:
                    continue
                a = sa_by_inst[iid]["operational_cost"]
                b = other[iid]["operational_cost"]
                d = b - a  # positive => solve_auto wins (cheaper)
                marker = " WIN" if d > 0 else (" LOSS" if d < 0 else " TIE")
                print(f"    {iid:32s} {a:>10.1f} {b:>10.1f} {d:>+9.1f}{marker}")
                d_sum += d
                n_tot += 1
                if d > 0: n_win += 1
            if n_tot:
                print(f"    -- solve_auto vs {s}: {n_win}/{n_tot} wins, mean delta=${d_sum/n_tot:+.1f}/inst")

    # Pareto: any solver dominate solve_auto on BOTH cost and quality?
    if "solve_auto" in by_solver:
        sa_by_inst = {r["instance_id"]: r for r in by_solver["solve_auto"]}
        print("\n=== Pareto check: any solver dominate solve_auto on BOTH axes? ===")
        for s, rs in by_solver.items():
            if s == "solve_auto":
                continue
            other = {r["instance_id"]: r for r in rs}
            wins_both = 0
            n = 0
            for iid in sa_by_inst:
                if iid not in other:
                    continue
                n += 1
                if (other[iid]["operational_cost"] < sa_by_inst[iid]["operational_cost"]
                    and other[iid]["quality_index"] > sa_by_inst[iid]["quality_index"]):
                    wins_both += 1
            print(f"  {s:22s}: dominates_both {wins_both}/{n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
