"""Sweep quality_weight on solve_auto(fast_construct_v2 warmstart).

Hypothesis: quality_weight=0 lets the bandit destroy fcv2's Louvain basin
(seen as q drop 0.794 -> 0.559 after refinement). Stronger quality_weight
should preserve the basin.

Sweep: qw in {0.0, 0.5, 1.0, 2.0, 5.0}. Score by both cost AND quality.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from svrptw.io import load_instance
from svrptw.config import Settings
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
from svrptw.metrics import score_solution
from webui import client as ui


def main() -> int:
    inst = load_instance("instances/v1_large/OSM-Manhattan-N0500-I000.json")
    print(f"qw sweep on {inst.instance_id} (75s budget each, fast_construct_v2 warmstart)")
    print(f"{'qw':>5s} {'cost':>10s} {'K':>4s} {'q_idx':>6s} {'cross':>6s} {'wall':>6s}")
    ui.push_agent("iter5q-qw-sweep", "running",
                   summary="quality_weight sweep on fcv2-warm: 0 / 0.5 / 1.0 / 2.0 / 5.0")
    rows = []
    for qw in (0.0, 0.5, 1.0, 2.0, 5.0):
        t0 = time.perf_counter()
        sol = solve_auto(inst, Settings(), budget_seconds=75,
                          construction="fast_construct_v2", seed=42,
                          shape_reward=(qw > 0),
                          shape_coefs={"qw": qw} if qw > 0 else None)
        # NOTE: solve_auto routes shape_reward through pm.solve;
        # quality_weight kwarg goes via shape_coefs key 'qw' (per Phase F sub-agent's API)
        wall = time.perf_counter() - t0
        score = score_solution(inst, sol)
        c = float(sol.metrics["operational_cost"])
        k = int(sol.metrics["num_vehicles_used"])
        print(f"{qw:>5.1f} {c:>10.1f} {k:>4d} {score.quality_index:>6.3f} "
              f"{score.inter_route_crossings:>6d} {wall:>5.1f}s")
        rows.append((qw, c, k, score.quality_index, score.inter_route_crossings, wall))
        ui.push_log(f"  qw={qw}: cost={c:.0f} q={score.quality_index:.3f} cross={score.inter_route_crossings}")
    ui.push_agent("iter5q-qw-sweep", "completed",
                   summary=f"5 qw values tested; see log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
