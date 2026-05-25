"""iter-5y -- driver-time-variance fairness: paired 6-instance bench on v1_large.

Recipe Step 0 (iter-5x addition): cross-route operators (relocate / swap /
two_opt_star / merge_routes) exist in the bandit's arm set, so the
prerequisite IS satisfied for this term. Tests whether the recipe
generalizes to a third structural category (cross-route).

Setup:
  driver_time_variance_penalty_coef = 0.001   # dimensionless multiplier
  budget 75s at N=500, 150s at N=1000, paired-seed.

Usage: PYTHONPATH=. python bench/scripts/iter5y_fairness_v1large.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

OUT = "bench/runs/iter5y_fairness_v1large_bis.json"
LOG = "bench/runs/iter5y_fairness_v1large_bis.log"

INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]

COEF = 0.5  # iter-5y-bis: original 0.001 gave baseline penalty $0.10 = noise; raised to target ~$50/inst penalty at baseline so the bandit has a real gradient


def _run_one(task: tuple[str, str, float]) -> dict[str, Any]:
    inst_path, mode, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
    from svrptw.solvers.common.solution import evaluate

    inst = load_instance(inst_path)
    s_base = Settings()
    s_fair = deepcopy(s_base)
    s_fair.economics.driver_time_variance_penalty_coef = COEF

    used_settings = s_fair if mode == "fair" else s_base
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, used_settings, budget_seconds=budget, seed=0)
    wall = time.perf_counter() - t0

    cost_no_fair = evaluate(inst, sol, s_base)["operational_cost"]
    cost_w_fair = evaluate(inst, sol, s_fair)["operational_cost"]
    return {
        "instance_id": inst.instance_id,
        "mode": mode,
        "K": int(sol.metrics["num_vehicles_used"]),
        "cost_no_fair": cost_no_fair,
        "cost_w_fair": cost_w_fair,
        "variance_penalty": cost_w_fair - cost_no_fair,
        "wall_s": wall,
        "budget_s": budget,
    }


def main() -> int:
    tasks = []
    for inst_path, budget in INSTANCES:
        for mode in ("baseline", "fair"):
            tasks.append((inst_path, mode, budget))

    print(f"=== iter-5y: {len(tasks)} solves on {len(INSTANCES)} v1_large instances ===")
    print(f"  driver_time_variance_penalty_coef = {COEF}")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                r = f.result()
                rows.append(r)
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['mode']:9s} on "
                      f"{r['instance_id']:32s} K={r['K']:>3d} "
                      f"ops=${r['cost_no_fair']:>9.1f} "
                      f"var_pen=${r['variance_penalty']:>9.1f} "
                      f"wall={r['wall_s']:>5.1f}s")
            except Exception as e:
                print(f"  FAIL: {e}")

    print()
    print("=== paired comparison ===")
    by_inst: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_inst.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_winners = 0
    total_savings = 0.0
    for iid, m in sorted(by_inst.items()):
        if "baseline" not in m or "fair" not in m:
            continue
        b, s = m["baseline"], m["fair"]
        net = b["cost_w_fair"] - s["cost_w_fair"]
        ops_delta = s["cost_no_fair"] - b["cost_no_fair"]
        var_delta = s["variance_penalty"] - b["variance_penalty"]
        print(f"  {iid:32s} baseline K={b['K']:>3d} ops=${b['cost_no_fair']:>8.1f} "
              f"var_pen=${b['variance_penalty']:>8.1f}")
        print(f"  {iid:32s} fair     K={s['K']:>3d} ops=${s['cost_no_fair']:>8.1f} "
              f"var_pen=${s['variance_penalty']:>8.1f}  "
              f"net={net:>+8.1f}  ops_d={ops_delta:>+7.1f}  var_d={var_delta:>+7.1f}")
        if net > 0:
            n_winners += 1
        total_savings += net
        print()

    n_inst = len(by_inst)
    print(f"=== verdict ===")
    print(f"  fair wins under fairness-aware obj: {n_winners}/{n_inst}")
    if n_inst:
        print(f"  mean net saving per instance: ${total_savings/n_inst:+.1f}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "coef": COEF, "rows": rows,
        "n_winners": n_winners, "n_instances": n_inst,
        "total_savings": total_savings,
        "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
