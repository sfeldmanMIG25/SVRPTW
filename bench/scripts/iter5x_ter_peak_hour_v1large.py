"""iter-5x -- peak-hour surcharge: paired 6-instance bench on v1_large.

Recipe gate #6/7 for the second cost term. If 6/6 net positive under the
peak-aware objective, the cost-model-expansion recipe is confirmed
reusable across per-route -> per-segment-of-time structural categories.

Setup:
  peak_window_starts = (480, 1020)   # 8am and 5pm
  peak_window_ends   = (600, 1140)   # 10am and 7pm
  peak_hour_wage_multiplier = 1.5    # 50% surcharge during rush
  budget 75s at N=500, 150s at N=1000, paired-seed.

Usage: PYTHONPATH=. python bench/scripts/iter5x_peak_hour_v1large.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

OUT = "bench/runs/iter5x_ter_peak_hour_b2x_v1large.json"
LOG = "bench/runs/iter5x_ter_peak_hour_b2x_v1large.log"

# iter-5x-ter: 2x budget per instance (vs prior iter-5x-bis) to test budget-starvation
# hypothesis on single-term peak_hour. iter-5w-bis shift_start operator already in arm set.
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 150.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 300.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 300.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 300.0),
]

PEAK_STARTS = (480, 1020)
PEAK_ENDS = (600, 1140)
PEAK_MULT = 1.5


def _run_one(task: tuple[str, str, float]) -> dict[str, Any]:
    inst_path, mode, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
    from svrptw.solvers.common.solution import evaluate

    inst = load_instance(inst_path)
    s_base = Settings()
    s_peak = deepcopy(s_base)
    s_peak.economics.peak_window_starts = PEAK_STARTS
    s_peak.economics.peak_window_ends = PEAK_ENDS
    s_peak.economics.peak_hour_wage_multiplier = PEAK_MULT

    used_settings = s_peak if mode == "peaked" else s_base
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, used_settings, budget_seconds=budget, seed=0)
    wall = time.perf_counter() - t0

    cost_no_peak = evaluate(inst, sol, s_base)["operational_cost"]
    cost_w_peak = evaluate(inst, sol, s_peak)["operational_cost"]
    return {
        "instance_id": inst.instance_id,
        "mode": mode,
        "K": int(sol.metrics["num_vehicles_used"]),
        "cost_no_peak": cost_no_peak,
        "cost_w_peak": cost_w_peak,
        "peak_surcharge_dollars": cost_w_peak - cost_no_peak,
        "wall_s": wall,
        "budget_s": budget,
    }


def main() -> int:
    tasks = []
    for inst_path, budget in INSTANCES:
        for mode in ("baseline", "peaked"):
            tasks.append((inst_path, mode, budget))

    print(f"=== iter-5x: {len(tasks)} solves on {len(INSTANCES)} v1_large instances ===")
    print(f"  peak windows: 8-10am, 5-7pm  mult={PEAK_MULT}")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                r = f.result()
                rows.append(r)
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['mode']:10s} on "
                      f"{r['instance_id']:32s} K={r['K']:>3d} "
                      f"ops=${r['cost_no_peak']:>9.1f} "
                      f"w_peak=${r['cost_w_peak']:>9.1f} "
                      f"wall={r['wall_s']:>5.1f}s")
            except Exception as e:
                print(f"  FAIL: {e}")

    print()
    print("=== paired comparison (baseline vs peaked, per instance) ===")
    by_inst: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_inst.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_winners = 0
    total_savings = 0.0
    for iid, m in sorted(by_inst.items()):
        if "baseline" not in m or "peaked" not in m:
            continue
        b, s = m["baseline"], m["peaked"]
        net = b["cost_w_peak"] - s["cost_w_peak"]
        ops_cost = s["cost_no_peak"] - b["cost_no_peak"]
        print(f"  {iid:32s} baseline K={b['K']:>3d} ops=${b['cost_no_peak']:>8.1f} "
              f"w_peak=${b['cost_w_peak']:>9.1f}")
        print(f"  {iid:32s} peaked   K={s['K']:>3d} ops=${s['cost_no_peak']:>8.1f} "
              f"w_peak=${s['cost_w_peak']:>9.1f}  net={net:>+8.1f}  ops_d={ops_cost:>+7.1f}")
        if net > 0:
            n_winners += 1
        total_savings += net
        print()

    n_inst = len(by_inst)
    print(f"=== verdict ===")
    print(f"  peaked wins under peak-aware obj: {n_winners}/{n_inst}")
    if n_inst:
        print(f"  mean net saving per instance: ${total_savings/n_inst:+.1f}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "peak_starts": list(PEAK_STARTS),
        "peak_ends": list(PEAK_ENDS),
        "peak_multiplier": PEAK_MULT,
        "rows": rows,
        "n_winners": n_winners, "n_instances": n_inst,
        "total_savings": total_savings,
        "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
