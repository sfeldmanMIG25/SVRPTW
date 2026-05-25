"""iter-5w -- shift_overrun term: paired 6-instance validation on v1_large.

Single-instance smoke (iter-5v) showed +$109/inst net saving under
shift-aware objective. This bench validates across all 6 v1_large
instances (Manhattan/Paris/SF x N=500/1000), paired seed for fair
comparison.

For each instance:
  baseline:  solve_auto(default Settings)  -- term off
  shifted:   solve_auto(shift_max=300, pen=$1/min)  -- term on
Each solution cross-evaluated under BOTH cost models.

Workers=4 parallel. 12 solves at scaled budgets (75s/150s) takes ~6-10 min.

Usage: PYTHONPATH=. python bench/scripts/iter5w_shift_overrun_v1large.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

OUT = "bench/runs/iter5w_shift_overrun_v1large.json"
LOG = "bench/runs/iter5w_shift_overrun_v1large.log"

INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]

CAP_MIN = 300.0
PEN_DOLLARS = 1.0


def _run_one(task: tuple[str, str, float]) -> dict[str, Any]:
    inst_path, mode, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
    from svrptw.solvers.common.solution import evaluate

    inst = load_instance(inst_path)
    s_base = Settings()
    s_over = deepcopy(s_base)
    s_over.economics.shift_max_minutes = CAP_MIN
    s_over.economics.shift_overrun_penalty_per_min = PEN_DOLLARS

    used_settings = s_over if mode == "shifted" else s_base
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, used_settings, budget_seconds=budget, seed=0)
    wall = time.perf_counter() - t0

    cost_no_shift = evaluate(inst, sol, s_base)["operational_cost"]
    cost_w_shift = evaluate(inst, sol, s_over)["operational_cost"]
    return {
        "instance_id": inst.instance_id,
        "mode": mode,
        "K": int(sol.metrics["num_vehicles_used"]),
        "cost_no_shift": cost_no_shift,
        "cost_w_shift": cost_w_shift,
        "overrun_dollars": cost_w_shift - cost_no_shift,
        "wall_s": wall,
        "budget_s": budget,
    }


def main() -> int:
    tasks = []
    for inst_path, budget in INSTANCES:
        for mode in ("baseline", "shifted"):
            tasks.append((inst_path, mode, budget))

    print(f"=== iter-5w: {len(tasks)} solves on {len(INSTANCES)} v1_large instances ===")
    print(f"  shift cap={CAP_MIN} min  penalty=${PEN_DOLLARS}/min")
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
                      f"ops=${r['cost_no_shift']:>9.1f} "
                      f"w_shift=${r['cost_w_shift']:>9.1f} "
                      f"wall={r['wall_s']:>5.1f}s")
            except Exception as e:
                print(f"  FAIL: {e}")

    # Pair by instance, compute deltas
    print()
    print("=== paired comparison (baseline vs shifted, per instance) ===")
    print(f"  {'instance':32s}  {'mode':>10s}  {'K':>3s}  "
          f"{'ops_cost':>9s}  {'cost_w_shift':>13s}")
    by_inst: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_inst.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_winners = 0
    total_savings = 0.0
    for iid, m in sorted(by_inst.items()):
        if "baseline" not in m or "shifted" not in m:
            continue
        b, s = m["baseline"], m["shifted"]
        print(f"  {iid:32s}  {'baseline':>10s}  {b['K']:>3d}  "
              f"{b['cost_no_shift']:>9.1f}  {b['cost_w_shift']:>13.1f}")
        print(f"  {iid:32s}  {'shifted':>10s}  {s['K']:>3d}  "
              f"{s['cost_no_shift']:>9.1f}  {s['cost_w_shift']:>13.1f}")
        net = b["cost_w_shift"] - s["cost_w_shift"]
        ops_cost = s["cost_no_shift"] - b["cost_no_shift"]
        if net > 0:
            n_winners += 1
            verdict = "  shifted WINS"
        else:
            verdict = "  shifted LOSES"
        total_savings += net
        print(f"  {'':32s}  {'NET':>10s}  {s['K']-b['K']:>+3d}  "
              f"{ops_cost:>+9.1f}  {net:>+13.1f}{verdict}")
        print()

    n_inst = len(by_inst)
    print(f"=== verdict ===")
    print(f"  shifted wins net (under shift-aware obj): {n_winners}/{n_inst}")
    if n_inst:
        print(f"  mean net saving per instance: ${total_savings/n_inst:+.1f}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "cap_min": CAP_MIN, "pen_per_min": PEN_DOLLARS,
        "rows": rows,
        "n_winners": n_winners, "n_instances": n_inst,
        "total_savings": total_savings,
        "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
