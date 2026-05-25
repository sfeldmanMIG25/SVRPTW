"""iter-6a-1 -- driver-breaks (EU 561 lite): paired 6-instance bench on v1_large.

Cap: 270 min driving per route (4.5h EU rule). Penalty: $2/min over.
Step 0 check: split_route + merge_routes operators exist in the bandit's
arm set -> the bandit CAN break overlong driving routes into smaller pieces.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

OUT = "bench/runs/iter6a1_driver_breaks_v1large_bis.json"
LOG = "bench/runs/iter6a1_driver_breaks_v1large_bis.log"

INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]

DRIVING_CAP_MIN = 90.0  # iter-6a-1-bis: 270min cap was inert (baseline already compliant); 90min forces baseline violation -> tests bandit responsiveness
PEN_PER_MIN = 2.0


def _run_one(task: tuple[str, str, float]) -> dict[str, Any]:
    inst_path, mode, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(inst_path)
    s_base = Settings()
    s_brk = deepcopy(s_base)
    s_brk.economics.driving_max_minutes = DRIVING_CAP_MIN
    s_brk.economics.break_violation_penalty_per_min = PEN_PER_MIN
    used = s_brk if mode == "breaks_aware" else s_base
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, used, budget_seconds=budget, seed=0)
    wall = time.perf_counter() - t0
    cost_no = evaluate(inst, sol, s_base)["operational_cost"]
    cost_w = evaluate(inst, sol, s_brk)["operational_cost"]
    return {
        "instance_id": inst.instance_id, "mode": mode,
        "K": int(sol.metrics["num_vehicles_used"]),
        "cost_no_breaks": cost_no, "cost_w_breaks": cost_w,
        "break_penalty": cost_w - cost_no,
        "wall_s": wall, "budget_s": budget,
    }


def main() -> int:
    tasks = []
    for p, b in INSTANCES:
        for mode in ("baseline", "breaks_aware"):
            tasks.append((p, mode, b))
    print(f"=== iter-6a-1: {len(tasks)} solves on {len(INSTANCES)} v1_large; cap={DRIVING_CAP_MIN}min pen=${PEN_PER_MIN}/min ===")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                r = f.result()
                rows.append(r)
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['mode']:13s} on {r['instance_id']:32s} K={r['K']:>3d} ops=${r['cost_no_breaks']:>9.1f} w_brk=${r['cost_w_breaks']:>9.1f} wall={r['wall_s']:>5.1f}s")
            except Exception as e:
                print(f"  FAIL: {e}")

    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_win = 0; total = 0.0; n = 0
    print("\n=== paired comparison ===")
    for iid in sorted(by):
        m = by[iid]
        if "baseline" not in m or "breaks_aware" not in m: continue
        b, s = m["baseline"], m["breaks_aware"]
        net = b["cost_w_breaks"] - s["cost_w_breaks"]
        print(f"  {iid:32s} baseline K={b['K']} ops=${b['cost_no_breaks']:.1f} w_brk=${b['cost_w_breaks']:.1f}")
        print(f"  {iid:32s} breaks   K={s['K']} ops=${s['cost_no_breaks']:.1f} w_brk=${s['cost_w_breaks']:.1f}  net={net:+.1f}")
        if net > 0: n_win += 1
        total += net; n += 1
    print(f"\nverdict: breaks_aware wins {n_win}/{n} mean=${total/max(1,n):+.1f}/inst  wall={time.perf_counter()-t0:.1f}s")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "cap_min": DRIVING_CAP_MIN, "pen_per_min": PEN_PER_MIN,
        "rows": rows, "n_winners": n_win, "n_instances": n,
        "total_savings": total, "wall_total_s": time.perf_counter()-t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
