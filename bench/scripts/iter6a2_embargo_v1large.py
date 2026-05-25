"""iter-6a-2 -- hard zones / global embargo windows on v1_large.

Two embargo windows: 8-8:30am (school zone closed) + 12-12:30pm (lunch
delivery embargo). Per-visit penalty $50. Step 0: bandit can reorder
customers within routes to shift arrival times -- limited by customer TWs.
"""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

OUT = "bench/runs/iter6a2_embargo_v1large_postfix.json"
LOG = "bench/runs/iter6a2_embargo_v1large_postfix.log"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]
EMBARGO_STARTS = (480, 720)   # 8am, noon
EMBARGO_ENDS = (510, 750)     # 8:30am, 12:30pm
PEN_PER_VISIT = 50.0


def _run_one(task):
    p, mode, b = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(p)
    s_base = Settings()
    s_em = deepcopy(s_base)
    s_em.economics.embargo_window_starts = EMBARGO_STARTS
    s_em.economics.embargo_window_ends = EMBARGO_ENDS
    s_em.economics.embargo_violation_penalty_per_visit = PEN_PER_VISIT
    used = s_em if mode == "embargo_aware" else s_base
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, used, budget_seconds=b, seed=0)
    wall = time.perf_counter() - t0
    cost_no = evaluate(inst, sol, s_base)["operational_cost"]
    cost_w = evaluate(inst, sol, s_em)["operational_cost"]
    return {
        "instance_id": inst.instance_id, "mode": mode,
        "K": int(sol.metrics["num_vehicles_used"]),
        "cost_no_em": cost_no, "cost_w_em": cost_w,
        "embargo_penalty": cost_w - cost_no,
        "wall_s": wall, "budget_s": b,
    }


def main():
    tasks = []
    for p, b in INSTANCES:
        for m in ("baseline", "embargo_aware"): tasks.append((p, m, b))
    print(f"=== iter-6a-2: embargo 8-8:30am + 12-12:30pm @ ${PEN_PER_VISIT}/visit, {len(tasks)} solves ===")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            print(f"  [{len(rows):>2}/{len(tasks)}] {r['mode']:13s} on {r['instance_id']:32s} K={r['K']:>3d} ops=${r['cost_no_em']:>9.1f} w_em=${r['cost_w_em']:>9.1f} wall={r['wall_s']:>5.1f}s")
    by = {}
    for r in rows: by.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_win = 0; total = 0.0; n = 0
    print("\n=== paired comparison ===")
    for iid in sorted(by):
        m = by[iid]
        if "baseline" not in m or "embargo_aware" not in m: continue
        b, s = m["baseline"], m["embargo_aware"]
        net = b["cost_w_em"] - s["cost_w_em"]
        print(f"  {iid:32s} baseline K={b['K']} pen=${b['embargo_penalty']:.1f}")
        print(f"  {iid:32s} embargo  K={s['K']} pen=${s['embargo_penalty']:.1f}  net={net:+.1f}")
        if net > 0: n_win += 1
        total += net; n += 1
    print(f"\nverdict: embargo_aware wins {n_win}/{n} mean=${total/max(1,n):+.1f}/inst")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "embargo_starts": list(EMBARGO_STARTS), "embargo_ends": list(EMBARGO_ENDS),
        "pen_per_visit": PEN_PER_VISIT,
        "rows": rows, "n_winners": n_win, "n_instances": n,
        "total_savings": total, "wall_total_s": time.perf_counter()-t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
