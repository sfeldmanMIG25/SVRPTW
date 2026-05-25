"""iter-6a-3 -- mixed fleets paired bench on v1_large.

Two vehicle classes: small (50% cap, $5 fixed premium), large (100% cap,
$20 fixed premium + $0.10/mile premium). Bandit can consolidate routes
onto fewer large-class vehicles OR split onto more small-class vehicles
depending on what's cheaper for each instance.
"""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

OUT = "bench/runs/iter6a3_mixed_fleets_v1large.json"
LOG = "bench/runs/iter6a3_mixed_fleets_v1large.log"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]


def _run_one(task):
    p, mode, b = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(p)
    s_base = Settings()
    s_mf = deepcopy(s_base)
    cap = float(inst.vehicle_capacity)
    s_mf.economics.vehicle_class_capacities = (cap * 0.5, cap)
    s_mf.economics.vehicle_class_fixed_premiums = (5.0, 20.0)
    s_mf.economics.vehicle_class_per_mile_premiums = (0.0, 0.10)
    used = s_mf if mode == "mixed_fleets" else s_base
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, used, budget_seconds=b, seed=0)
    wall = time.perf_counter() - t0
    cost_no = evaluate(inst, sol, s_base)["operational_cost"]
    cost_w = evaluate(inst, sol, s_mf)["operational_cost"]
    return {"instance_id": inst.instance_id, "mode": mode,
            "K": int(sol.metrics["num_vehicles_used"]),
            "cost_no_mf": cost_no, "cost_w_mf": cost_w,
            "mf_premium": cost_w - cost_no,
            "wall_s": wall, "budget_s": b}


def main():
    tasks = []
    for p, b in INSTANCES:
        for m in ("baseline", "mixed_fleets"): tasks.append((p, m, b))
    print(f"=== iter-6a-3 mixed_fleets, {len(tasks)} solves ===")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            print(f"  [{len(rows):>2}/{len(tasks)}] {r['mode']:14s} on {r['instance_id']:32s} K={r['K']:>3d} ops=${r['cost_no_mf']:>9.1f} w_mf=${r['cost_w_mf']:>9.1f} wall={r['wall_s']:>5.1f}s")
    by = {}
    for r in rows: by.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_win = 0; total = 0.0; n = 0
    for iid in sorted(by):
        m = by[iid]
        if "baseline" not in m or "mixed_fleets" not in m: continue
        b, s = m["baseline"], m["mixed_fleets"]
        net = b["cost_w_mf"] - s["cost_w_mf"]
        print(f"  {iid:32s} K {b['K']}->{s['K']} pen ${b['mf_premium']:.1f}->${s['mf_premium']:.1f}  net=${net:+.1f}")
        if net > 0: n_win += 1
        total += net; n += 1
    print(f"\nverdict: {n_win}/{n} mean=${total/max(1,n):+.1f}")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({"rows": rows, "n_winners": n_win, "n_instances": n,
                                       "total_savings": total, "wall_total_s": time.perf_counter()-t0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
