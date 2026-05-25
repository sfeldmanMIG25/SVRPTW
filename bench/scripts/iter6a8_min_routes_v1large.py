"""iter-6a-8 min routes (labor contract) bench."""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

OUT = "bench/runs/iter6a8_min_routes_v1large.json"
LOG = "bench/runs/iter6a8_min_routes_v1large.log"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]
# Set min ABOVE the typical solve_auto K so the bandit has to add routes.
# Typical: 16-32 routes. Set min to 25 (N500) and 40 (N1000).
MIN_N500 = 25
MIN_N1000 = 40
PEN = 100.0


def _run_one(task):
    p, mode, b = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(p)
    s_base = Settings()
    s_mr = deepcopy(s_base)
    n_min = MIN_N1000 if inst.num_customers >= 800 else MIN_N500
    s_mr.economics.min_routes_required = n_min
    s_mr.economics.under_min_routes_penalty_per_route = PEN
    used = s_mr if mode == "min_routes_aware" else s_base
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, used, budget_seconds=b, seed=0)
    wall = time.perf_counter() - t0
    cost_no = evaluate(inst, sol, s_base)["operational_cost"]
    cost_w = evaluate(inst, sol, s_mr)["operational_cost"]
    return {"instance_id": inst.instance_id, "mode": mode,
            "K": int(sol.metrics["num_vehicles_used"]),
            "min_req": n_min,
            "cost_no_mr": cost_no, "cost_w_mr": cost_w,
            "mr_penalty": cost_w - cost_no, "wall_s": wall}


def main():
    tasks = [(p, m, b) for p, b in INSTANCES for m in ("baseline", "min_routes_aware")]
    print(f"=== iter-6a-8 min_routes pen=${PEN}/route-short, {len(tasks)} solves ===")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            print(f"  [{len(rows):>2}/{len(tasks)}] {r['mode']:18s} {r['instance_id']:32s} K={r['K']:>3d}/min={r['min_req']} pen=${r['mr_penalty']:.1f} wall={r['wall_s']:.1f}s")
    by = {}
    for r in rows: by.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_win = 0; total = 0.0; n = 0
    for iid in sorted(by):
        m = by[iid]
        if "baseline" not in m or "min_routes_aware" not in m: continue
        b, s = m["baseline"], m["min_routes_aware"]
        net = b["cost_w_mr"] - s["cost_w_mr"]
        if net > 0: n_win += 1
        total += net; n += 1
        print(f"  {iid:32s} K {b['K']}->{s['K']} pen ${b['mr_penalty']:.1f}->${s['mr_penalty']:.1f}  net=${net:+.1f}")
    print(f"\nverdict: {n_win}/{n} mean=${total/max(1,n):+.1f}")
    Path(OUT).write_text(json.dumps({"rows": rows, "n_winners": n_win, "n_instances": n,
                                       "total_savings": total, "wall_total_s": time.perf_counter()-t0}, indent=2))


if __name__ == "__main__": main()
