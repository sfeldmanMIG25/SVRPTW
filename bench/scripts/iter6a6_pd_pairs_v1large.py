"""iter-6a-6 PD pairs bench. Use synthetic pairs: pair customer 1<->2,
3<->4, ..., up to 20 pairs. These are arbitrary but force the bandit
to consider precedence in route construction."""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

OUT = "bench/runs/iter6a6_pd_pairs_v1large.json"
LOG = "bench/runs/iter6a6_pd_pairs_v1large.log"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]
# 20 synthetic pairs: (1,2,3,4,...,39,40)
PD_PAIRS_FLAT = tuple(range(1, 41))
PEN_PER_PAIR = 50.0


def _run_one(task):
    p, mode, b = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(p)
    s_base = Settings()
    s_pd = deepcopy(s_base)
    s_pd.economics.pd_pairs_flat = PD_PAIRS_FLAT
    s_pd.economics.pd_violation_penalty_per_pair = PEN_PER_PAIR
    used = s_pd if mode == "pd_aware" else s_base
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, used, budget_seconds=b, seed=0)
    wall = time.perf_counter() - t0
    cost_no = evaluate(inst, sol, s_base)["operational_cost"]
    cost_w = evaluate(inst, sol, s_pd)["operational_cost"]
    return {"instance_id": inst.instance_id, "mode": mode,
            "K": int(sol.metrics["num_vehicles_used"]),
            "cost_no_pd": cost_no, "cost_w_pd": cost_w,
            "pd_penalty": cost_w - cost_no, "wall_s": wall}


def main():
    tasks = [(p, m, b) for p, b in INSTANCES for m in ("baseline", "pd_aware")]
    print(f"=== iter-6a-6 PD pairs (20 pairs, pen=${PEN_PER_PAIR}/violation), {len(tasks)} solves ===")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            print(f"  [{len(rows):>2}/{len(tasks)}] {r['mode']:9s} {r['instance_id']:32s} K={r['K']:>3d} pen=${r['pd_penalty']:.1f} wall={r['wall_s']:.1f}s")
    by = {}
    for r in rows: by.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_win = 0; total = 0.0; n = 0
    for iid in sorted(by):
        m = by[iid]
        if "baseline" not in m or "pd_aware" not in m: continue
        b, s = m["baseline"], m["pd_aware"]
        net = b["cost_w_pd"] - s["cost_w_pd"]
        if net > 0: n_win += 1
        total += net; n += 1
        print(f"  {iid:32s} K {b['K']}->{s['K']} pen ${b['pd_penalty']:.1f}->${s['pd_penalty']:.1f}  net=${net:+.1f}")
    print(f"\nverdict: {n_win}/{n} mean=${total/max(1,n):+.1f}")
    Path(OUT).write_text(json.dumps({"rows": rows, "n_winners": n_win, "n_instances": n,
                                       "total_savings": total, "wall_total_s": time.perf_counter()-t0}, indent=2))


if __name__ == "__main__": main()
