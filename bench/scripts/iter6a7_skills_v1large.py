"""iter-6a-7 skills bench. Activates mixed_fleets alongside (skills needs
class assignment to be meaningful). Customers with cid in [1..30] require
skill level 2; class 0 (smallest) provides level 0; class 1 (largest)
provides level 2. Bandit should relocate skill-2 customers onto class-1
(large) routes."""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

OUT = "bench/runs/iter6a7_skills_v1large.json"
LOG = "bench/runs/iter6a7_skills_v1large.log"
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
    # Always activate mixed_fleets in both baseline and aware: skills
    # requires class assignment to be meaningful.
    cap = float(inst.vehicle_capacity)
    s_base.economics.vehicle_class_capacities = (cap * 0.5, cap)
    s_base.economics.vehicle_class_fixed_premiums = (0.0, 0.0)
    s_base.economics.vehicle_class_per_mile_premiums = (0.0, 0.0)
    s_skl = deepcopy(s_base)
    # 30 customers require skill 2; class 0 provides 0, class 1 provides 2.
    pairs = []
    for cid in range(1, 31):
        pairs.append(cid); pairs.append(2)
    s_skl.economics.customer_skill_levels_flat = tuple(pairs)
    s_skl.economics.vehicle_class_skill_levels = (0, 2)
    s_skl.economics.skill_mismatch_penalty_per_visit = 100.0
    used = s_skl if mode == "skills_aware" else s_base
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, used, budget_seconds=b, seed=0)
    wall = time.perf_counter() - t0
    cost_no = evaluate(inst, sol, s_base)["operational_cost"]
    cost_w = evaluate(inst, sol, s_skl)["operational_cost"]
    return {"instance_id": inst.instance_id, "mode": mode,
            "K": int(sol.metrics["num_vehicles_used"]),
            "cost_no_skl": cost_no, "cost_w_skl": cost_w,
            "skill_pen": cost_w - cost_no, "wall_s": wall}


def main():
    tasks = [(p, m, b) for p, b in INSTANCES for m in ("baseline", "skills_aware")]
    print(f"=== iter-6a-7 skills (30 high-skill customers, 2 classes), {len(tasks)} solves ===")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            print(f"  [{len(rows):>2}/{len(tasks)}] {r['mode']:13s} {r['instance_id']:32s} K={r['K']:>3d} pen=${r['skill_pen']:.1f} wall={r['wall_s']:.1f}s")
    by = {}
    for r in rows: by.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_win = 0; total = 0.0; n = 0
    for iid in sorted(by):
        m = by[iid]
        if "baseline" not in m or "skills_aware" not in m: continue
        b, s = m["baseline"], m["skills_aware"]
        net = b["cost_w_skl"] - s["cost_w_skl"]
        if net > 0: n_win += 1
        total += net; n += 1
        print(f"  {iid:32s} K {b['K']}->{s['K']} pen ${b['skill_pen']:.1f}->${s['skill_pen']:.1f}  net=${net:+.1f}")
    print(f"\nverdict: {n_win}/{n} mean=${total/max(1,n):+.1f}")
    Path(OUT).write_text(json.dumps({"rows": rows, "n_winners": n_win, "n_instances": n,
                                       "total_savings": total, "wall_total_s": time.perf_counter()-t0}, indent=2))


if __name__ == "__main__": main()
