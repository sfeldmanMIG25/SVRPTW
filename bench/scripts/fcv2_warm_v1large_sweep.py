"""solve_auto(fcv2 warmstart) vs solve_auto(pyvrp warmstart) on ALL 6 v1_large.

Iter-5q closed warmstart-source-switching on Manhattan N=500 alone, but the
user flagged that one instance isn't enough -- full v1_large sweep is what
either confirms or refutes the closure.

Hypothesis: fcv2's Louvain basin + bandit refinement might pull cost down
toward solve_auto's level while preserving more of fcv2's 0.794 quality
than the bandit destroyed on Manhattan alone.
"""
from __future__ import annotations
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")


def _run_one(task):
    inst_path, construction, budget = task
    from svrptw.io import load_instance
    from svrptw.config import Settings
    from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
    from svrptw.metrics import score_solution
    inst = load_instance(inst_path)
    t0 = time.perf_counter()
    sol = solve_auto(inst, Settings(), budget_seconds=budget,
                      construction=construction, seed=42)
    wall = time.perf_counter() - t0
    sc = score_solution(inst, sol)
    return {
        "instance_id": inst.instance_id, "construction": construction,
        "N": inst.num_customers,
        "cost": float(sol.metrics["operational_cost"]),
        "K": int(sol.metrics["num_vehicles_used"]),
        "q_idx": sc.quality_index,
        "crossings": sc.inter_route_crossings,
        "wall": wall, "budget": budget,
    }


def main() -> int:
    # Match BOTH N0500 (3 instances) and N1000 (3 instances). Original
    # glob "N0*" silently dropped the N1000 set. Now matches both.
    paths = sorted(set(
        list(Path("instances/v1_large").glob("OSM-*-N0500-I000.json"))
        + list(Path("instances/v1_large").glob("OSM-*-N1000-I000.json"))
    ))
    paths = [p for p in paths if "smoke" not in p.name and "v3" not in p.name]
    print(f"sweep on {len(paths)} v1_large instances x 2 constructions = {len(paths)*2} solves")
    tasks = []
    for p in paths:
        N = int(p.stem.split("-N")[-1].split("-")[0])
        budget = max(60.0, 30.0 * N / 200.0)  # 75s @ N=500, 150s @ N=1000
        for cm in ("pyvrp", "fast_construct_v2"):
            tasks.append((str(p), cm, budget))
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            rows.append(r)
            print(f"  {r['construction']:18s} on {r['instance_id']:<32s} "
                  f"cost={r['cost']:>9.1f}  q={r['q_idx']:.3f}  "
                  f"x={r['crossings']:>5d}  K={r['K']:>3d}  wall={r['wall']:.1f}s")
    Path("bench/runs/fcv2_warm_v1large.json").write_text(json.dumps(rows, indent=2))
    # Pivot table
    print(f"\n=== fcv2 vs pyvrp warmstart, paired by instance ===")
    print(f"  {'instance':38s} {'pyvrp_cost':>10s} {'fcv2_cost':>10s} {'d_cost':>8s} "
          f"{'pyvrp_q':>7s} {'fcv2_q':>7s} {'d_q':>7s}")
    by_inst = {}
    for r in rows:
        by_inst.setdefault(r["instance_id"], {})[r["construction"]] = r
    for iid in sorted(by_inst):
        p = by_inst[iid].get("pyvrp", {}); f = by_inst[iid].get("fast_construct_v2", {})
        if not p or not f: continue
        dc = f["cost"] - p["cost"]; dq = f["q_idx"] - p["q_idx"]
        print(f"  {iid:38s} {p['cost']:>10.1f} {f['cost']:>10.1f} {dc:>+8.1f} "
              f"{p['q_idx']:>7.3f} {f['q_idx']:>7.3f} {dq:>+7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
