"""iter-7 sequential driver_breaks bench: re-runs iter7v4_driver_breaks
configuration with max_workers=1 to get the contention-free truth.

The parallel iter7v4_driver_breaks_warmstart bench reported v4 loses 0/6
mean -$291/inst (driving_cap=90min, $2/min). Embargo's similar sequential
re-bench (iter-7-embargo-sequential) inverted that finding from +$178 to
-$122. This bench checks if driver_breaks also shifts under sequential.
"""
from __future__ import annotations
import json, time
from pathlib import Path

OUT = "bench/runs/iter7v4_driver_breaks_sequential.json"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]
DRIVING_CAP_MIN = 90.0
BREAK_PENALTY_PER_MIN = 2.0
CONSTRUCTIONS = ("pyvrp", "fast_construct_v4")


def main():
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm
    from svrptw.solvers.common.solution import evaluate

    tasks = [(p, c, b) for p, b in INSTANCES for c in CONSTRUCTIONS]
    print(f"=== iter-7 SEQUENTIAL driver_breaks bench: {len(tasks)} solves "
          f"(max_workers=1) ===", flush=True)
    t0_total = time.perf_counter()
    rows = []
    for i, (p, ctor, budget) in enumerate(tasks, 1):
        inst = load_instance(p)
        s = Settings()
        s.economics.driving_max_minutes = DRIVING_CAP_MIN
        s.economics.break_violation_penalty_per_min = BREAK_PENALTY_PER_MIN
        s_base = Settings()
        t0 = time.perf_counter()
        sol = pwm.solve_auto(
            inst, s, budget_seconds=budget, construction=ctor, seed=0,
        )
        wall = time.perf_counter() - t0
        cost_w = evaluate(inst, sol, s)["operational_cost"]
        cost_no = evaluate(inst, sol, s_base)["operational_cost"]
        K = int(sol.metrics["num_vehicles_used"])
        rows.append({
            "instance_id": inst.instance_id, "construction": ctor,
            "N": inst.num_customers, "K": K,
            "cost_w_break": cost_w, "break_penalty": cost_w - cost_no,
            "wall_s": wall, "budget_s": budget,
        })
        print(f"  [{i:>2}/{len(tasks)}] {ctor:18s} {inst.instance_id:32s} "
              f"K={K:>3d} cost=${cost_w:>8.1f} "
              f"brk=${cost_w - cost_no:>6.1f} wall={wall:>5.1f}s",
              flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], {})[r["construction"]] = r
    print("\n=== paired comparison vs pyvrp warmstart ===")
    n_v4_wins = 0; total = 0.0; n = 0
    pyvrp_sum = 0.0; v4_sum = 0.0
    for iid in sorted(by):
        m = by[iid]
        py, v4 = m["pyvrp"], m["fast_construct_v4"]
        delta = py["cost_w_break"] - v4["cost_w_break"]
        if delta > 0: n_v4_wins += 1
        total += delta; n += 1
        pyvrp_sum += py["cost_w_break"]; v4_sum += v4["cost_w_break"]
        print(f"  {iid:32s} pyvrp K={py['K']:>3d} ${py['cost_w_break']:>8.1f} "
              f"(brk ${py['break_penalty']:>5.0f})  "
              f"v4 K={v4['K']:>3d} ${v4['cost_w_break']:>8.1f} "
              f"(brk ${v4['break_penalty']:>5.0f})  delta={delta:+.1f}")
    if n > 0:
        print(f"\nverdict: v4 wins {n_v4_wins}/{n}, mean delta=${total/n:+.1f}/inst")
        print(f"  mean pyvrp ${pyvrp_sum/n:.1f} vs v4 ${v4_sum/n:.1f}")
        print(f"  historical claim (parallel): v4 0/6 -$291/inst")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "driving_cap_min": DRIVING_CAP_MIN,
        "break_penalty_per_min": BREAK_PENALTY_PER_MIN,
        "rows": rows, "n_instances": n, "n_v4_wins": n_v4_wins,
        "mean_delta_v4_minus_pyvrp": total / max(1, n),
        "wall_total_s": time.perf_counter() - t0_total,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
