"""iter-7-splitroute-full6: full 6-instance stack-16 bench with the new
split_route operator registered. Compares pyvrp-warmstart vs v4-warmstart
across all 6 v1_large instances at standard budget.

Question: does split_route generalize beyond Manhattan-N500 (where the
single-seed gain was +$2,639/inst)? If consistent, ship the operator and
update the stack-16 numbers in README.
"""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

OUT = "bench/runs/iter7_splitroute_full6.json"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 150.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 300.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 300.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 300.0),
]
CONSTRUCTIONS = ("pyvrp", "fast_construct_v4")


def _make_all_16(s, inst):
    s2 = deepcopy(s)
    e = s2.economics
    e.crossings_penalty_per_pair = 1.0
    e.util_imbalance_penalty_coef = 20.0
    e.tw_buffer_bonus_coef = 10.0
    e.shift_max_minutes = 300.0
    e.shift_overrun_penalty_per_min = 1.0
    e.driver_time_variance_penalty_coef = 0.5
    e.driving_max_minutes = 90.0
    e.break_violation_penalty_per_min = 2.0
    e.embargo_window_starts = (480, 720)
    e.embargo_window_ends = (510, 750)
    e.embargo_violation_penalty_per_visit = 50.0
    e.vehicle_class_capacities = (inst.vehicle_capacity * 0.5, inst.vehicle_capacity)
    e.vehicle_class_fixed_premiums = (5.0, 20.0)
    e.vehicle_class_per_mile_premiums = (0.0, 0.1)
    e.vehicle_range_miles = 8.0
    e.range_violation_penalty_per_mile = 1.0
    e.pd_pairs_flat = (1, 2, 3, 4, 5, 6, 7, 8)
    e.pd_violation_penalty_per_pair = 50.0
    e.min_routes_required = 25
    e.under_min_routes_penalty_per_route = 100.0
    e.peak_window_starts = (480, 1020)
    e.peak_window_ends = (600, 1140)
    e.peak_hour_wage_multiplier = 1.5
    e.per_route_fixed_cost = 50.0
    return s2


def _run_one(task):
    p, ctor, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(p)
    s_full = _make_all_16(Settings(), inst)
    t0 = time.perf_counter()
    try:
        sol = pwm.solve_auto(
            inst, s_full, budget_seconds=budget, construction=ctor, seed=0,
        )
        wall = time.perf_counter() - t0
        cost_full = evaluate(inst, sol, s_full)["operational_cost"]
        return {
            "instance_id": inst.instance_id, "construction": ctor,
            "N": inst.num_customers,
            "K": int(sol.metrics["num_vehicles_used"]),
            "cost_stack16": cost_full,
            "feasible": bool(sol.metrics["feasible"]),
            "wall_s": wall, "budget_s": budget, "ok": True,
        }
    except Exception as e:
        return {
            "instance_id": inst.instance_id, "construction": ctor,
            "ok": False, "error": str(e)[:200],
            "wall_s": time.perf_counter() - t0,
        }


def main():
    tasks = [(p, c, bud) for p, bud in INSTANCES for c in CONSTRUCTIONS]
    print(f"=== iter-7-splitroute-full6: {len(tasks)} solves "
          f"(stack-16 across 6 inst, split_route in pool) ===", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            if r.get("ok"):
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['construction']:18s} "
                      f"{r['instance_id']:32s} K={r['K']:>3d} "
                      f"cost=${r['cost_stack16']:>9.1f} "
                      f"wall={r['wall_s']:>6.1f}s", flush=True)
            else:
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['construction']:18s} "
                      f"{r['instance_id']:32s} FAILED: "
                      f"{r.get('error', '?')[:80]}", flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], {})[r["construction"]] = r
    print("\n=== paired comparison vs pyvrp warmstart (stack-16 objective) ===")
    n_v4_wins = 0; total = 0.0; n = 0
    pyvrp_sum = 0.0; v4_sum = 0.0
    for iid in sorted(by):
        m = by[iid]
        if "pyvrp" not in m or "fast_construct_v4" not in m: continue
        py, v4 = m["pyvrp"], m["fast_construct_v4"]
        if not (py.get("ok") and v4.get("ok")): continue
        delta = py["cost_stack16"] - v4["cost_stack16"]
        if delta > 0: n_v4_wins += 1
        total += delta; n += 1
        pyvrp_sum += py["cost_stack16"]; v4_sum += v4["cost_stack16"]
        print(f"  {iid:32s} pyvrp K={py['K']:>3d} ${py['cost_stack16']:>9.1f}  "
              f"v4 K={v4['K']:>3d} ${v4['cost_stack16']:>9.1f}  delta={delta:+.1f}")
    if n > 0:
        print(f"\nverdict: v4 wins {n_v4_wins}/{n} on stack-16 cost, "
              f"mean delta=${total/n:+.1f}/inst")
        print(f"  mean pyvrp ${pyvrp_sum/n:.1f} vs mean v4 ${v4_sum/n:.1f}")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "rows": rows, "n_instances": n, "n_v4_wins": n_v4_wins,
        "mean_delta_v4_minus_pyvrp": total / max(1, n),
        "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
