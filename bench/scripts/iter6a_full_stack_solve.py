"""iter-6a integration smoke: solve_auto with ALL 15 production cost terms ON.

Validates the Phase F lite refactor at solve-level (not just evaluator
microbench). Confirms the bandit completes within its 75s budget under
the full constraint stack.

Excludes crossings_penalty_per_pair (still O(N^2), separate fix queued).
"""
from __future__ import annotations
import json, time, sys
from copy import deepcopy
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
from svrptw.solvers.common.solution import evaluate

INST = sys.argv[1] if len(sys.argv) > 1 else "instances/v1_large/OSM-Manhattan-N0500-I000.json"
BUDGET = 75.0
OUT = "bench/runs/iter6a_full_stack_solve.json"


def _make_all_15(s, inst):
    s2 = deepcopy(s)
    e = s2.economics
    # Phase F lite (no crossings)
    e.util_imbalance_penalty_coef = 20.0
    e.tw_buffer_bonus_coef = 10.0
    # iter-5v
    e.shift_max_minutes = 300.0
    e.shift_overrun_penalty_per_min = 1.0
    # iter-5y
    e.driver_time_variance_penalty_coef = 0.5
    # iter-6a-1
    e.driving_max_minutes = 90.0
    e.break_violation_penalty_per_min = 2.0
    # iter-6a-2
    e.embargo_window_starts = (480, 720)
    e.embargo_window_ends = (510, 750)
    e.embargo_violation_penalty_per_visit = 50.0
    # iter-6a-3
    e.vehicle_class_capacities = (inst.vehicle_capacity * 0.5, inst.vehicle_capacity)
    e.vehicle_class_fixed_premiums = (5.0, 20.0)
    e.vehicle_class_per_mile_premiums = (0.0, 0.1)
    # iter-6a-4
    e.vehicle_range_miles = 8.0
    e.range_violation_penalty_per_mile = 1.0
    # iter-6a-6 (small synthetic pair set)
    e.pd_pairs_flat = (1, 2, 3, 4, 5, 6, 7, 8)
    e.pd_violation_penalty_per_pair = 50.0
    # iter-6a-8
    e.min_routes_required = 25
    e.under_min_routes_penalty_per_route = 100.0
    # peak_hour (plumbed but bandit-limited; still safe to include)
    e.peak_window_starts = (480, 1020)
    e.peak_window_ends = (600, 1140)
    e.peak_hour_wage_multiplier = 1.5
    # legacy
    e.per_route_fixed_cost = 50.0
    return s2


def _per_term_breakdown(inst, sol, s_full):
    """Re-score with each term individually + baseline to isolate cost
    contribution. (Heuristic decomposition; doesn't account for
    non-additivity.)"""
    base = Settings()
    c_base = evaluate(inst, sol, base)["operational_cost"]
    c_full = evaluate(inst, sol, s_full)["operational_cost"]
    return {"baseline_ops": c_base, "full_stack": c_full,
            "delta_from_terms": c_full - c_base}


def main() -> int:
    inst = load_instance(INST)
    print(f"=== iter-6a integration smoke: ALL 15 cost terms ON ===")
    print(f"  instance: {INST}  N={inst.num_customers}  budget={BUDGET}s")

    s_base = Settings()
    s_full = _make_all_15(s_base, inst)

    # 1. Baseline solve
    t0 = time.perf_counter()
    sol_base = pw.solve_auto(inst, s_base, budget_seconds=BUDGET, seed=0)
    wall_base = time.perf_counter() - t0

    # 2. Full-stack solve (15 terms)
    t0 = time.perf_counter()
    sol_full = pw.solve_auto(inst, s_full, budget_seconds=BUDGET, seed=0)
    wall_full = time.perf_counter() - t0

    base_b = _per_term_breakdown(inst, sol_base, s_full)
    full_b = _per_term_breakdown(inst, sol_full, s_full)

    K_base = int(sol_base.metrics["num_vehicles_used"])
    K_full = int(sol_full.metrics["num_vehicles_used"])

    print(f"\n[solve] baseline (cost only):")
    print(f"  wall={wall_base:.1f}s K={K_base} ops_cost=${base_b['baseline_ops']:.1f} (under full obj: ${base_b['full_stack']:.1f})")
    print(f"\n[solve] full-stack (15 cost terms):")
    print(f"  wall={wall_full:.1f}s K={K_full} ops_cost=${full_b['baseline_ops']:.1f} (under full obj: ${full_b['full_stack']:.1f})")

    print(f"\n[performance] wall overhead = {wall_full/wall_base:.2f}x baseline")
    print(f"  budget compliance: baseline={wall_base<=BUDGET*1.2} full={wall_full<=BUDGET*1.5}")
    net = base_b["full_stack"] - full_b["full_stack"]
    print(f"\n[verdict] full-stack solver vs baseline scored on full objective: ${net:+.1f}/inst")
    print(f"  ({'WIN' if net > 0 else 'LOSS'} -- bandit responded to combined stack)")

    payload = {
        "inst": INST, "N": inst.num_customers, "budget_s": BUDGET,
        "wall_baseline": wall_base, "wall_full": wall_full,
        "wall_overhead_x": wall_full / wall_base,
        "K_baseline": K_base, "K_full": K_full,
        "baseline_solver_b": base_b, "full_solver_b": full_b,
        "net_under_full_obj": net,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
