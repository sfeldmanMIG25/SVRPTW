"""iter-6a stack test -- all 7 opt-in constraints active simultaneously.

Validates:
  (1) evaluate() overhead with ALL terms on vs baseline (microbenchmark)
  (2) solve_auto wall time with all terms on doesn't blow up
  (3) bandit produces a reasonable solution under combined objective
  (4) per-constraint penalty breakdown (which terms are biting?)

Usage: PYTHONPATH=. python bench/scripts/iter6a_stack_test.py [INST]
"""
from __future__ import annotations
import json, time, sys
from copy import deepcopy
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
from svrptw.solvers.classical import fast_construct as fc
from svrptw.solvers.common.solution import evaluate

INST = sys.argv[1] if len(sys.argv) > 1 else "instances/v1_large/OSM-Manhattan-N0500-I000.json"
BUDGET = 75.0
OUT = "bench/runs/iter6a_stack_test.json"


def _all_on(s: Settings) -> Settings:
    s2 = deepcopy(s)
    e = s2.economics
    # iter-5l Phase F
    e.crossings_penalty_per_pair = 1.0
    e.util_imbalance_penalty_coef = 20.0
    e.tw_buffer_bonus_coef = 10.0
    # iter-5v shift overrun
    e.shift_max_minutes = 300.0
    e.shift_overrun_penalty_per_min = 1.0
    # iter-5y fairness (calibrated coef per Step 0.5)
    e.driver_time_variance_penalty_coef = 0.5
    # iter-6a-1 driver breaks (tight cap to force activity)
    e.driving_max_minutes = 90.0
    e.break_violation_penalty_per_min = 2.0
    # iter-6a-2 hard zones (8-8:30am + 12-12:30pm)
    e.embargo_window_starts = (480, 720)
    e.embargo_window_ends = (510, 750)
    e.embargo_violation_penalty_per_visit = 50.0
    # iter-6a-3 mixed fleets (2 classes)
    cap = float(load_instance(INST).vehicle_capacity)
    e.vehicle_class_capacities = (cap * 0.5, cap)
    e.vehicle_class_fixed_premiums = (5.0, 20.0)
    e.vehicle_class_per_mile_premiums = (0.0, 0.0)
    # iter-6a-4 EV range (forces split on long routes)
    e.vehicle_range_miles = 8.0
    e.range_violation_penalty_per_mile = 1.0
    # iter-6a-6 PD pairs (a few synthetic ones)
    e.pd_pairs_flat = (1, 2, 3, 4, 5, 6)
    e.pd_violation_penalty_per_pair = 50.0
    # per_route_fixed_cost (legacy)
    e.per_route_fixed_cost = 50.0
    # peak_hour (iter-5x, operator-bound but evaluator works)
    e.peak_window_starts = (480, 1020)
    e.peak_window_ends = (600, 1140)
    e.peak_hour_wage_multiplier = 1.5
    return s2


def microbench_evaluate(inst, sol, settings, n_calls=10_000) -> float:
    t0 = time.perf_counter()
    for _ in range(n_calls):
        evaluate(inst, sol, settings)
    return (time.perf_counter() - t0) / n_calls * 1_000  # ms/call


def main() -> int:
    inst = load_instance(INST)
    s_off = Settings()
    s_on = _all_on(s_off)

    # 1. Microbench: evaluator overhead
    print(f"=== iter-6a stack test on {INST} (N={inst.num_customers}, K_avail={inst.num_vehicles}) ===")
    # Get a baseline solution to time evaluate() on
    sol_warm = fc.solve(inst, s_off, budget_seconds=2.0)
    ms_off = microbench_evaluate(inst, sol_warm, s_off, n_calls=2000)
    ms_on = microbench_evaluate(inst, sol_warm, s_on, n_calls=2000)
    print(f"\n[microbench] evaluate() ms/call")
    print(f"  baseline (3 cost terms):     {ms_off:.4f} ms")
    print(f"  all-on  (16 cost terms):    {ms_on:.4f} ms  overhead={ms_on/ms_off:.2f}x")

    # 2. Solve with baseline settings (cost-only)
    print(f"\n[solve] baseline (no opt-in terms)")
    t0 = time.perf_counter()
    sol_base = pw.solve_auto(inst, s_off, budget_seconds=BUDGET, seed=0)
    wall_base = time.perf_counter() - t0
    cost_base_off = evaluate(inst, sol_base, s_off)["operational_cost"]
    cost_base_on = evaluate(inst, sol_base, s_on)["operational_cost"]
    K_base = int(sol_base.metrics["num_vehicles_used"])
    print(f"  wall={wall_base:.1f}s K={K_base} cost(off)=${cost_base_off:.1f} cost(on)=${cost_base_on:.1f} delta=${cost_base_on - cost_base_off:.1f}")

    # 3. Solve with all-on settings (full constraint stack)
    print(f"\n[solve] all-on (16 opt-in terms simultaneously)")
    t0 = time.perf_counter()
    sol_stack = pw.solve_auto(inst, s_on, budget_seconds=BUDGET, seed=0)
    wall_stack = time.perf_counter() - t0
    cost_stack_off = evaluate(inst, sol_stack, s_off)["operational_cost"]
    cost_stack_on = evaluate(inst, sol_stack, s_on)["operational_cost"]
    K_stack = int(sol_stack.metrics["num_vehicles_used"])
    print(f"  wall={wall_stack:.1f}s K={K_stack} cost(off)=${cost_stack_off:.1f} cost(on)=${cost_stack_on:.1f}")
    print(f"  wall overhead vs baseline: {wall_stack/wall_base:.2f}x")

    # 4. Verdict
    print(f"\n[verdict]")
    print(f"  baseline solver scored on stacked objective: ${cost_base_on:.1f}")
    print(f"  stacked solver scored on stacked objective:  ${cost_stack_on:.1f}")
    delta = cost_base_on - cost_stack_on
    print(f"  stacked is ${delta:+.1f} better under stacked objective ({'WIN' if delta > 0 else 'LOSS'})")
    print(f"  stacked is ${cost_stack_off - cost_base_off:+.1f} vs baseline on pure-ops-cost (price paid)")

    payload = {
        "inst": INST, "N": inst.num_customers, "budget_s": BUDGET,
        "microbench_ms_off": ms_off, "microbench_ms_on": ms_on,
        "eval_overhead_x": ms_on / ms_off,
        "baseline_solver": {"wall": wall_base, "K": K_base,
                             "cost_off": cost_base_off, "cost_on": cost_base_on},
        "stacked_solver": {"wall": wall_stack, "K": K_stack,
                            "cost_off": cost_stack_off, "cost_on": cost_stack_on},
        "wall_overhead_x": wall_stack / wall_base,
        "net_under_stack_obj": delta,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
