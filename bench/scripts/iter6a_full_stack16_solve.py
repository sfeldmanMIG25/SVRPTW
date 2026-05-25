"""iter-6a-perf-2 validation: ALL 16 cost terms ON (incl crossings_penalty).

Previous attempt (iter6a_full_stack_solve.py) hung at 22min because crossings
was O(N^2). With the per-route-pair cache now landed, this version includes
crossings_penalty_per_pair and confirms the bandit completes within budget.
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
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 75.0
# Derive OUT path from INST stem + budget to avoid overwriting prior runs.
_stem = Path(INST).stem
OUT = f"bench/runs/iter6a_full_stack16_{_stem}_b{int(BUDGET)}.json"


def _make_all_16(s, inst):
    s2 = deepcopy(s)
    e = s2.economics
    # Phase F (full)
    e.crossings_penalty_per_pair = 1.0   # the previously-blocking term
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
    # iter-6a-6
    e.pd_pairs_flat = (1, 2, 3, 4, 5, 6, 7, 8)
    e.pd_violation_penalty_per_pair = 50.0
    # iter-6a-8
    e.min_routes_required = 25
    e.under_min_routes_penalty_per_route = 100.0
    # peak_hour
    e.peak_window_starts = (480, 1020)
    e.peak_window_ends = (600, 1140)
    e.peak_hour_wage_multiplier = 1.5
    # legacy
    e.per_route_fixed_cost = 50.0
    return s2


def main() -> int:
    inst = load_instance(INST)
    print(f"=== iter-6a-perf-2 validation: ALL 16 terms (incl crossings_penalty=1.0) ===")
    print(f"  instance: {INST}  N={inst.num_customers}  budget={BUDGET}s")

    s_base = Settings()
    s_full = _make_all_16(s_base, inst)

    # 1. Baseline solve (no opt-in terms)
    t0 = time.perf_counter()
    sol_base = pw.solve_auto(inst, s_base, budget_seconds=BUDGET, seed=0)
    wall_base = time.perf_counter() - t0

    # 2. Full-stack solve (16 terms incl crossings)
    t0 = time.perf_counter()
    sol_full = pw.solve_auto(inst, s_full, budget_seconds=BUDGET, seed=0)
    wall_full = time.perf_counter() - t0

    c_base_off = evaluate(inst, sol_base, s_base)["operational_cost"]
    c_base_on = evaluate(inst, sol_base, s_full)["operational_cost"]
    c_full_off = evaluate(inst, sol_full, s_base)["operational_cost"]
    c_full_on = evaluate(inst, sol_full, s_full)["operational_cost"]

    K_base = int(sol_base.metrics["num_vehicles_used"])
    K_full = int(sol_full.metrics["num_vehicles_used"])

    print(f"\n[solve] baseline (cost only):")
    print(f"  wall={wall_base:.1f}s K={K_base} ops=${c_base_off:.1f} (under full obj: ${c_base_on:.1f})")
    print(f"\n[solve] full-stack (16 cost terms):")
    print(f"  wall={wall_full:.1f}s K={K_full} ops=${c_full_off:.1f} (under full obj: ${c_full_on:.1f})")

    net = c_base_on - c_full_on
    overhead = wall_full / wall_base
    print(f"\n[verdict]")
    print(f"  wall overhead: {overhead:.2f}x baseline")
    print(f"  budget compliance: full wall <= 2x budget? {wall_full <= 2 * BUDGET}")
    print(f"  net under stacked-16 obj: ${net:+.1f}/inst ({'WIN' if net > 0 else 'LOSS'})")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "inst": INST, "N": inst.num_customers, "budget_s": BUDGET,
        "wall_baseline": wall_base, "wall_full": wall_full,
        "wall_overhead_x": overhead,
        "K_baseline": K_base, "K_full": K_full,
        "cost_base_off": c_base_off, "cost_base_on": c_base_on,
        "cost_full_off": c_full_off, "cost_full_on": c_full_on,
        "net_under_full_obj": net,
        "budget_complied": wall_full <= 2 * BUDGET,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
