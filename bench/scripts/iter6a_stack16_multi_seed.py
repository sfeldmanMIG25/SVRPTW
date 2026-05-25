"""Multi-seed stability bench for iter-6a 16-term full-stack solve.

Runs stack16 at K different seeds on ONE instance to estimate seed-variance.
Converts the headline "+$X/inst" claim from "single-seed snapshot" to
"mean +/- std across K seeds".

Usage: PYTHONPATH=. python bench/scripts/iter6a_stack16_multi_seed.py [INST] [N_SEEDS]
"""
from __future__ import annotations
import json, time, sys
from copy import deepcopy
from pathlib import Path
import statistics

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
from svrptw.solvers.common.solution import evaluate

INST = sys.argv[1] if len(sys.argv) > 1 else "instances/v1_large/OSM-Manhattan-N0500-I000.json"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 75.0


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


def _run_at_seed(inst, s_base, s_full, seed: int) -> dict:
    t0 = time.perf_counter()
    sol_base = pw.solve_auto(inst, s_base, budget_seconds=BUDGET, seed=seed)
    wall_base = time.perf_counter() - t0
    t0 = time.perf_counter()
    sol_full = pw.solve_auto(inst, s_full, budget_seconds=BUDGET, seed=seed)
    wall_full = time.perf_counter() - t0
    c_base_on = evaluate(inst, sol_base, s_full)["operational_cost"]
    c_full_on = evaluate(inst, sol_full, s_full)["operational_cost"]
    net = c_base_on - c_full_on
    return {
        "seed": seed,
        "wall_base": wall_base, "wall_full": wall_full,
        "wall_x": wall_full / wall_base,
        "K_base": int(sol_base.metrics["num_vehicles_used"]),
        "K_full": int(sol_full.metrics["num_vehicles_used"]),
        "cost_base_on": c_base_on, "cost_full_on": c_full_on,
        "net": net,
    }


def main() -> int:
    inst = load_instance(INST)
    s_base = Settings()
    s_full = _make_all_16(s_base, inst)
    print(f"=== multi-seed stack16 stability bench ===")
    print(f"  instance: {INST}  N={inst.num_customers}  budget={BUDGET}s  seeds={N_SEEDS}")

    runs = []
    for sd in range(N_SEEDS):
        print(f"\n[seed {sd}] solving ...", flush=True)
        r = _run_at_seed(inst, s_base, s_full, seed=sd)
        runs.append(r)
        print(f"  wall_x={r['wall_x']:.2f} K {r['K_base']}->{r['K_full']} net=${r['net']:+.1f}", flush=True)

    nets = [r["net"] for r in runs]
    walls = [r["wall_x"] for r in runs]
    mean_net = statistics.mean(nets)
    sd_net = statistics.stdev(nets) if len(nets) >= 2 else 0.0
    mean_wall = statistics.mean(walls)
    print(f"\n=== aggregate (n={N_SEEDS}) ===")
    print(f"  net under stack obj: ${mean_net:+.1f} +/- ${sd_net:.1f}/inst")
    print(f"  wall overhead:       {mean_wall:.2f}x +/- {statistics.stdev(walls) if len(walls)>=2 else 0:.2f}x")
    print(f"  individual nets:     {[f'${n:+.0f}' for n in nets]}")

    out_path = Path(f"bench/runs/iter6a_stack16_multi_seed_{Path(INST).stem}_b{int(BUDGET)}.json")
    out_path.write_text(json.dumps({
        "inst": INST, "N": inst.num_customers, "budget_s": BUDGET,
        "n_seeds": N_SEEDS, "runs": runs,
        "mean_net": mean_net, "sd_net": sd_net,
        "mean_wall_x": mean_wall,
        "verdict_positive": mean_net > 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
