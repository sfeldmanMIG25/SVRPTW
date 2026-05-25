"""iter-7 bandit determinism audit.

Run the SAME stack-16 bench 3 times sequentially at seed=0 (no parallelism).
Measure cost variance across runs. If same-seed-same-code produces tight
results, the prior $2K+ swings between bench runs were parallelism /
system-load anomalies (and any bench's measurements are reliable as long
as run conditions match). If the swing is large, the bandit has internal
nondeterminism that needs anchoring (or all bench claims need wider
confidence intervals).

Single instance (Manhattan-N500), single seed, single construction (v4)
to keep wall short. 3 sequential runs, no overlap, no parallelism.
"""
from __future__ import annotations
import json, time, statistics
from copy import deepcopy
from pathlib import Path


def _make_stack16(s, inst):
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


def main():
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm

    inst = load_instance('instances/v1_large/OSM-Manhattan-N0500-I000.json')
    s_full = _make_stack16(Settings(), inst)
    N_RUNS = 3
    BUDGET = 150.0

    print(f"=== iter-7 determinism audit: {N_RUNS} sequential runs, "
          f"Manhattan-N500 stack-16, seed=0, v4 warmstart ===", flush=True)
    print(f"  budget={BUDGET}s per solve; total expected wall ~{N_RUNS * BUDGET:.0f}s", flush=True)
    results = []
    for run_idx in range(N_RUNS):
        t0 = time.perf_counter()
        sol = pwm.solve_auto(
            inst, s_full, budget_seconds=BUDGET,
            construction='fast_construct_v4', seed=0,
        )
        wall = time.perf_counter() - t0
        result = {
            "run_idx": run_idx,
            "K": int(sol.metrics["num_vehicles_used"]),
            "cost": float(sol.metrics["operational_cost"]),
            "wall_s": wall,
            "feasible": bool(sol.metrics["feasible"]),
        }
        results.append(result)
        print(f"  run {run_idx}: K={result['K']} cost=${result['cost']:.1f} "
              f"wall={wall:.1f}s feas={result['feasible']}", flush=True)

    costs = [r["cost"] for r in results]
    walls = [r["wall_s"] for r in results]
    mean_cost = statistics.mean(costs)
    std_cost = statistics.stdev(costs) if len(costs) >= 2 else 0.0
    swing_cost = max(costs) - min(costs)
    mean_wall = statistics.mean(walls)
    std_wall = statistics.stdev(walls) if len(walls) >= 2 else 0.0

    print(f"\n=== determinism summary ===")
    print(f"  costs: {[f'${c:.0f}' for c in costs]}")
    print(f"  cost mean=${mean_cost:.1f}  std=${std_cost:.1f}  swing=${swing_cost:.1f}")
    print(f"  walls: {[f'{w:.1f}s' for w in walls]}")
    print(f"  wall mean={mean_wall:.1f}s std={std_wall:.1f}s")
    print()
    if swing_cost < 50.0:
        print("VERDICT: bandit IS effectively deterministic (swing < $50).")
        print("  Prior bench swings were system-load / parallelism anomalies.")
    elif swing_cost < 500.0:
        print("VERDICT: bandit has SMALL nondeterminism (swing $50-$500).")
        print("  Acceptable but bench claims should include +/- this as error bar.")
    else:
        print("VERDICT: bandit has LARGE nondeterminism (swing > $500).")
        print("  Needs investigation; bench claims unreliable without multi-run averaging.")

    out = Path("bench/runs/iter7_determinism_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_runs": N_RUNS,
        "instance": "OSM-Manhattan-N0500-I000",
        "construction": "fast_construct_v4",
        "seed": 0,
        "budget_s": BUDGET,
        "results": results,
        "cost_mean": mean_cost, "cost_std": std_cost, "cost_swing": swing_cost,
        "wall_mean": mean_wall, "wall_std": std_wall,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
