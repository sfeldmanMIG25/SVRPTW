"""iter-7 sequential embargo bench: re-runs iter7v4_embargo_warmstart
configuration but with max_workers=1 (sequential execution) to get the
noise-free ground truth on v4-vs-pyvrp under embargo.

The parallel iter7v4_embargo_warmstart bench reported +$178/inst (single-
seed) and +$194/inst (multi-seed) v4 wins, but the iter-7-determinism-
audit showed parallel benches have $50-$2,600/inst noise from CPU
contention. This sequential rerun gives the calibrated true magnitude.
"""
from __future__ import annotations
import json, time
from pathlib import Path

OUT = "bench/runs/iter7v4_embargo_sequential.json"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]
EMBARGO_STARTS = (480, 720)
EMBARGO_ENDS = (510, 750)
EMBARGO_PEN = 50.0
CONSTRUCTIONS = ("pyvrp", "fast_construct_v4")


def main():
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm
    from svrptw.solvers.common.solution import evaluate

    tasks = [(p, c, b) for p, b in INSTANCES for c in CONSTRUCTIONS]
    print(f"=== iter-7 SEQUENTIAL embargo bench: {len(tasks)} solves "
          f"(max_workers=1, no contention) ===", flush=True)
    t0_total = time.perf_counter()
    rows = []
    for i, (p, ctor, budget) in enumerate(tasks, 1):
        inst = load_instance(p)
        s = Settings()
        s.economics.embargo_window_starts = EMBARGO_STARTS
        s.economics.embargo_window_ends = EMBARGO_ENDS
        s.economics.embargo_violation_penalty_per_visit = EMBARGO_PEN
        t0 = time.perf_counter()
        sol = pwm.solve_auto(
            inst, s, budget_seconds=budget, construction=ctor, seed=0,
        )
        wall = time.perf_counter() - t0
        cost_w = evaluate(inst, sol, s)["operational_cost"]
        K = int(sol.metrics["num_vehicles_used"])
        rows.append({
            "instance_id": inst.instance_id, "construction": ctor,
            "N": inst.num_customers, "K": K,
            "cost_w_embargo": cost_w, "wall_s": wall, "budget_s": budget,
        })
        print(f"  [{i:>2}/{len(tasks)}] {ctor:18s} {inst.instance_id:32s} "
              f"K={K:>3d} cost=${cost_w:>8.1f} wall={wall:>5.1f}s",
              flush=True)
    # Paired comparison
    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], {})[r["construction"]] = r
    print("\n=== paired comparison vs pyvrp warmstart ===")
    n_v4_wins = 0; total = 0.0; n = 0
    pyvrp_sum = 0.0; v4_sum = 0.0
    for iid in sorted(by):
        m = by[iid]
        py, v4 = m["pyvrp"], m["fast_construct_v4"]
        delta = py["cost_w_embargo"] - v4["cost_w_embargo"]
        if delta > 0: n_v4_wins += 1
        total += delta; n += 1
        pyvrp_sum += py["cost_w_embargo"]; v4_sum += v4["cost_w_embargo"]
        print(f"  {iid:32s} pyvrp K={py['K']:>3d} ${py['cost_w_embargo']:>8.1f}  "
              f"v4 K={v4['K']:>3d} ${v4['cost_w_embargo']:>8.1f}  delta={delta:+.1f}")
    if n > 0:
        print(f"\nverdict: v4 wins {n_v4_wins}/{n}, mean delta=${total/n:+.1f}/inst")
        print(f"  mean pyvrp ${pyvrp_sum/n:.1f} vs v4 ${v4_sum/n:.1f}")
        print(f"  historical claim (parallel): v4 +$178 single-seed, +$194 multi-seed")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "rows": rows, "n_instances": n, "n_v4_wins": n_v4_wins,
        "mean_delta_v4_minus_pyvrp": total / max(1, n),
        "wall_total_s": time.perf_counter() - t0_total,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
