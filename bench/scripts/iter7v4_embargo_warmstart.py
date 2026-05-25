"""iter-7-v4-embargo: does v4-warmstart handle embargo cost term as well
as pyvrp-warmstart?

The user's original /loop ask: "should still be able to deal with complex
constraints on construction." v4 only handles TW+capacity at construction
time; cost-term constraints (embargo, skills, etc.) apply via the bandit
refinement that follows. This bench validates that composition by running
the same embargo-aware paired bench (iter-6a-2-postfix style) but with two
construction backends -- pyvrp vs v4 -- and comparing the refined costs.

If v4-warmstart + bandit produces similar embargo-aware cost as pyvrp-
warmstart + bandit, the composition works as designed.
"""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

OUT = "bench/runs/iter7v4_embargo_warmstart.json"
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


def _run_one(task):
    p, ctor, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(p)
    s = Settings()
    s.economics.embargo_window_starts = EMBARGO_STARTS
    s.economics.embargo_window_ends = EMBARGO_ENDS
    s.economics.embargo_violation_penalty_per_visit = EMBARGO_PEN
    s_base = Settings()  # baseline (no embargo) for cost decomposition
    t0 = time.perf_counter()
    try:
        sol = pwm.solve_auto(
            inst, s, budget_seconds=budget, construction=ctor, seed=0,
        )
        wall = time.perf_counter() - t0
        cost_no = evaluate(inst, sol, s_base)["operational_cost"]
        cost_em = evaluate(inst, sol, s)["operational_cost"]
        return {
            "instance_id": inst.instance_id, "construction": ctor,
            "N": inst.num_customers,
            "K": int(sol.metrics["num_vehicles_used"]),
            "cost_no_embargo": cost_no,
            "cost_w_embargo": cost_em,
            "embargo_penalty": cost_em - cost_no,
            "feasible": bool(sol.metrics["feasible"]),
            "wall_s": wall, "budget_s": budget, "ok": True,
        }
    except Exception as e:
        return {
            "instance_id": inst.instance_id, "construction": ctor,
            "N": inst.num_customers, "budget_s": budget,
            "wall_s": time.perf_counter() - t0, "ok": False,
            "error": str(e)[:200],
        }


def main():
    tasks = [(p, c, bud) for p, bud in INSTANCES for c in CONSTRUCTIONS]
    print(f"=== iter-7-v4-embargo: {len(tasks)} solves "
          f"(2 constructions x 6 instances, embargo cost term ON) ===",
          flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            if r.get("ok"):
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['construction']:18s} "
                      f"{r['instance_id']:32s} K={r['K']:>3d} "
                      f"cost=${r['cost_w_embargo']:>8.1f} "
                      f"em_pen=${r['embargo_penalty']:>7.1f} "
                      f"wall={r['wall_s']:>5.1f}s", flush=True)
            else:
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['construction']:18s} "
                      f"{r['instance_id']:32s} FAILED: "
                      f"{r.get('error', '?')[:120]}", flush=True)
    # Paired comparison under embargo-aware objective.
    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], {})[r["construction"]] = r
    print("\n=== paired comparison vs pyvrp warmstart (cost under embargo objective) ===")
    n_v4_wins = 0; total = 0.0; n = 0
    for iid in sorted(by):
        m = by[iid]
        if "pyvrp" not in m or "fast_construct_v4" not in m: continue
        py, v4 = m["pyvrp"], m["fast_construct_v4"]
        if not (py.get("ok") and v4.get("ok")): continue
        delta = py["cost_w_embargo"] - v4["cost_w_embargo"]
        if delta > 0: n_v4_wins += 1
        total += delta; n += 1
        print(f"  {iid:32s} pyvrp K={py['K']:>3d} ${py['cost_w_embargo']:>8.1f} "
              f"(pen ${py['embargo_penalty']:>5.0f})  "
              f"v4 K={v4['K']:>3d} ${v4['cost_w_embargo']:>8.1f} "
              f"(pen ${v4['embargo_penalty']:>5.0f})  delta={delta:+.1f}")
    if n > 0:
        print(f"\nverdict: v4 wins {n_v4_wins}/{n} on embargo-aware cost, "
              f"mean delta=${total/n:+.1f}/inst")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "embargo_starts": list(EMBARGO_STARTS),
        "embargo_ends": list(EMBARGO_ENDS),
        "embargo_pen": EMBARGO_PEN,
        "rows": rows, "n_instances": n, "n_v4_wins": n_v4_wins,
        "mean_delta_v4_minus_pyvrp": total / max(1, n),
        "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
