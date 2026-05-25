"""iter-7-v4-driver_breaks: does v4-warmstart compose with the per-route
driver_breaks cost term (iter-6a-1-bis: 90min driving cap, $2/min penalty)?

Per-route cost terms (driver_breaks, shift_overrun, EV_range) don't have a
direct per-insertion analog like embargo does. v4's cost_aware mechanism is
a no-op for them. The question is whether v4's construction basin allows
the bandit to refine the per-route penalty as well as pyvrp's does.

If v4 ties or beats pyvrp: v4 generalizes to per-route cost terms via
bandit refinement alone (no construction-side work needed).
If v4 loses by > $200/inst: the per-route cost term exposes a basin gap
that would need either a per-route v4 extension or a different fix.
"""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = "bench/runs/iter7v4_driver_breaks_warmstart.json"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]
# iter-6a-1-bis config: 90min driving cap, $2/min penalty (tight cap so
# violations are forced; loose cap leaves baseline already compliant).
DRIVING_CAP_MIN = 90.0
BREAK_PENALTY_PER_MIN = 2.0

CONSTRUCTIONS = ("pyvrp", "fast_construct_v4")


def _run_one(task):
    p, ctor, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(p)
    s = Settings()
    s.economics.driving_max_minutes = DRIVING_CAP_MIN
    s.economics.break_violation_penalty_per_min = BREAK_PENALTY_PER_MIN
    s_base = Settings()  # baseline -- gives cost without break penalty
    t0 = time.perf_counter()
    try:
        sol = pwm.solve_auto(
            inst, s, budget_seconds=budget, construction=ctor, seed=0,
        )
        wall = time.perf_counter() - t0
        cost_no = evaluate(inst, sol, s_base)["operational_cost"]
        cost_w = evaluate(inst, sol, s)["operational_cost"]
        return {
            "instance_id": inst.instance_id, "construction": ctor,
            "N": inst.num_customers,
            "K": int(sol.metrics["num_vehicles_used"]),
            "cost_no_break": cost_no,
            "cost_w_break": cost_w,
            "break_penalty": cost_w - cost_no,
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
    print(f"=== iter-7-v4-driver_breaks: {len(tasks)} solves "
          f"(2 ctors x 6 inst, driving_cap={DRIVING_CAP_MIN}min "
          f"${BREAK_PENALTY_PER_MIN}/min) ===", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            if r.get("ok"):
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['construction']:18s} "
                      f"{r['instance_id']:32s} K={r['K']:>3d} "
                      f"cost=${r['cost_w_break']:>8.1f} "
                      f"brk_pen=${r['break_penalty']:>7.1f} "
                      f"wall={r['wall_s']:>5.1f}s", flush=True)
            else:
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['construction']:18s} "
                      f"{r['instance_id']:32s} FAILED: "
                      f"{r.get('error', '?')[:100]}", flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], {})[r["construction"]] = r
    print("\n=== paired comparison vs pyvrp warmstart (cost under driver_breaks objective) ===")
    n_v4_wins = 0; total = 0.0; n = 0
    for iid in sorted(by):
        m = by[iid]
        if "pyvrp" not in m or "fast_construct_v4" not in m: continue
        py, v4 = m["pyvrp"], m["fast_construct_v4"]
        if not (py.get("ok") and v4.get("ok")): continue
        delta = py["cost_w_break"] - v4["cost_w_break"]
        if delta > 0: n_v4_wins += 1
        total += delta; n += 1
        print(f"  {iid:32s} pyvrp K={py['K']:>3d} ${py['cost_w_break']:>8.1f} "
              f"(brk ${py['break_penalty']:>5.0f})  "
              f"v4 K={v4['K']:>3d} ${v4['cost_w_break']:>8.1f} "
              f"(brk ${v4['break_penalty']:>5.0f})  delta={delta:+.1f}")
    if n > 0:
        print(f"\nverdict: v4 wins {n_v4_wins}/{n} on break-aware cost, "
              f"mean delta=${total/n:+.1f}/inst")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "driving_cap_min": DRIVING_CAP_MIN,
        "break_penalty_per_min": BREAK_PENALTY_PER_MIN,
        "rows": rows, "n_instances": n, "n_v4_wins": n_v4_wins,
        "mean_delta_v4_minus_pyvrp": total / max(1, n),
        "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
