"""iter-7-v4-embargo-multiseed: 3-seed stability check on the +$178/inst
embargo headline. Tests two representative instances (Paris-N500 and
Paris-N1000 -- where v4 won by largest margins in the single-seed bench)
at seeds 0, 1, 2 for both pyvrp and v4 warmstarts. Reports mean +/- std
per instance per construction.

Question: is the +$178/inst mean delta robust across seeds, or was it
a single-seed artifact?
"""
from __future__ import annotations
import json, time, statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = "bench/runs/iter7v4_embargo_multiseed.json"
# Two representative instances; 3 seeds each; 2 constructions = 12 solves
TASKS = []
for inst_path, budget in [
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
]:
    for seed in (0, 1, 2):
        for ctor in ("pyvrp", "fast_construct_v4"):
            TASKS.append((inst_path, ctor, seed, budget))
EMBARGO_STARTS = (480, 720)
EMBARGO_ENDS = (510, 750)
EMBARGO_PEN = 50.0


def _run_one(task):
    p, ctor, seed, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(p)
    s = Settings()
    s.economics.embargo_window_starts = EMBARGO_STARTS
    s.economics.embargo_window_ends = EMBARGO_ENDS
    s.economics.embargo_violation_penalty_per_visit = EMBARGO_PEN
    t0 = time.perf_counter()
    try:
        sol = pwm.solve_auto(
            inst, s, budget_seconds=budget, construction=ctor, seed=seed,
        )
        wall = time.perf_counter() - t0
        cost_w = evaluate(inst, sol, s)["operational_cost"]
        return {
            "instance_id": inst.instance_id, "construction": ctor, "seed": seed,
            "N": inst.num_customers,
            "K": int(sol.metrics["num_vehicles_used"]),
            "cost_w_embargo": cost_w,
            "feasible": bool(sol.metrics["feasible"]),
            "wall_s": wall, "budget_s": budget, "ok": True,
        }
    except Exception as e:
        return {
            "instance_id": inst.instance_id, "construction": ctor, "seed": seed,
            "ok": False, "error": str(e)[:200],
            "wall_s": time.perf_counter() - t0,
        }


def main():
    print(f"=== iter-7-v4-embargo-multiseed: {len(TASKS)} solves "
          f"(2 instances x 3 seeds x 2 constructions, embargo ON) ===",
          flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_run_one, t): t for t in TASKS}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            if r.get("ok"):
                print(f"  [{len(rows):>2}/{len(TASKS)}] "
                      f"{r['construction']:18s} {r['instance_id']:32s} "
                      f"seed={r['seed']} K={r['K']:>3d} "
                      f"cost=${r['cost_w_embargo']:>8.1f} wall={r['wall_s']:>5.1f}s",
                      flush=True)
            else:
                print(f"  [{len(rows):>2}/{len(TASKS)}] {r['construction']:18s} "
                      f"{r['instance_id']:32s} seed={r['seed']} FAILED: "
                      f"{r.get('error', '?')[:100]}", flush=True)
    # Aggregate per (instance, construction): mean +/- std across seeds
    print("\n=== per-instance stability ===")
    print(f"  {'instance':32s} {'ctor':18s} "
          f"{'seeds':>20s} {'mean':>10s} {'std':>8s}")
    grouped = {}
    for r in rows:
        if not r.get("ok"): continue
        key = (r["instance_id"], r["construction"])
        grouped.setdefault(key, []).append(r)
    for (iid, ctor), rs in sorted(grouped.items()):
        costs = [x["cost_w_embargo"] for x in sorted(rs, key=lambda x: x["seed"])]
        mean = statistics.mean(costs)
        std = statistics.stdev(costs) if len(costs) >= 2 else 0.0
        costs_str = "[" + ", ".join(f"${c:.0f}" for c in costs) + "]"
        print(f"  {iid:32s} {ctor:18s} {costs_str:>20s} ${mean:>8.1f} ${std:>6.1f}")
    # Paired analysis: v4-vs-pyvrp per seed per instance
    print("\n=== paired comparison per seed ===")
    n_v4_wins = 0; n_total = 0; total_delta = 0.0
    by = {}
    for r in rows:
        if not r.get("ok"): continue
        by.setdefault((r["instance_id"], r["seed"]), {})[r["construction"]] = r
    for (iid, seed), m in sorted(by.items()):
        if "pyvrp" not in m or "fast_construct_v4" not in m: continue
        py, v4 = m["pyvrp"], m["fast_construct_v4"]
        delta = py["cost_w_embargo"] - v4["cost_w_embargo"]
        if delta > 0: n_v4_wins += 1
        total_delta += delta; n_total += 1
        verdict = "v4 wins" if delta > 0 else "pyvrp wins"
        print(f"  {iid:32s} seed={seed} pyvrp=${py['cost_w_embargo']:.0f} "
              f"v4=${v4['cost_w_embargo']:.0f} delta={delta:+.0f} ({verdict})")
    if n_total > 0:
        print(f"\nverdict: v4 wins {n_v4_wins}/{n_total} pairs, "
              f"mean delta=${total_delta/n_total:+.1f}/inst")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "embargo_starts": list(EMBARGO_STARTS),
        "embargo_ends": list(EMBARGO_ENDS),
        "embargo_pen": EMBARGO_PEN,
        "rows": rows, "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
