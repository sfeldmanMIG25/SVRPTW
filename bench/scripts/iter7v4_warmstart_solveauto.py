"""iter-7-bis-v4 warmstart comparison at solve_auto level.

Earlier we showed v4 standalone beats pyvrp on construction-only:
  $1,004 vs $1,034 mean cost, 3.28s vs 12.59s mean wall.

But a single-instance smoke (Manhattan-N500 b=75s) showed pyvrp warmstart
beats v4 warmstart at solve_auto level: $767.8 vs $846.4 (pyvrp +9% better).
This bench confirms whether that holds across all 6 v1_large instances or
whether some instances actually favor v4.

Two-builder x 6-instance = 12 solves. Each uses the same per-N budget
that the README specifies (N=500 -> 75s, N=1000 -> 150s).
"""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = "bench/runs/iter7v4_warmstart_solveauto.json"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]
BUILDERS = ("pyvrp", "fast_construct_v4")


def _run_one(task):
    p, builder, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm
    inst = load_instance(p)
    s = Settings()
    t0 = time.perf_counter()
    try:
        sol = pwm.solve_auto(
            inst, s, budget_seconds=budget, construction=builder, seed=0,
        )
        wall = time.perf_counter() - t0
        return {
            "instance_id": inst.instance_id, "builder": builder,
            "N": inst.num_customers,
            "K": int(sol.metrics["num_vehicles_used"]),
            "cost": float(sol.metrics["operational_cost"]),
            "feasible": bool(sol.metrics["feasible"]),
            "wall_s": wall, "budget_s": budget, "ok": True,
        }
    except Exception as e:
        return {
            "instance_id": inst.instance_id, "builder": builder,
            "N": inst.num_customers, "budget_s": budget,
            "wall_s": time.perf_counter() - t0, "ok": False,
            "error": str(e)[:200],
        }


def main():
    tasks = [(p, b, bud) for p, bud in INSTANCES for b in BUILDERS]
    print(f"=== iter-7-v4 warmstart vs pyvrp warmstart at solve_auto, "
          f"{len(tasks)} solves ===", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            if r.get("ok"):
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['builder']:18s} "
                      f"{r['instance_id']:32s} K={r['K']:>3d} "
                      f"cost=${r['cost']:>8.1f} wall={r['wall_s']:>5.1f}s",
                      flush=True)
            else:
                print(f"  [{len(rows):>2}/{len(tasks)}] {r['builder']:18s} "
                      f"{r['instance_id']:32s} FAILED: "
                      f"{r.get('error', '?')[:120]}", flush=True)
    # Paired comparison
    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], {})[r["builder"]] = r
    print("\n=== paired comparison vs pyvrp warmstart at solve_auto ===")
    n_v4_wins = 0; total = 0.0; n = 0
    cost_pyvrp_sum = 0.0; cost_v4_sum = 0.0
    wall_pyvrp_sum = 0.0; wall_v4_sum = 0.0
    for iid in sorted(by):
        m = by[iid]
        if "pyvrp" not in m or "fast_construct_v4" not in m: continue
        py, v4 = m["pyvrp"], m["fast_construct_v4"]
        if not (py.get("ok") and v4.get("ok")): continue
        delta = py["cost"] - v4["cost"]  # positive = v4 wins
        if delta > 0: n_v4_wins += 1
        total += delta; n += 1
        cost_pyvrp_sum += py["cost"]; cost_v4_sum += v4["cost"]
        wall_pyvrp_sum += py["wall_s"]; wall_v4_sum += v4["wall_s"]
        ratio = v4["cost"] / py["cost"]
        print(f"  {iid:32s} pyvrp K={py['K']:>3d} ${py['cost']:>8.1f} "
              f"({py['wall_s']:>5.1f}s)  v4 K={v4['K']:>3d} "
              f"${v4['cost']:>8.1f} ({v4['wall_s']:>5.1f}s)  "
              f"ratio={ratio:.3f}x  delta={delta:+.1f}")
    if n > 0:
        print(f"\nverdict: v4 wins {n_v4_wins}/{n} on cost, "
              f"mean delta=${total/n:+.1f}/inst")
        print(f"mean pyvrp cost=${cost_pyvrp_sum/n:.1f}, "
              f"mean v4 cost=${cost_v4_sum/n:.1f}")
        print(f"mean pyvrp wall={wall_pyvrp_sum/n:.1f}s, "
              f"mean v4 wall={wall_v4_sum/n:.1f}s")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "rows": rows, "n_instances": n, "n_v4_wins": n_v4_wins,
        "mean_delta_v4_minus_pyvrp": total / max(1, n),
        "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
