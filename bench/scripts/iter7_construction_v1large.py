"""iter-7 construction comparison: pyvrp_warm vs fast_construct vs
fast_construct_v2 on v1_large at N=500 + N=1000.

Question: do we have a viable PyVRP-independent construction at scale?
The user's original ask: "make our own construction process instead of
relying on pyvrp ... use items that are readily available like the dual
graph to get clustering rather than trying to translate to euclidean
and rendering ... handle multiple 1000 customer problems at speed."

This is a CONSTRUCTION-ONLY bench (no portfolio refinement). Each builder
gets the same wall budget; we report cost, K, feasibility, and wall to
characterize each option's scale-vs-quality trade-off.

Output: bench/runs/iter7_construction_v1large.json
"""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = "bench/runs/iter7_construction_v1large.json"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 8.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 8.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 8.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 16.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 16.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 16.0),
]
BUILDERS = ("pyvrp_warm", "fast_construct", "fast_construct_v2", "fast_construct_v3", "fast_construct_v4")


def _build_one(builder: str, inst, settings, budget: float, seed: int = 0):
    """Run one construction. Returns (sol, wall_s)."""
    from svrptw.solvers.common.solution import evaluate
    t0 = time.perf_counter()
    if builder == "pyvrp_warm":
        from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm
        # solve_auto with budget == construction budget = construction-only
        sol = pwm.solve_auto(
            inst, settings,
            budget_seconds=budget,
            pyvrp_construction_budget=budget,
            seed=seed,
        )
    elif builder == "fast_construct":
        from svrptw.solvers.classical import fast_construct as fc
        sol = fc.solve(inst, settings, budget_seconds=budget, seed=seed)
    elif builder == "fast_construct_v2":
        from svrptw.solvers.classical import fast_construct_v2 as fc2
        sol = fc2.solve(inst, settings, budget_seconds=budget, seed=seed)
    elif builder == "fast_construct_v3":
        from svrptw.solvers.classical import fast_construct_v3 as fc3
        sol = fc3.solve(inst, settings, budget_seconds=budget, seed=seed)
    elif builder == "fast_construct_v4":
        from svrptw.solvers.classical import fast_construct_v4 as fc4
        sol = fc4.solve(inst, settings, budget_seconds=budget, seed=seed)
    else:
        raise ValueError(f"unknown builder: {builder}")
    wall = time.perf_counter() - t0
    # Evaluate with default Settings to get a clean cost number
    m = evaluate(inst, sol, settings)
    return sol, wall, m


def _run_one(task):
    p, builder, budget = task
    from svrptw.config import Settings
    from svrptw.io import load_instance
    inst = load_instance(p)
    s = Settings()
    try:
        sol, wall, m = _build_one(builder, inst, s, budget)
        return {
            "instance_id": inst.instance_id, "builder": builder,
            "N": inst.num_customers,
            "K": int(m["num_vehicles_used"]),
            "cost": float(m["operational_cost"]),
            "feasible": bool(m["feasible"]),
            "wall_s": wall, "budget_s": budget,
            "ok": True,
        }
    except Exception as e:
        return {
            "instance_id": inst.instance_id, "builder": builder,
            "N": inst.num_customers, "budget_s": budget,
            "wall_s": 0.0, "ok": False, "error": str(e)[:200],
        }


def main():
    tasks = [(p, b, bud) for p, bud in INSTANCES for b in BUILDERS]
    print(f"=== iter-7 construction bench: {len(tasks)} builds "
          f"({len(BUILDERS)} builders x {len(INSTANCES)} instances) ===")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            if r.get("ok"):
                print(f"  [{len(rows):>2}/{len(tasks)}] "
                      f"{r['builder']:18s} {r['instance_id']:32s} "
                      f"K={r['K']:>3d} cost=${r['cost']:>9.1f} "
                      f"feas={int(r['feasible'])} wall={r['wall_s']:>5.2f}s",
                      flush=True)
            else:
                print(f"  [{len(rows):>2}/{len(tasks)}] "
                      f"{r['builder']:18s} {r['instance_id']:32s} "
                      f"FAILED: {r.get('error', '?')[:120]}",
                      flush=True)
    # Paired comparison: per-instance, fc/fc_v2 cost relative to pyvrp_warm
    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], {})[r["builder"]] = r
    print("\n=== paired comparison vs pyvrp_warm (cost_x = builder_cost / pyvrp_cost) ===")
    header = f"  {'instance':32s} {'N':>5s}"
    for b in BUILDERS:
        header += f"  {b[:14]:>14s}_K {'cost_x':>7s} {'wall':>6s}"
    print(header)
    summary = {b: {"n_ok": 0, "n_feas": 0, "cost_sum": 0.0, "wall_sum": 0.0}
               for b in BUILDERS}
    n_inst = 0
    for iid in sorted(by):
        m = by[iid]
        if not all(b in m for b in BUILDERS):
            continue
        if not all(m[b].get("ok") for b in BUILDERS):
            continue
        n_inst += 1
        py_cost = m["pyvrp_warm"]["cost"]
        for b in BUILDERS:
            r = m[b]
            summary[b]["n_ok"] += 1
            if r.get("feasible"): summary[b]["n_feas"] += 1
            summary[b]["cost_sum"] += r["cost"]
            summary[b]["wall_sum"] += r["wall_s"]
        line = f"  {iid:32s} {m['pyvrp_warm']['N']:>5d}"
        for b in BUILDERS:
            r = m[b]
            cost_x = r["cost"] / py_cost if py_cost > 0 else float("nan")
            line += f"  K={r['K']:>3d} {cost_x:>6.2f}x {r['wall_s']:>5.1f}s"
        print(line)
    if n_inst > 0:
        print(f"\n=== aggregates (n={n_inst} instances) ===")
        for b in BUILDERS:
            s = summary[b]
            print(f"  {b:18s} mean cost=${s['cost_sum']/n_inst:>9.1f} "
                  f"mean wall={s['wall_sum']/n_inst:>5.2f}s "
                  f"feas={s['n_feas']}/{s['n_ok']}")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "rows": rows, "n_instances": n_inst,
        "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
