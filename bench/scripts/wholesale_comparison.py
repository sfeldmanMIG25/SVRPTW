"""Wholesale solver comparison vs public solvers + known benchmarks.

Once Phase G's fast_construct_v2 lands and matches PyVRP-quality, this
script runs the production solver against every available baseline at
matched wall-time, on a stratified instance set spanning real-world
(OSM v1 + v1_large) and academic (Solomon, Homberger).

Output: per-solver leaderboard + per-instance breakdown, sorted by:
  - mean operational_cost (cost-only baseline)
  - mean unified score (cost + quality_index from iter-5f metrics suite)
  - wall_clock distribution (p50, p95)

Solvers wired (each via `--solvers` arg):
  greedy           - svrptw.solvers.classical.greedy
  regret_3         - svrptw.solvers.classical.regret_k k=3
  ortools          - svrptw.solvers.classical.ortools_solver (CP-SAT)
  lkh3             - svrptw.solvers.classical.lkh3 (LKH-3 wrapper)
  pyvrp            - svrptw.solvers.classical.pyvrp_solver (HGS)
  auction_gart     - svrptw.solvers.classical.auction_gart
  fast_construct   - svrptw.solvers.classical.fast_construct (v1)
  fast_construct_v2 - svrptw.solvers.classical.fast_construct_v2 (Phase G; once landed)
  solve_auto       - svrptw.solvers.classical.portfolio_pyvrp_warm.solve_auto (production)

Instance sets (each via `--instance-set`):
  v1_ood          - held-out OSM (I003+I004 across 8 cities x N=100/200/500)
  v1_leaderboard  - mixed (I000+I001 across 4 cities x N=50/100/200)
  v1_large        - large realistic (Phase G; N=500/1000 across 3 cities)
  solomon         - Solomon C/R/RC at N=100 (56 instances)
  homberger200    - Gehring-Homberger N=200 (24 instances)
  homberger400    - Gehring-Homberger N=400 (24 instances)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, "D:/SVRPTW")
import os
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")


# --- Solver registry --------------------------------------------------
# Each entry: (loader_callable, default_budget_seconds_for_N_50, scaling_with_N)
# `scaling_with_N` lets the harness pick a sane budget per N. Default is
# linear in N (so N=200 gets 4x the N=50 budget).

def _solve_greedy(inst, settings, budget):
    from svrptw.solvers.classical import greedy
    return greedy.solve(inst, settings)

def _solve_regret_3(inst, settings, budget):
    from svrptw.solvers.classical import regret_k
    return regret_k.solve(inst, settings, k=3)

def _solve_pyvrp(inst, settings, budget):
    from svrptw.solvers.classical import pyvrp_solver as pv
    return pv.solve(inst, settings, budget_seconds=budget)

def _solve_ortools(inst, settings, budget):
    from svrptw.solvers.classical import ortools_solver
    # API may differ; budget passed via settings or kwarg
    return ortools_solver.solve(inst, settings, budget_seconds=budget)

def _solve_lkh3(inst, settings, budget):
    from svrptw.solvers.classical import lkh3
    return lkh3.solve(inst, settings, budget_seconds=budget)

def _solve_auction_gart(inst, settings, budget):
    from svrptw.solvers.classical import auction_gart
    return auction_gart.solve(inst, settings, budget_seconds=budget)

def _solve_fast_construct(inst, settings, budget):
    from svrptw.solvers.classical import fast_construct
    return fast_construct.solve(inst, settings, budget_seconds=budget)

def _solve_fast_construct_v2(inst, settings, budget):
    # Phase G output -- may not exist yet; bench tolerates ImportError below.
    from svrptw.solvers.classical import fast_construct_v2
    return fast_construct_v2.solve(inst, settings, budget_seconds=budget)

def _solve_solve_auto(inst, settings, budget):
    from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
    return solve_auto(inst, settings, budget_seconds=budget)

_SOLVERS = {
    "greedy":            _solve_greedy,
    "regret_3":          _solve_regret_3,
    "pyvrp":             _solve_pyvrp,
    "ortools":           _solve_ortools,
    "lkh3":              _solve_lkh3,
    "auction_gart":      _solve_auction_gart,
    "fast_construct":    _solve_fast_construct,
    "fast_construct_v2": _solve_fast_construct_v2,
    "solve_auto":        _solve_solve_auto,
}


# --- Instance set registry --------------------------------------------

def _v1_ood_set() -> list[Path]:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix",
              "Charleston", "Austin", "Pittsburgh", "Cambridge"]
    out = []
    for N in (100, 200, 500):
        for c in cities:
            for r in ("I003", "I004"):
                p = Path(f"instances/v1/OSM-{c}-N{N:03d}-{r}.json")
                if p.exists(): out.append(p)
    return out

def _v1_leaderboard_set() -> list[Path]:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Charleston"]
    out = []
    for N in (50, 100, 200):
        for c in cities:
            for r in ("I000", "I001"):
                p = Path(f"instances/v1/OSM-{c}-N{N:03d}-{r}.json")
                if p.exists(): out.append(p)
    return out

def _v1_large_set() -> list[Path]:
    base = Path("instances/v1_large")
    if not base.exists(): return []
    return sorted(base.glob("OSM-*.json"))

def _solomon_set() -> list[Path]:
    return sorted(Path("instances/solomon").glob("*.txt"))

def _homberger200_set() -> list[Path]:
    return sorted(Path("instances/homberger/200").glob("*.txt"))

def _homberger400_set() -> list[Path]:
    return sorted(Path("instances/homberger/400").glob("*.txt"))

_SETS = {
    "v1_ood":         _v1_ood_set,
    "v1_leaderboard": _v1_leaderboard_set,
    "v1_large":       _v1_large_set,
    "solomon":        _solomon_set,
    "homberger200":   _homberger200_set,
    "homberger400":   _homberger400_set,
}


def _budget_for_N(N: int, base: float = 30.0) -> float:
    """Linear scaling so N=50 -> 7.5s, N=200 -> 30s, N=1000 -> 150s."""
    return max(2.0, base * (N / 200.0))


def _run_one(task: tuple[str, str, float]) -> dict[str, Any]:
    """Worker: load instance, run one solver, score, return record.

    Designed for ProcessPoolExecutor so each solver gets fresh CPU.
    """
    inst_path, solver_name, budget = task
    if inst_path.lower().endswith(".txt"):
        from svrptw.io.solomon import load_solomon
        inst = load_solomon(inst_path)
    else:
        from svrptw.io import load_instance
        inst = load_instance(inst_path)
    from svrptw.config import Settings
    from svrptw.metrics import score_solution
    settings = Settings()
    fn = _SOLVERS[solver_name]
    t0 = time.perf_counter()
    try:
        sol = fn(inst, settings, budget)
        wall = time.perf_counter() - t0
        score = score_solution(inst, sol)
        return {
            "instance_id": inst.instance_id,
            "instance_path": inst_path,
            "solver": solver_name,
            "budget_seconds": budget,
            "operational_cost": float(sol.metrics["operational_cost"]),
            "n_routes": int(sol.metrics["num_vehicles_used"]),
            "wall_clock_s": wall,
            "quality_index": score.quality_index,
            "inter_route_crossings": score.inter_route_crossings,
            "load_util_cv": score.load_util_cv,
            "mean_tw_buffer_score": score.mean_tw_buffer_score,
            "feasible": bool(sol.metrics.get("feasible", True)),
            "N": inst.num_customers,
        }
    except ImportError as e:
        return {"instance_id": getattr(inst, "instance_id", "?"),
                "solver": solver_name, "_failed": True,
                "_error": f"ImportError: {e}", "N": inst.num_customers}
    except Exception as e:
        return {"instance_id": getattr(inst, "instance_id", "?"),
                "solver": solver_name, "_failed": True,
                "_error": f"{type(e).__name__}: {e}",
                "N": inst.num_customers,
                "wall_clock_s": time.perf_counter() - t0}


def _summarize(rows: list[dict]) -> None:
    """Print per-solver leaderboard, sorted by mean unified score."""
    from collections import defaultdict
    by_solver = defaultdict(list)
    for r in rows:
        if r.get("_failed"): continue
        by_solver[r["solver"]].append(r)
    print("\n=== wholesale leaderboard (mean across all instances run) ===")
    print(f"  {'solver':22s} {'n':>3s} {'cost':>10s} {'q_idx':>6s} {'cross':>6s} {'wall_p50':>9s}")
    table = []
    for solver, rs in by_solver.items():
        n = len(rs)
        if n == 0: continue
        mean_cost = sum(r["operational_cost"] for r in rs) / n
        mean_q = sum(r["quality_index"] for r in rs) / n
        mean_x = sum(r["inter_route_crossings"] for r in rs) / n
        walls = sorted(r["wall_clock_s"] for r in rs)
        p50 = walls[len(walls)//2]
        table.append((mean_q + (1 - mean_cost / max(1.0, max(r["operational_cost"] for r in rs))) * 0.5,
                      solver, n, mean_cost, mean_q, mean_x, p50))
    table.sort(reverse=True)
    for unified, solver, n, mc, mq, mx, p50 in table:
        print(f"  {solver:22s} {n:>3d} {mc:>10.1f} {mq:>6.3f} {mx:>6.0f} {p50:>9.1f}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solvers", nargs="+", required=True,
                   choices=sorted(_SOLVERS.keys()))
    p.add_argument("--instance-set", required=True, choices=sorted(_SETS.keys()))
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of instances (debug)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget-base", type=float, default=30.0,
                   help="base wall budget at N=200; scales linearly with N")
    p.add_argument("--out", default="bench/runs/wholesale_comparison.json")
    args = p.parse_args()

    instances = _SETS[args.instance_set]()
    if args.limit:
        instances = instances[: args.limit]
    if not instances:
        print(f"no instances found for set={args.instance_set}")
        return 1

    print(f"[wholesale] {len(args.solvers)} solvers x {len(instances)} instances "
          f"= {len(args.solvers) * len(instances)} solves")

    # Build task list
    tasks = []
    for inst_path in instances:
        # peek at N from filename or load (cheap)
        try:
            if str(inst_path).lower().endswith(".txt"):
                from svrptw.io.solomon import load_solomon
                N = load_solomon(str(inst_path)).num_customers
            else:
                from svrptw.io import load_instance
                N = load_instance(str(inst_path)).num_customers
        except Exception:
            N = 200
        budget = _budget_for_N(N, base=args.budget_base)
        for s in args.solvers:
            tasks.append((str(inst_path), s, budget))


    rows: list[dict] = []
    t0 = time.perf_counter()
    try:
        from webui import client as _ui
        _ui.push_agent("wholesale", "running",
                       summary=f"{len(args.solvers)} solvers x {len(instances)} insts = {len(tasks)} solves")
    except Exception:
        _ui = None
    # iter-6a-cleanup-2: hard per-task timeout. The previous wholesale run hung
    # for 75min at 46/48 because two solver invocations exceeded their internal
    # budget (LKH-3 or PyVRP at N=1000). Wrap each future result fetch in a
    # timeout = budget * 5 (= 750s at N=1000) so a stuck task doesn't block
    # the whole bench. Failed tasks are marked _failed with _error="timeout".
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, t): (i, t) for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            i, t = futures[fut]
            try:
                # t is (inst_path, solver_name, budget); cap at 5x budget.
                hard_timeout = float(t[2]) * 5.0 + 60.0
                r = fut.result(timeout=hard_timeout)
                r["_task_index"] = i
                rows.append(r)
                done = len(rows)
                if r.get("_failed"):
                    print(f"  [{done:>3}/{len(tasks)}] FAILED {r['solver']:18s} on {r.get('instance_id','?')}: "
                          f"{r.get('_error','')[:60]}")
                else:
                    print(f"  [{done:>3}/{len(tasks)}] {r['solver']:18s} on {r['instance_id']:<32s} "
                          f"cost={r['operational_cost']:.1f} q={r['quality_index']:.3f} "
                          f"x={r['inter_route_crossings']:>3d} wall={r['wall_clock_s']:.1f}s")
                if _ui is not None:
                    _ui.push_progress("wholesale", done, len(tasks))
            except TimeoutError:
                # iter-6a-cleanup-2 hard timeout. Cancel the future, record
                # a failed row, continue with remaining tasks.
                fut.cancel()
                rows.append({"_task_index": i, "_failed": True,
                             "_error": f"hard_timeout after {hard_timeout:.0f}s",
                             "solver": t[1], "instance_path": t[0]})
                print(f"  [task {i}] TIMEOUT {t[1]} on {Path(t[0]).stem} (>{hard_timeout:.0f}s)")
            except Exception as e:
                rows.append({"_task_index": i, "_failed": True, "_error": str(e),
                             "solver": t[1], "instance_path": t[0]})
                print(f"  [task {i}] EXCEPT: {e}")

    rows.sort(key=lambda r: r.get("_task_index", 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {args.out} in {time.perf_counter()-t0:.1f}s")
    _summarize(rows)
    if _ui is not None:
        _ui.push_agent("wholesale", "completed",
                       summary=f"{len(rows)} solves done; see {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
