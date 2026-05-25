"""v1 leaderboard rerun via solve_auto (2-tier dispatch + cb-scaling).

This is the Phase A3 bench from the 2026-05-15 epic. It supersedes
``auto_dispatch_headline.py`` which:
  - tested only I=000 (in-distribution, single rep)
  - was single-threaded
  - had no live UI
  - predates cb-scaling (no Homberger N=400 fix)

This script runs solve_auto across N=50/100/200/500 on multiple reps
per city, parallelized, with live progress + per-snapshot push to
the web UI when --webui is set.

Output: bench/runs/v1_leaderboard_solve_auto.json plus a Pareto-style
summary printed to stdout (wins / losses / mean delta per N).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import pyvrp_solver as pv
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto


def _run_pair(task: tuple[str, int, float, float, str]) -> dict[str, Any]:
    inst_path, N, b_auto, b_pyvrp, rep_tag = task
    inst = load_instance(inst_path)
    s = Settings()
    on_accept = None
    import os as _os
    if _os.environ.get("SVRPTW_WEBUI_URL"):
        try:
            from webui.snapshot import make_on_accept
            stream = f"v1-{rep_tag}-N{N:03d}-{Path(inst_path).stem}"
            on_accept = make_on_accept(inst, stream=stream,
                                        every_n_accepts=5,
                                        min_improvement=5.0)
        except Exception:
            on_accept = None
    auto_sol = solve_auto(inst, s, budget_seconds=b_auto, on_accept=on_accept)
    pyvrp_sol = pv.solve(inst, s, budget_seconds=b_pyvrp)
    return {
        "instance_path": inst_path,
        "instance_id": inst.instance_id,
        "N": N,
        "rep": rep_tag,
        "in_distribution": rep_tag == "I000",
        "auto_dispatched_to": auto_sol.solver,
        "auto_cost": float(auto_sol.metrics["operational_cost"]),
        "pyvrp_cost": float(pyvrp_sol.metrics["operational_cost"]),
        "delta": float(pyvrp_sol.metrics["operational_cost"]
                       - auto_sol.metrics["operational_cost"]),
        "auto_routes": int(auto_sol.metrics.get("num_vehicles_used", -1)),
        "pyvrp_routes": int(pyvrp_sol.metrics.get("num_vehicles_used", -1)),
        "budget_auto": b_auto,
        "budget_pyvrp": b_pyvrp,
    }


_BUDGETS = {50: 10.0, 100: 15.0, 200: 30.0, 500: 60.0}
_PYVRP_BUDGETS = {50: 30.0, 100: 45.0, 200: 60.0, 500: 120.0}
_DEFAULT_REPS = ("I000", "I001", "I002")
_DEFAULT_CITIES = ("Manhattan", "Paris", "SanFrancisco", "Phoenix",
                   "Charleston", "Austin", "Pittsburgh", "Cambridge")


def _build_tasks(N_list: tuple[int, ...], cities: tuple[str, ...],
                  reps: tuple[str, ...]) -> list[tuple]:
    out: list[tuple] = []
    for N in N_list:
        b = _BUDGETS[N]; pb = _PYVRP_BUDGETS[N]
        for c in cities:
            for r in reps:
                ip = Path(f"instances/v1/OSM-{c}-N{N:03d}-{r}.json")
                if ip.exists():
                    out.append((str(ip), N, b, pb, r))
    return out


def _push(stage: str, msg: str = "", **extra: Any) -> None:
    try:
        from webui import client as ui
        ui.push_stage(stage, msg, **extra)
    except Exception:
        pass


def _push_progress(name: str, completed: int, total: int,
                    eta_s: float | None = None) -> None:
    try:
        from webui import client as ui
        ui.push_progress(name, completed, total, eta_seconds=eta_s)
    except Exception:
        pass


def _summarize(rows: list[dict[str, Any]]) -> None:
    """Pareto-style summary: per-N and per-(N, in-distribution) breakdown."""
    from collections import defaultdict
    by_N: dict[int, list[dict]] = defaultdict(list)
    by_N_id: dict[tuple[int, bool], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("_failed"):
            continue
        by_N[r["N"]].append(r)
        by_N_id[(r["N"], r["in_distribution"])].append(r)

    print("\n[v1_leaderboard_solve_auto headline by N]")
    print(f"  {'N':>4}  {'budget':>7}  {'n':>3}  {'W%':>5} {'L%':>5}  {'mean':>9}  dispatched")
    for N in sorted(by_N):
        rs = by_N[N]; n = len(rs)
        deltas = [r["delta"] for r in rs]
        w = sum(1 for d in deltas if d > 1.0)
        l = sum(1 for d in deltas if d < -1.0)
        wpct = 100.0 * w / n; lpct = 100.0 * l / n
        disp = ", ".join(sorted({r["auto_dispatched_to"] for r in rs}))
        print(f"  {N:>4}  {_BUDGETS[N]:>7.0f}  {n:>3}  {wpct:>5.1f} {lpct:>5.1f}  "
              f"{sum(deltas)/n:+9.2f}  {disp}")

    print("\n[in-distribution vs held-out split]")
    print(f"  {'N':>4}  {'set':>9}  {'n':>3}  {'W%':>5}  {'mean':>+9}")
    for (N, in_dist) in sorted(by_N_id):
        rs = by_N_id[(N, in_dist)]; n = len(rs)
        deltas = [r["delta"] for r in rs]
        w = sum(1 for d in deltas if d > 1.0)
        tag = "in-dist" if in_dist else "held-out"
        print(f"  {N:>4}  {tag:>9}  {n:>3}  {100.0*w/n:>5.1f}  {sum(deltas)/n:+9.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--webui", action="store_true",
                    help="Push live progress to webui.event_bus")
    ap.add_argument("--N", nargs="+", type=int, default=[50, 100, 200, 500],
                    help="Customer counts to bench (default: all 4)")
    ap.add_argument("--reps", nargs="+", default=list(_DEFAULT_REPS),
                    help="Instance reps to include (default: I000 I001 I002)")
    ap.add_argument("--cities", nargs="+", default=list(_DEFAULT_CITIES))
    ap.add_argument("--out", default="bench/runs/v1_leaderboard_solve_auto.json")
    args = ap.parse_args()

    tasks = _build_tasks(tuple(args.N), tuple(args.cities), tuple(args.reps))
    if not tasks:
        print("No tasks built. Check that instance files exist on disk.")
        return 1

    print(f"[v1_leaderboard_solve_auto] {len(tasks)} tasks "
          f"(N={args.N}, reps={args.reps}, workers={args.workers})")
    _push("v1_leaderboard", f"{len(tasks)} tasks queued", n_tasks=len(tasks))
    _push_progress("v1_leaderboard_solve_auto", 0, len(tasks))


    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_pair, t): (i, t)
                    for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            i, t = futures[fut]
            try:
                r = fut.result()
                r["_task_index"] = i
                r["_elapsed_s"] = time.perf_counter() - t0
                results.append(r)
                done = len(results)
                eta = ((time.perf_counter() - t0) / done) * (len(tasks) - done)
                print(f"  [{done:>3}/{len(tasks)}] N={r['N']:>3} "
                      f"{r['instance_id']:<32}  "
                      f"auto={r['auto_cost']:.1f}  pyvrp={r['pyvrp_cost']:.1f}  "
                      f"delta={r['delta']:+.1f}  ({r['_elapsed_s']:.0f}s)")
                sys.stdout.flush()
                if args.webui:
                    _push_progress("v1_leaderboard_solve_auto", done,
                                   len(tasks), eta_s=eta)
                    _push("v1_leaderboard",
                          f"{done}/{len(tasks)}, last delta={r['delta']:+.1f}",
                          last_instance=r["instance_id"])
            except Exception as e:
                print(f"  [task {i}] FAILED: {e}  task={t}")
                results.append({"_task_index": i, "_failed": True,
                                 "_error": str(e), "task": str(t)})

    results.sort(key=lambda r: r.get("_task_index", 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {args.out} "
          f"in {time.perf_counter()-t0:.1f}s")
    _summarize(results)
    _push("idle", f"v1_leaderboard finished, {len(results)} results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
