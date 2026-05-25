"""cb-scaled rebench: validate the N-aware construction-budget fix.

This is the Phase A2 bench from the 2026-05-15 epic. It tests whether
``solve_auto`` (which now applies ``scaled_cb(N)`` linearly above
N=200) restores parity at Homberger N=400 where fixed cb=8s lost
17/24 vs PyVRP@120s in the prior session.

Coverage:
  - v1 OOD held-out: I=003/I=004 across 8 cities at N=100/200/500
  - Homberger N=400: all 24 academic instances

For each instance, runs ``solve_auto(...)`` vs ``PyVRP@2x-budget``.
Results dumped to ``bench/runs/cb_scaled_rebench.json``.

Pass ``--webui`` to stream progress + per-N completion to the live UI
(start ``python -m webui.app`` first, then load ``http://127.0.0.1:8765``).
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


# Module-level worker -- ProcessPoolExecutor needs picklable functions.
def _run_pair(task: tuple[str, int, float, float]) -> dict[str, Any]:
    inst_path, N, b_auto, b_pyvrp = task
    # Dispatch by extension: .txt = Solomon/Homberger native; .json = our format
    if inst_path.lower().endswith(".txt"):
        from svrptw.io.solomon import load_solomon
        inst = load_solomon(inst_path)
    else:
        inst = load_instance(inst_path)
    s = Settings()
    # Live web-UI hook: if the parent set SVRPTW_WEBUI_URL the worker
    # inherits it and pushes a rate-limited snapshot stream during the
    # bandit phase. Otherwise on_accept stays None (zero overhead).
    on_accept = None
    import os as _os
    if _os.environ.get("SVRPTW_WEBUI_URL"):
        try:
            from webui.snapshot import make_on_accept
            stream = f"warm-N{N:03d}-{Path(inst_path).stem}"
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


def _v1_ood_tasks(budgets: dict) -> list[tuple]:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix",
              "Charleston", "Austin", "Pittsburgh", "Cambridge"]
    reps = ["I003", "I004"]
    out: list[tuple] = []
    for N in (100, 200, 500):
        b = budgets["v1"][N]
        pb = budgets["v1"][f"pyvrp_{N}"]
        for c in cities:
            for r in reps:
                ip = Path(f"instances/v1/OSM-{c}-N{N:03d}-{r}.json")
                if ip.exists():
                    out.append((str(ip), N, b, pb))
    return out


def _homberger_n400_tasks(budgets: dict) -> list[tuple]:
    out: list[tuple] = []
    base = Path("instances/homberger/400")
    if not base.exists():
        return out
    b = budgets["homberger"][400]
    pb = budgets["homberger"]["pyvrp_400"]
    # Homberger native format is .txt; load_instance handles both via the
    # solomon loader's Homberger compatibility path.
    patterns = ("*.json", "*.txt")
    seen: set[str] = set()
    for pat in patterns:
        for f in sorted(base.glob(pat)):
            stem = f.stem
            if stem in seen:
                continue
            seen.add(stem)
            out.append((str(f), 400, b, pb))
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
    from collections import defaultdict
    by_N: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("_failed"):
            continue
        by_N[r["N"]].append(r)
    print("\n[cb_scaled_rebench headline by N]")
    print(f"  {'N':>4}  {'n':>3}  {'W':>3} {'L':>3}  {'mean_delta':>12}  dispatched")
    for N in sorted(by_N):
        rs = by_N[N]
        n = len(rs)
        deltas = [r["delta"] for r in rs]
        w = sum(1 for d in deltas if d > 1.0)
        l = sum(1 for d in deltas if d < -1.0)
        disp = ", ".join(sorted({r["auto_dispatched_to"] for r in rs}))
        print(f"  {N:>4}  {n:>3}  {w:>3} {l:>3}  {sum(deltas)/n:+12.2f}  {disp}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--webui", action="store_true",
                    help="Push live progress to webui.event_bus")
    ap.add_argument("--include", nargs="+",
                    default=["v1_ood", "homberger_n400"],
                    choices=["v1_ood", "homberger_n400"])
    ap.add_argument("--out", default="bench/runs/cb_scaled_rebench.json")
    args = ap.parse_args()

    budgets = {
        "v1": {100: 15.0, "pyvrp_100": 45.0,
               200: 30.0, "pyvrp_200": 60.0,
               500: 60.0, "pyvrp_500": 120.0},
        "homberger": {400: 60.0, "pyvrp_400": 120.0},
    }

    tasks: list[tuple] = []
    if "v1_ood" in args.include:
        tasks += _v1_ood_tasks(budgets)
    if "homberger_n400" in args.include:
        tasks += _homberger_n400_tasks(budgets)

    if not tasks:
        print("No tasks selected. Check that instance files exist on disk.")
        return 1

    print(f"[cb_scaled_rebench] {len(tasks)} tasks across "
          f"{len(args.include)} sets, workers={args.workers}")
    _push("cb_rebench", f"{len(tasks)} tasks queued", n_tasks=len(tasks))
    _push_progress("cb_scaled_rebench", 0, len(tasks))


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
                line = (f"  [{done:>3}/{len(tasks)}] "
                        f"N={r['N']:>3} {r['instance_id']:<32}  "
                        f"auto={r['auto_cost']:.1f}  pyvrp={r['pyvrp_cost']:.1f}  "
                        f"delta={r['delta']:+.1f}  ({r['_elapsed_s']:.0f}s, eta {eta:.0f}s)")
                print(line)
                sys.stdout.flush()
                if args.webui:
                    _push_progress("cb_scaled_rebench", done, len(tasks),
                                   eta_s=eta)
                    _push("cb_rebench",
                          f"{done}/{len(tasks)} done, last delta={r['delta']:+.1f}",
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
    _push("idle", f"cb_rebench finished, {len(results)} results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
