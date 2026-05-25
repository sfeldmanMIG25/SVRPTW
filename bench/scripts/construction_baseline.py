"""Construction-only baseline bench (Phase E1).

Runs each construction option in isolation (NO bandit refinement) on a
stratified instance set, then reports a per-N basin-quality leaderboard
keyed off PyVRP@8s as the reference (basin_quality = cost / pyvrp_8_cost,
lower is better).

Constructions covered:
  greedy, regret_k_{1,3,5}, auction_gart_{0p5,2,8}, pyvrp_{1,2,4,8,16}.

Stratified set (16 total):
  v1 N=100/200/500: I003 across {Manhattan, Paris, SanFrancisco, Charleston}
  Solomon: C101, R101, RC101, R201

Output: bench/runs/construction_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Allow `python bench/scripts/construction_baseline.py` invocation in
# addition to `python -m bench.scripts.construction_baseline`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.io.solomon import load_solomon
from bench.parallel import map_instances


# ---- Webui hooks (best-effort; never block) -------------------------------

def _push_agent(name: str, status: str, summary: str = "", **extra: Any) -> None:
    try:
        from webui import client as ui
        ui.push_agent(name, status, summary=summary, **extra)
    except Exception:
        pass


def _push_progress(name: str, completed: int, total: int,
                   eta_s: float | None = None) -> None:
    try:
        from webui import client as ui
        ui.push_progress(name, completed, total, eta_seconds=eta_s)
    except Exception:
        pass


def _push_log(line: str) -> None:
    try:
        from webui import client as ui
        ui.push_log(line)
    except Exception:
        pass


# ---- Construction registry ------------------------------------------------

CONSTRUCTIONS = [
    "greedy",
    "regret_k_1", "regret_k_3", "regret_k_5",
    "auction_gart_0p5", "auction_gart_2", "auction_gart_8",
    "pyvrp_1", "pyvrp_2", "pyvrp_4", "pyvrp_8", "pyvrp_16",
]


def _run_construction(name: str, inst, settings):
    """Dispatch one construction by name. Returns the resulting Solution."""
    if name == "greedy":
        from svrptw.solvers.classical import greedy as g
        return g.solve(inst, settings)
    if name.startswith("regret_k_"):
        k = int(name.split("_")[-1])
        from svrptw.solvers.classical import regret_k as rk
        # budget_seconds=0 disables the LS improvement chain so we
        # measure the construction itself.
        return rk.solve(inst, settings, budget_seconds=0.0, k=k)
    if name.startswith("auction_gart_"):
        tag = name[len("auction_gart_"):]
        budget = float(tag.replace("p", "."))
        from svrptw.solvers.classical import auction_gart as ag
        return ag.solve(inst, settings, budget_seconds=budget)
    if name.startswith("pyvrp_"):
        budget = float(name[len("pyvrp_"):])
        from svrptw.solvers.classical import pyvrp_solver as pv
        return pv.solve(inst, settings, budget_seconds=budget)
    raise ValueError(f"unknown construction: {name}")


# ---- Worker (top-level so ProcessPoolExecutor can pickle it) --------------

def _row(task: tuple) -> dict[str, Any]:
    """Pickle-safe worker. task = (construction, instance_path, instance_kind)."""
    construction, inst_path, kind = task
    s = Settings()
    if kind == "solomon":
        inst = load_solomon(inst_path)
    else:
        inst = load_instance(inst_path)
    t0 = time.perf_counter()
    sol = _run_construction(construction, inst, s)
    wall = time.perf_counter() - t0
    return {
        "construction": construction,
        "instance_id": inst.instance_id,
        "instance_path": str(inst_path),
        "instance_kind": kind,
        "N": int(inst.num_customers),
        "cost": float(sol.metrics.get("operational_cost", float("inf"))),
        "n_routes": int(sol.metrics.get("num_vehicles_used", -1)),
        "missed": float(sol.metrics.get("missed_deliveries", -1.0)),
        "feasible": bool(sol.metrics.get("feasible", 0.0)),
        "wall_s": float(wall),
    }


# ---- Task building --------------------------------------------------------

def _v1_tasks(N_set: list[int], cities: list[str]) -> list[tuple]:
    """Pick (city, N, I003) instances that exist on disk."""
    out: list[tuple] = []
    base = Path("instances/v1")
    for N in N_set:
        for c in cities:
            ip = base / f"OSM-{c}-N{N:03d}-I003.json"
            if ip.exists():
                out.append((str(ip), "v1", N))
    return out


def _solomon_tasks(names: list[str]) -> list[tuple]:
    out: list[tuple] = []
    base = Path("instances/solomon")
    for n in names:
        ip = base / f"{n}.txt"
        if ip.exists():
            out.append((str(ip), "solomon", -1))  # N populated by worker
    return out


def _build_task_list(args) -> list[tuple]:
    cities = ["Manhattan", "Paris", "SanFrancisco", "Charleston"]
    solomon = ["C101", "R101", "RC101", "R201"]
    instances: list[tuple] = []
    if args.instances_only in (None, "all", "v1"):
        instances += _v1_tasks([100, 200, 500], cities)
    if args.instances_only in (None, "all", "solomon"):
        instances += _solomon_tasks(solomon)
    if args.instances_only == "smoke":
        # 1 instance only, used by the verify step
        instances += _solomon_tasks(["C101"])
    # Cross with constructions
    constructions = (CONSTRUCTIONS if not args.constructions_only
                     else [c for c in CONSTRUCTIONS
                           if c in set(args.constructions_only)])
    out: list[tuple] = []
    for inst_path, kind, _N in instances:
        for c in constructions:
            out.append((c, inst_path, kind))
    return out


# ---- Reporting ------------------------------------------------------------

def _summarize(rows: list[dict[str, Any]]) -> str:
    """Per-N basin-quality leaderboard against pyvrp_8."""
    lines: list[str] = []
    by_N: dict[int, list[dict]] = {}
    for r in rows:
        if r.get("_failed"):
            continue
        by_N.setdefault(int(r["N"]), []).append(r)

    for N in sorted(by_N):
        group = by_N[N]
        # Per-instance pyvrp_8 reference cost.
        pv8_by_inst: dict[str, float] = {}
        for r in group:
            if r["construction"] == "pyvrp_8":
                pv8_by_inst[r["instance_id"]] = r["cost"]
        lines.append(f"\n[N={N}] basin quality vs pyvrp_8 (lower is better; "
                     f"n_instances={len(pv8_by_inst)})")
        lines.append(f"  {'construction':<22} {'mean_bq':>9} {'mean_cost':>11} "
                     f"{'mean_routes':>11} {'mean_wall_s':>11} {'feas%':>6}")
        # Aggregate per construction.
        per_constr: dict[str, list[dict]] = {}
        for r in group:
            per_constr.setdefault(r["construction"], []).append(r)
        # Stable display order matches CONSTRUCTIONS.
        for c in CONSTRUCTIONS:
            rs = per_constr.get(c, [])
            if not rs:
                continue
            bqs: list[float] = []
            for r in rs:
                ref = pv8_by_inst.get(r["instance_id"])
                if ref and ref > 0 and r["cost"] != float("inf"):
                    bqs.append(r["cost"] / ref)
            mean_bq = sum(bqs) / len(bqs) if bqs else float("nan")
            mean_cost = sum(r["cost"] for r in rs if r["cost"] != float("inf")) / max(1, len(rs))
            mean_routes = sum(r["n_routes"] for r in rs) / len(rs)
            mean_wall = sum(r["wall_s"] for r in rs) / len(rs)
            feas_pct = 100.0 * sum(1 for r in rs if r["feasible"]) / len(rs)
            lines.append(f"  {c:<22} {mean_bq:>9.3f} {mean_cost:>11.2f} "
                         f"{mean_routes:>11.2f} {mean_wall:>11.2f} {feas_pct:>5.0f}%")
    return "\n".join(lines)


# ---- Main -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="bench/runs/construction_baseline.json")
    ap.add_argument("--webui", action="store_true",
                    help="Push live progress / agent status to webui.")
    ap.add_argument("--instances-only",
                    choices=["v1", "solomon", "smoke", "all"],
                    default="all",
                    help="Restrict instance set (smoke = single C101 row).")
    ap.add_argument("--constructions-only", nargs="+", default=None,
                    help=f"Subset of constructions: {CONSTRUCTIONS}")
    args = ap.parse_args()

    if args.webui and not os.environ.get("SVRPTW_WEBUI_URL"):
        # Default to running webui at localhost.
        os.environ["SVRPTW_WEBUI_URL"] = "http://127.0.0.1:8765"

    tasks = _build_task_list(args)
    if not tasks:
        print("No tasks built. Check that instance files exist on disk.")
        return 1

    print(f"[construction_baseline] {len(tasks)} tasks "
          f"({len({t[1] for t in tasks})} instances x "
          f"{len({t[0] for t in tasks})} constructions), workers={args.workers}",
          flush=True)
    _push_agent("phase-E1", "running",
                summary=f"{len(tasks)} tasks; workers={args.workers}")
    _push_progress("construction_baseline", 0, len(tasks))
    # Live progress wrapper around map_instances. We re-implement the
    # ProcessPoolExecutor loop here so we can push per-completion webui
    # events. Falls back to bench.parallel.map_instances semantics
    # (results in submission order, _elapsed_s + _task_index).
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_row, t): (i, t) for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            i, t = futures[fut]
            try:
                r = fut.result()
                r["_elapsed_s"] = time.perf_counter() - t0
                r["_task_index"] = i
                results.append(r)
                done = len(results)
                eta = (r["_elapsed_s"] / done) * (len(tasks) - done) if done else None
                line = (f"[{done:>3}/{len(tasks)}] {r['construction']:<22} "
                        f"N={r['N']:<3} {Path(r['instance_path']).stem:<28} "
                        f"cost={r['cost']:>10.2f} routes={r['n_routes']:>3} "
                        f"wall={r['wall_s']:>6.2f}s")
                print(line, flush=True)
                _push_progress("construction_baseline", done, len(tasks),
                               eta_s=eta)
                _push_log(line)
                if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                    _push_agent("phase-E1", "running",
                                summary=f"{done}/{len(tasks)} done; "
                                        f"elapsed={r['_elapsed_s']:.0f}s")
            except Exception as e:
                results.append({"_task_index": i, "_failed": True,
                                "_error": str(e), "task": str(t)})
                print(f"[task {i}] FAILED: {e}", flush=True)
                _push_log(f"FAILED task {i}: {e}")
    results.sort(key=lambda r: r.get("_task_index", 0))

    # Persist
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"tasks": len(tasks),
                    "elapsed_s": time.perf_counter() - t0,
                    "rows": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[wrote] {out_path}", flush=True)

    summary = _summarize(results)
    print(summary)
    _push_agent("phase-E1", "completed",
                summary=f"{len(tasks)} tasks done; "
                        f"elapsed={time.perf_counter() - t0:.0f}s; "
                        f"out={args.out}")
    _push_progress("construction_baseline", len(tasks), len(tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
