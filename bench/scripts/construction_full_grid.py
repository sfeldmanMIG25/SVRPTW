"""Construction full-grid bench (iter-5g).

Cross product:
  5 constructions x 4 instances x 2 post-process modes = 40 solves

Constructions:
  - pyvrp_4s
  - pyvrp_8s
  - fast_construct_2s
  - fast_construct_4s_polished  (uses iter-5g polish + quality-pick)
  - regret_3                    (regret-k k=3, no LS)

Post-process modes:
  - none
  - bandit_30s_quality_weighted  (pm.solve with quality_weight=0.5)

Output: bench/runs/construction_full_grid.json + leaderboard ranked by
mean unified score (0.5 * cost_norm + 0.5 * quality_index).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python bench/scripts/construction_full_grid.py` invocation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.metrics import score_solution


CONSTRUCTIONS = [
    "pyvrp_4s",
    "pyvrp_8s",
    "fast_construct_2s",
    "fast_construct_4s_polished",
    "regret_3",
]

POST_MODES = ["none", "bandit_30s_quality_weighted"]


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


# ---- Construction dispatch ------------------------------------------------

def _run_construction(name: str, inst, settings, *, seed: int = 0):
    """Build a construction by name. Returns (Solution, wall_seconds)."""
    t0 = time.perf_counter()
    if name == "pyvrp_4s":
        from svrptw.solvers.classical import pyvrp_solver as pv
        sol = pv.solve(inst, settings, budget_seconds=4.0)
    elif name == "pyvrp_8s":
        from svrptw.solvers.classical import pyvrp_solver as pv
        sol = pv.solve(inst, settings, budget_seconds=8.0)
    elif name == "fast_construct_2s":
        from svrptw.solvers.classical import fast_construct as fc
        sol = fc.solve(inst, settings, budget_seconds=2.0,
                       seed=seed, polish=False)
    elif name == "fast_construct_4s_polished":
        from svrptw.solvers.classical import fast_construct as fc
        sol = fc.solve(inst, settings, budget_seconds=4.0,
                       seed=seed, polish=True, quality_pick=True)
    elif name == "regret_3":
        from svrptw.solvers.classical import regret_k as rk
        sol = rk.solve(inst, settings, budget_seconds=0.0, k=3)
    else:
        raise ValueError(f"unknown construction {name!r}")
    return sol, time.perf_counter() - t0


def _run_post(mode: str, inst, settings, sol, *, seed: int = 0):
    """Apply a post-processing mode. Returns (Solution, wall_seconds)."""
    t0 = time.perf_counter()
    if mode == "none":
        return sol, 0.0
    if mode == "bandit_30s_quality_weighted":
        from svrptw.solvers.classical import portfolio as pm
        out = pm.solve(inst, settings, budget_seconds=30.0,
                       initial_solution=sol,
                       seed=seed, quality_weight=0.5,
                       plateaus_to_stop=20)
        return out, time.perf_counter() - t0
    raise ValueError(f"unknown post-mode {mode!r}")


# ---- Worker (top-level so ProcessPoolExecutor can pickle it) --------------

def _row(task: tuple) -> dict[str, Any]:
    construction, post_mode, inst_path, seed = task
    s = Settings()
    inst = load_instance(inst_path)
    name = f"trial-{construction}-{Path(inst_path).stem}-{post_mode}"
    _push_agent(name, "running",
                summary=f"start {construction} + {post_mode}")
    try:
        sol, wall_constr = _run_construction(construction, inst, s, seed=seed)
    except Exception as e:
        _push_agent(name, "failed", summary=f"construction error: {e}")
        return {
            "construction": construction, "post_mode": post_mode,
            "instance_id": inst.instance_id,
            "instance_path": str(inst_path),
            "N": int(inst.num_customers),
            "_failed": True, "_error": f"construction: {e}",
        }
    try:
        sol, wall_post = _run_post(post_mode, inst, s, sol, seed=seed)
    except Exception as e:
        _push_agent(name, "failed", summary=f"post error: {e}")
        return {
            "construction": construction, "post_mode": post_mode,
            "instance_id": inst.instance_id,
            "instance_path": str(inst_path),
            "N": int(inst.num_customers),
            "_failed": True, "_error": f"post: {e}",
        }
    cost = float(sol.metrics.get("operational_cost", float("inf")))
    feas = bool(sol.metrics.get("feasible", 0.0))
    n_routes = int(sol.metrics.get("num_vehicles_used", -1))
    try:
        q = score_solution(inst, sol)
        q_index = float(q.quality_index)
        crossings = int(q.inter_route_crossings)
        tw_buf = float(q.mean_tw_buffer_score)
        on_time = float(q.on_time_frac)
        wait_min = float(q.total_wait_min)
    except Exception as e:
        q_index = 0.0; crossings = -1; tw_buf = 0.0
        on_time = 0.0; wait_min = 0.0
    out = {
        "construction": construction, "post_mode": post_mode,
        "instance_id": inst.instance_id,
        "instance_path": str(inst_path),
        "N": int(inst.num_customers),
        "cost": cost, "feasible": feas, "n_routes": n_routes,
        "quality_index": q_index,
        "inter_route_crossings": crossings,
        "mean_tw_buffer_score": tw_buf,
        "on_time_frac": on_time,
        "total_wait_min": wait_min,
        "wall_construction_s": wall_constr,
        "wall_post_s": wall_post,
        "wall_total_s": wall_constr + wall_post,
    }
    _push_agent(name, "completed",
                summary=f"cost={cost:.2f} q={q_index:.4f} cross={crossings}",
                **{k: v for k, v in out.items()
                   if k in ("cost", "quality_index", "inter_route_crossings",
                            "wall_total_s")})
    return out


# ---- Task building --------------------------------------------------------

DEFAULT_INSTANCES = [
    "instances/v1/OSM-Manhattan-N050-I003.json",
    "instances/v1/OSM-Paris-N050-I003.json",
    "instances/v1/OSM-Manhattan-N100-I003.json",
    "instances/v1/OSM-Paris-N100-I003.json",
]


def _build_tasks(instances: list[str], seed: int) -> list[tuple]:
    out: list[tuple] = []
    for inst_path in instances:
        for c in CONSTRUCTIONS:
            for pm_mode in POST_MODES:
                out.append((c, pm_mode, inst_path, seed))
    return out


# ---- Reporting ------------------------------------------------------------

def _unified(cost: float, q: float, cost_ref: float | None = None) -> float:
    """0.5 * cost_norm + 0.5 * quality_index. cost_ref is the cheapest
    feasible cost in the row's cohort (per-instance). Cost-norm clipped
    to [0, 1.0]."""
    if cost_ref is None or cost_ref <= 0 or cost <= 0:
        cost_norm = 0.0
    else:
        cost_norm = max(0.0, min(1.0, cost_ref / cost))
    return 0.5 * cost_norm + 0.5 * float(q)


def _summarize(rows: list[dict[str, Any]]) -> str:
    """Per-instance + global leaderboard ranked by mean unified score."""
    lines: list[str] = []
    by_inst: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("_failed"):
            continue
        by_inst.setdefault(r["instance_id"], []).append(r)
    # Per-instance unified score (cost_ref = cheapest in instance).
    per_arm: dict[tuple[str, str], list[float]] = {}
    for inst_id, group in by_inst.items():
        feas = [r for r in group if r.get("feasible") and r["cost"] > 0]
        cost_ref = min((r["cost"] for r in feas), default=None)
        for r in group:
            unified = _unified(r["cost"], r["quality_index"], cost_ref)
            r["unified_score"] = unified
            arm_key = (r["construction"], r["post_mode"])
            per_arm.setdefault(arm_key, []).append(unified)
    # Per-instance breakdown
    lines.append("\n=== per-instance leaderboard (top by unified) ===")
    for inst_id in sorted(by_inst):
        group = sorted(by_inst[inst_id],
                        key=lambda r: r.get("unified_score", -1),
                        reverse=True)
        lines.append(f"\n[{inst_id}]")
        lines.append(f"  {'arm':<48} {'unified':>8} {'cost':>10} "
                     f"{'q':>6} {'cross':>6} {'wall':>7}")
        for r in group:
            arm = f"{r['construction']}/{r['post_mode']}"
            lines.append(f"  {arm:<48} {r.get('unified_score',0):>8.4f} "
                         f"{r['cost']:>10.2f} {r['quality_index']:>6.3f} "
                         f"{r['inter_route_crossings']:>6} "
                         f"{r['wall_total_s']:>7.2f}")
    # Global: mean unified score per arm
    lines.append("\n=== global leaderboard (mean unified across instances) ===")
    lines.append(f"  {'arm':<48} {'mean_unified':>14} {'n':>4}")
    ranked = sorted(per_arm.items(),
                     key=lambda kv: -sum(kv[1])/max(len(kv[1]), 1))
    for (constr, pm_mode), scores in ranked:
        m = sum(scores) / max(len(scores), 1)
        arm = f"{constr}/{pm_mode}"
        lines.append(f"  {arm:<48} {m:>14.4f} {len(scores):>4}")
    return "\n".join(lines)


# ---- Main -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="bench/runs/construction_full_grid.json")
    ap.add_argument("--webui", action="store_true",
                    help="Push live progress / agent status to webui.")
    ap.add_argument("--instances", nargs="+", default=None,
                    help=f"Override default instance set ({DEFAULT_INSTANCES}).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.webui and not os.environ.get("SVRPTW_WEBUI_URL"):
        os.environ["SVRPTW_WEBUI_URL"] = "http://127.0.0.1:8765"

    instances = args.instances or DEFAULT_INSTANCES
    # Skip instances that don't exist (best-effort smoke).
    instances = [p for p in instances if Path(p).exists()]
    if not instances:
        print("No instance files found on disk.", file=sys.stderr)
        return 1
    tasks = _build_tasks(instances, seed=args.seed)
    print(f"[construction_full_grid] {len(tasks)} tasks "
          f"({len(instances)} instances x {len(CONSTRUCTIONS)} constr "
          f"x {len(POST_MODES)} post), workers={args.workers}",
          flush=True)
    _push_agent("iter5g-grid", "running",
                summary=f"{len(tasks)} tasks; workers={args.workers}")
    _push_progress("construction_full_grid", 0, len(tasks))

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_row, t): (i, t) for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            i, t = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"_task_index": i, "_failed": True, "_error": str(e),
                     "task": str(t)}
            r["_elapsed_s"] = time.perf_counter() - t0
            r["_task_index"] = i
            results.append(r)
            done = len(results)
            eta = (r["_elapsed_s"] / done) * (len(tasks) - done) if done else None
            if r.get("_failed"):
                print(f"[{done:>3}/{len(tasks)}] FAILED: {r.get('_error')}",
                      flush=True)
            else:
                print(f"[{done:>3}/{len(tasks)}] "
                      f"{r['construction']:<32} {r['post_mode']:<32} "
                      f"{Path(r['instance_path']).stem:<28} "
                      f"cost={r['cost']:>10.2f} q={r['quality_index']:.3f} "
                      f"cross={r['inter_route_crossings']:>3} "
                      f"wall={r['wall_total_s']:>6.2f}s",
                      flush=True)
            _push_progress("construction_full_grid", done, len(tasks),
                           eta_s=eta)
    results.sort(key=lambda r: r.get("_task_index", 0))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"tasks": len(tasks),
                    "elapsed_s": time.perf_counter() - t0,
                    "instances": instances,
                    "constructions": CONSTRUCTIONS,
                    "post_modes": POST_MODES,
                    "rows": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[wrote] {out_path}", flush=True)
    summary = _summarize(results)
    print(summary)
    _push_agent("iter5g-grid", "completed",
                summary=f"{len(tasks)} tasks; "
                        f"elapsed={time.perf_counter() - t0:.0f}s; "
                        f"out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
