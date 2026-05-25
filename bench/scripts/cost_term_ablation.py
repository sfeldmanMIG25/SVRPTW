"""SPEC-F-COST-01 -- ablation bench for opt-in structural cost terms.

Phase F3 from ``WorldsFinestVRP/15 - Cost-Model Exploration.md``. Runs
five solver arms across six OSM instances (Manhattan/Paris/SanFrancisco
x N=100/200) and compares them to a PyVRP baseline. Each arm is a
Settings preset that toggles one or more of the new cost terms:

  A: baseline                                 (all coefs = 0)
  B: crossings_penalty_per_pair=2.0
  C: util_imbalance_penalty_coef=20.0
  D: tw_buffer_bonus_coef=15.0
  E: all three combined

Per (arm, instance) row records:
  - operational_cost (BASE -- no new terms; for fair cross-arm compare)
  - crossings_count, load_util_cv, mean_tw_buffer_score, quality_index
  - wall_clock_s, num_vehicles_used

PyVRP baseline runs with a vanilla Settings (no new terms anyway, since
PyVRP can't see them). Each PyVRP row is keyed by instance and joined
to all five arms during the leaderboard summary.

Win condition: arm E >= arm A on BOTH operational_cost and
quality_index, AND arm E > PyVRP on quality_index.

Output: bench/runs/cost_term_ablation.json (rows + summary).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Allow direct ``python bench/scripts/cost_term_ablation.py ...`` invocation
# in addition to ``python -m bench.scripts.cost_term_ablation`` (which sets
# the parent dir of the package on sys.path automatically).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from svrptw.config import Economics, Settings
from svrptw.io import load_instance
from svrptw.metrics import score_solution
from svrptw.solvers.classical import pyvrp_solver as pv
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto


# Arm presets. Each maps the arm key to an Economics override dict that
# layers on top of the default Settings. Keep the names short ("A".."E")
# so the leaderboard fits on one terminal line.
ARMS: dict[str, dict[str, float]] = {
    "A": {},  # baseline -- all coefs default 0.0
    "B": {"crossings_penalty_per_pair": 2.0},
    "C": {"util_imbalance_penalty_coef": 20.0},
    "D": {"tw_buffer_bonus_coef": 15.0},
    "E": {
        "crossings_penalty_per_pair": 2.0,
        "util_imbalance_penalty_coef": 20.0,
        "tw_buffer_bonus_coef": 15.0,
    },
}


def _settings_for_arm(arm: str) -> Settings:
    """Construct a Settings instance for a named arm. Pure function so
    the worker can re-derive it without sharing pickled state."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm: {arm!r} (known: {sorted(ARMS)})")
    s = Settings()
    if ARMS[arm]:
        s.economics = Economics(**ARMS[arm])
    return s


def _base_cost(inst, sol) -> float:
    """Re-evaluate a solution under the BASELINE Settings (no new terms),
    so that arms can be compared fairly on operational_cost. Without
    this, arms B-E would report inflated costs that include their own
    structural penalties.

    Imported lazily to keep top-level imports fast for --help calls.
    """
    from svrptw.solvers.common.solution import evaluate
    base_settings = Settings()
    return float(evaluate(inst, sol, base_settings)["operational_cost"])


def _row_for_solution(arm: str, inst, sol, wall_s: float,
                       solver_label: str) -> dict[str, Any]:
    """Build one results row from an instance + solution + arm label."""
    qs = score_solution(inst, sol)
    return {
        "arm": arm,
        "solver": solver_label,
        "instance_id": inst.instance_id,
        "N": int(inst.num_customers),
        "operational_cost": _base_cost(inst, sol),
        "crossings_count": int(qs.inter_route_crossings),
        "load_util_cv": float(qs.load_util_cv),
        "mean_tw_buffer_score": float(qs.mean_tw_buffer_score),
        "quality_index": float(qs.quality_index),
        "num_vehicles_used": int(sol.num_vehicles_used),
        "wall_clock_s": float(wall_s),
    }


# Module-level worker -- ProcessPoolExecutor needs picklable functions.
def _run_one(task: tuple[str, str, float]) -> dict[str, Any]:
    """Run one (arm, instance, budget) cell and return the row dict."""
    arm, inst_path, budget = task
    inst = load_instance(inst_path)
    s = _settings_for_arm(arm)
    # Live web-UI hook: pushes a rate-limited snapshot stream during
    # the bandit phase if SVRPTW_WEBUI_URL is set in the worker env.
    on_accept = None
    if os.environ.get("SVRPTW_WEBUI_URL"):
        try:
            from webui.snapshot import make_on_accept
            stream = f"costterm-{arm}-{Path(inst_path).stem}"
            on_accept = make_on_accept(inst, stream=stream,
                                        every_n_accepts=5,
                                        min_improvement=5.0)
        except Exception:
            on_accept = None
    t0 = time.perf_counter()
    sol = solve_auto(inst, s, budget_seconds=budget, on_accept=on_accept)
    wall = time.perf_counter() - t0
    return _row_for_solution(arm, inst, sol, wall, solver_label="solve_auto")


def _run_pyvrp(task: tuple[str, float]) -> dict[str, Any]:
    """Baseline: PyVRP with vanilla Settings (no new terms anyway)."""
    inst_path, budget = task
    inst = load_instance(inst_path)
    s = Settings()
    t0 = time.perf_counter()
    sol = pv.solve(inst, s, budget_seconds=budget)
    wall = time.perf_counter() - t0
    return _row_for_solution("PYVRP", inst, sol, wall, solver_label="pyvrp")


def _default_instances() -> list[str]:
    """Manhattan/Paris/SanFrancisco x N=100/200, rep I000 for stability."""
    cities = ["Manhattan", "Paris", "SanFrancisco"]
    sizes = (100, 200)
    out: list[str] = []
    for c in cities:
        for N in sizes:
            p = Path(f"instances/v1/OSM-{c}-N{N:03d}-I000.json")
            if p.exists():
                out.append(str(p))
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


def _summarize(arm_rows: list[dict[str, Any]],
                pyvrp_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the leaderboard. For each arm, compute mean delta-cost
    (arm - PyVRP) and mean delta-quality (arm - PyVRP) across the
    instances the arm and PyVRP both ran.

    Win condition: arm E >= arm A on cost AND quality, AND arm E >
    PyVRP on quality.
    """
    pv_by_inst = {r["instance_id"]: r for r in pyvrp_rows}
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in arm_rows:
        by_arm.setdefault(r["arm"], []).append(r)
    summary: dict[str, dict[str, float]] = {}
    for arm, rows in sorted(by_arm.items()):
        deltas_cost = []
        deltas_qi = []
        deltas_cross = []
        for r in rows:
            pvr = pv_by_inst.get(r["instance_id"])
            if pvr is None:
                continue
            deltas_cost.append(r["operational_cost"] - pvr["operational_cost"])
            deltas_qi.append(r["quality_index"] - pvr["quality_index"])
            deltas_cross.append(r["crossings_count"] - pvr["crossings_count"])
        n = max(1, len(rows))
        summary[arm] = {
            "n": len(rows),
            "n_paired": len(deltas_cost),
            "mean_op_cost": sum(r["operational_cost"] for r in rows) / n,
            "mean_quality_index": sum(r["quality_index"] for r in rows) / n,
            "mean_crossings": sum(r["crossings_count"] for r in rows) / n,
            "mean_load_util_cv": sum(r["load_util_cv"] for r in rows) / n,
            "mean_tw_buffer": sum(r["mean_tw_buffer_score"] for r in rows) / n,
            "mean_routes": sum(r["num_vehicles_used"] for r in rows) / n,
            "mean_wall_s": sum(r["wall_clock_s"] for r in rows) / n,
            "mean_delta_cost_vs_pyvrp": (
                sum(deltas_cost) / len(deltas_cost) if deltas_cost else float("nan")),
            "mean_delta_qi_vs_pyvrp": (
                sum(deltas_qi) / len(deltas_qi) if deltas_qi else float("nan")),
            "mean_delta_crossings_vs_pyvrp": (
                sum(deltas_cross) / len(deltas_cross) if deltas_cross else float("nan")),
        }
    return summary


def _print_leaderboard(summary: dict[str, dict[str, float]],
                        pyvrp_rows: list[dict[str, Any]]) -> None:
    """Print the leaderboard, sorted twice: by mean cost-delta and by
    mean quality-delta. Surfaces whether enabling structural terms HURT
    operational cost AND whether they HELP structural quality."""
    if pyvrp_rows:
        n_pv = len(pyvrp_rows)
        mean_pv_cost = sum(r["operational_cost"] for r in pyvrp_rows) / n_pv
        mean_pv_qi = sum(r["quality_index"] for r in pyvrp_rows) / n_pv
        mean_pv_cross = sum(r["crossings_count"] for r in pyvrp_rows) / n_pv
        print("\n[cost_term_ablation] PYVRP baseline")
        print(f"  n={n_pv}  mean_op_cost={mean_pv_cost:.2f}  "
              f"mean_qi={mean_pv_qi:.4f}  mean_crossings={mean_pv_cross:.2f}")
    print("\n[cost_term_ablation arms]")
    print(f"  {'arm':<3} {'n':>2} {'op_cost':>10} {'d_cost':>9} "
          f"{'qi':>7} {'d_qi':>8} {'cross':>6} {'cv':>6} "
          f"{'tw_buf':>7} {'routes':>7} {'wall_s':>7}")
    for arm in sorted(summary):
        s = summary[arm]
        print(f"  {arm:<3} {s['n']:>2} "
              f"{s['mean_op_cost']:>10.2f} "
              f"{s['mean_delta_cost_vs_pyvrp']:>+9.2f} "
              f"{s['mean_quality_index']:>7.4f} "
              f"{s['mean_delta_qi_vs_pyvrp']:>+8.4f} "
              f"{s['mean_crossings']:>6.1f} "
              f"{s['mean_load_util_cv']:>6.3f} "
              f"{s['mean_tw_buffer']:>7.4f} "
              f"{s['mean_routes']:>7.2f} "
              f"{s['mean_wall_s']:>7.2f}")
    # Win-condition check (E >= A on both axes, E > PyVRP on quality).
    if "A" in summary and "E" in summary:
        a, e = summary["A"], summary["E"]
        cost_ok = e["mean_op_cost"] <= a["mean_op_cost"] + 1e-6
        qi_ok = e["mean_quality_index"] >= a["mean_quality_index"] - 1e-6
        pv_qi_ok = e["mean_delta_qi_vs_pyvrp"] > 0.0
        verdict = ("WIN" if (cost_ok and qi_ok and pv_qi_ok)
                   else "MIXED" if (cost_ok or qi_ok)
                   else "LOSS")
        print(f"\n[verdict] arm-E vs arm-A: cost_ok={cost_ok} "
              f"qi_ok={qi_ok} pv_qi_ok={pv_qi_ok} -> {verdict}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4,
                    help="ProcessPool workers for parallel solves.")
    ap.add_argument("--budget", type=float, default=30.0,
                    help="solve_auto budget seconds per (arm, instance).")
    ap.add_argument("--pyvrp-budget", type=float, default=60.0,
                    help="PyVRP baseline budget seconds.")
    ap.add_argument("--arms", nargs="+", default=list(ARMS.keys()),
                    choices=list(ARMS.keys()),
                    help="Which arms to run (subset of A B C D E).")
    ap.add_argument("--instances", nargs="*", default=None,
                    help="Override instance paths; default = "
                         "Manhattan/Paris/SF x N=100/200 I000.")
    ap.add_argument("--skip-pyvrp", action="store_true",
                    help="Skip the PyVRP baseline (e.g. for fast smoke).")
    ap.add_argument("--out", default="bench/runs/cost_term_ablation.json")
    ap.add_argument("--webui", action="store_true",
                    help="Push live progress to webui via webui.client.")
    args = ap.parse_args(argv)

    instances = args.instances if args.instances else _default_instances()
    instances = [str(p) for p in instances]
    missing = [p for p in instances if not Path(p).exists()]
    if missing:
        print(f"[cost_term_ablation] missing instance files: {missing}",
              file=sys.stderr)
        return 1
    if not instances:
        print("[cost_term_ablation] no instances selected", file=sys.stderr)
        return 1

    arm_tasks: list[tuple[str, str, float]] = [
        (arm, ip, args.budget) for arm in args.arms for ip in instances
    ]
    pyvrp_tasks: list[tuple[str, float]] = (
        [] if args.skip_pyvrp else
        [(ip, args.pyvrp_budget) for ip in instances]
    )
    total = len(arm_tasks) + len(pyvrp_tasks)
    print(f"[cost_term_ablation] arms={args.arms} "
          f"instances={len(instances)} budget={args.budget}s "
          f"pyvrp_budget={args.pyvrp_budget}s -> {total} solves "
          f"(workers={args.workers})")
    _push("cost_term_ablation",
          f"{total} solves queued ({len(args.arms)} arms x "
          f"{len(instances)} insts + {len(pyvrp_tasks)} pyvrp)",
          n_tasks=total)
    _push_progress("cost_term_ablation", 0, total)

    t0 = time.perf_counter()
    arm_rows: list[dict[str, Any]] = []
    pyvrp_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs: dict = {}
        for i, t in enumerate(arm_tasks):
            futs[pool.submit(_run_one, t)] = ("arm", i, t)
        for i, t in enumerate(pyvrp_tasks):
            futs[pool.submit(_run_pyvrp, t)] = ("pv", i, t)
        for fut in as_completed(futs):
            kind, idx, t = futs[fut]
            try:
                row = fut.result()
                row["_task_index"] = idx
                row["_elapsed_s"] = time.perf_counter() - t0
                if kind == "arm":
                    arm_rows.append(row)
                else:
                    pyvrp_rows.append(row)
                done += 1
                eta = ((time.perf_counter() - t0) / done) * (total - done)
                line = (f"  [{done:>3}/{total}] "
                        f"{row['arm']:<5} {row['solver']:<10} "
                        f"{row['instance_id']:<32} "
                        f"cost={row['operational_cost']:.1f} "
                        f"qi={row['quality_index']:.4f} "
                        f"cross={row['crossings_count']:>3} "
                        f"({row['_elapsed_s']:.0f}s, eta {eta:.0f}s)")
                print(line); sys.stdout.flush()
                if args.webui:
                    _push_progress("cost_term_ablation", done, total, eta_s=eta)
                    _push("cost_term_ablation",
                          f"{done}/{total} done, last={row['arm']} "
                          f"qi={row['quality_index']:.4f}",
                          last_instance=row["instance_id"])
            except Exception as exc:
                failed.append({"_task_index": idx, "_kind": kind,
                               "_error": str(exc), "task": str(t)})
                print(f"  [task {kind}#{idx}] FAILED: {exc}")

    arm_rows.sort(key=lambda r: (r["arm"], r["instance_id"]))
    pyvrp_rows.sort(key=lambda r: r["instance_id"])
    summary = _summarize(arm_rows, pyvrp_rows)
    out_payload = {
        "config": {
            "arms": {a: ARMS[a] for a in args.arms},
            "instances": instances,
            "budget_seconds": args.budget,
            "pyvrp_budget_seconds": args.pyvrp_budget,
        },
        "arm_rows": arm_rows,
        "pyvrp_rows": pyvrp_rows,
        "summary": summary,
        "failed": failed,
        "wall_total_s": time.perf_counter() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_payload, indent=2))
    print(f"\n[cost_term_ablation] wrote {len(arm_rows)} arm rows + "
          f"{len(pyvrp_rows)} pyvrp rows to {args.out} "
          f"in {time.perf_counter()-t0:.1f}s")
    _print_leaderboard(summary, pyvrp_rows)
    _push("idle", f"cost_term_ablation finished: "
                  f"{len(arm_rows)} arms + {len(pyvrp_rows)} pyvrp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
