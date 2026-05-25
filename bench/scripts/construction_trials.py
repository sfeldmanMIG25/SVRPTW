"""Head-to-head trial bench across construction techniques.

Runs every construction in a fixed catalog against a stratified
instance set, scores each with the unified objective, and prints a
leaderboard ranked by mean ``unified`` score per construction.

Usage::

    # full default sweep (8 instances x ~14 constructions)
    set PYTHONPATH=D:/SVRPTW
    python bench/scripts/construction_trials.py

    # smoke test on one instance + 3 constructions, no bandit
    python bench/scripts/construction_trials.py \\
        --instances instances/v1/OSM-Manhattan-N050-I003.json \\
        --no-bandit \\
        --constructions pyvrp_4s fast_construct_2s greedy

The bench writes ``bench/runs/construction_trials.json`` and pushes
per-(construction, instance) progress to the webui via
``webui.client``.

PYTHONPATH=D:/SVRPTW must be set so the bench can import the svrptw
package (we do not install editable for the trial run).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

_LOG = logging.getLogger("bench.construction_trials")


def _import_solvers():
    """Lazy-import the classical solver modules so --help is fast even
    on a machine where torch/ortools wheels aren't fully ready."""
    from svrptw.solvers.classical import (
        auction_gart, fast_construct, greedy, portfolio,
        portfolio_pyvrp_warm, pyvrp_solver, regret_k,
    )
    return {
        "pyvrp": pyvrp_solver,
        "auction_gart": auction_gart,
        "regret_k": regret_k,
        "fast_construct": fast_construct,
        "greedy": greedy,
        "portfolio": portfolio,
        "portfolio_pyvrp_warm": portfolio_pyvrp_warm,
    }


# ---------------------------------------------------------------------------
# Construction catalog. Each entry is a callable taking (inst, settings)
# and returning a Solution. Wall budgets are baked in.
# ---------------------------------------------------------------------------


def _construction_catalog(mods) -> dict[str, Callable]:
    pv = mods["pyvrp"]
    ag = mods["auction_gart"]
    rk = mods["regret_k"]
    fc = mods["fast_construct"]
    gr = mods["greedy"]

    def _pyvrp(seconds: float):
        return lambda inst, s: pv.solve(inst, s, budget_seconds=seconds)

    def _auction(seconds: float):
        return lambda inst, s: ag.solve(inst, s, budget_seconds=seconds)

    def _regret(k: int):
        return lambda inst, s: rk.solve(inst, s, k=k)

    def _fc(seconds: float, k_starts: int):
        return lambda inst, s: fc.solve(inst, s,
                                        budget_seconds=seconds,
                                        k_starts=k_starts)

    def _greedy():
        return lambda inst, s: gr.solve(inst, s)

    return {
        "pyvrp_8s":          _pyvrp(8.0),
        "pyvrp_4s":          _pyvrp(4.0),
        "auction_gart_2s":   _auction(2.0),
        "regret_3":          _regret(3),
        "fast_construct_2s": _fc(2.0, 4),
        "fast_construct_4s": _fc(4.0, 8),
        "greedy":            _greedy(),
    }


def _bandit_postprocess(mods, warm, inst, settings,
                        budget_seconds: float = 30.0):
    """Feed ``warm`` into portfolio.solve as the initial_solution.

    Returns the refined Solution. Used by every "<construction>_then_bandit_30s"
    arm of the catalog.
    """
    pm = mods["portfolio"]
    return pm.solve(inst, settings,
                    budget_seconds=budget_seconds,
                    initial_solution=warm,
                    plateaus_to_stop=20)


# ---------------------------------------------------------------------------
# Per-instance / per-construction trial driver
# ---------------------------------------------------------------------------


def _stratified_default_set(root: Path) -> list[Path]:
    """Default 8-instance stratified set: 4 cities x N=50,100, I=003 each."""
    cities = ["Manhattan", "Paris", "SanFrancisco", "Charleston"]
    sizes = [50, 100]
    out: list[Path] = []
    for c in cities:
        for n in sizes:
            cand = root / f"OSM-{c}-N{n:03d}-I003.json"
            if cand.exists():
                out.append(cand)
            else:
                _LOG.warning("instance %s missing — skipping", cand.name)
    return out


def _run_one(inst, sol_fn, settings, *,
             tag: str, want_bandit: bool, mods,
             bandit_seconds: float) -> dict[str, Any]:
    """Run a single (construction, instance) trial; optionally chain bandit.

    Returns a row with raw construction + (optional) bandit metrics. The
    UnifiedScore is added by the caller after a per-instance cost_max
    is known.
    """
    t0 = time.perf_counter()
    warm = sol_fn(inst, settings)
    warm_wall = time.perf_counter() - t0
    warm_cost = float(warm.metrics.get("operational_cost", float("inf")))
    row: dict[str, Any] = {
        "construction": tag,
        "instance_id": inst.instance_id,
        "city": inst.city,
        "N": inst.num_customers,
        "warm_solver": warm.solver,
        "warm_cost": warm_cost,
        "warm_routes": int(warm.metrics.get("num_vehicles_used",
                                            float(warm.num_vehicles_used))),
        "warm_feasible": bool(warm.metrics.get("feasible", warm.feasible)),
        "warm_wall_s": warm_wall,
    }
    if want_bandit:
        try:
            t1 = time.perf_counter()
            refined = _bandit_postprocess(mods, warm, inst, settings,
                                          budget_seconds=bandit_seconds)
            row["bandit_wall_s"] = time.perf_counter() - t1
            row["refined_cost"] = float(refined.metrics.get(
                "operational_cost", float("inf")))
            row["refined_routes"] = int(refined.metrics.get(
                "num_vehicles_used", float(refined.num_vehicles_used)))
            row["final_solution_obj"] = refined  # transient
        except Exception as e:  # pragma: no cover -- defensive
            _LOG.warning("bandit refine failed for %s/%s: %s",
                         tag, inst.instance_id, e)
            row["bandit_error"] = str(e)
            row["final_solution_obj"] = warm
    else:
        row["final_solution_obj"] = warm
    return row


def _push_progress(name: str, completed: int, total: int) -> None:
    """Best-effort webui push (silent on failure — no server is OK)."""
    try:
        from webui import client as ui
        ui.push_progress(name, completed, total)
    except Exception as e:
        _LOG.debug("webui push_progress failed: %s", e)


def _push_agent(name: str, status: str, progress: float | None = None,
                summary: str = "") -> None:
    try:
        from webui import client as ui
        ui.push_agent(name, status, progress=progress, summary=summary)
    except Exception as e:
        _LOG.debug("webui push_agent failed: %s", e)


# ---------------------------------------------------------------------------
# Leaderboard formatting
# ---------------------------------------------------------------------------


def _print_leaderboard(rows: list[dict[str, Any]]) -> None:
    """Group rows by construction tag and print mean unified score, then
    sort descending. Headers + columns kept to a fixed width so the
    output is grepable from the parent agent."""
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_tag.setdefault(r["construction"], []).append(r)
    summary = []
    for tag, rs in by_tag.items():
        unif = [r["unified"] for r in rs if r.get("unified") is not None]
        cost = [r["operational_cost"] for r in rs
                if r.get("operational_cost") is not None
                and r["operational_cost"] != float("inf")]
        qual = [r["quality_index"] for r in rs if r.get("quality_index") is not None]
        if not unif:
            continue
        summary.append({
            "construction": tag,
            "n": len(rs),
            "mean_unified": sum(unif) / len(unif),
            "mean_cost": (sum(cost) / len(cost)) if cost else float("nan"),
            "mean_quality": (sum(qual) / len(qual)) if qual else float("nan"),
        })

    summary.sort(key=lambda r: r["mean_unified"], reverse=True)
    print()
    print("=" * 78)
    print("CONSTRUCTION-TRIAL LEADERBOARD (mean unified per construction)")
    print("=" * 78)
    print(f"{'rank':<5}{'construction':<32}{'n':<4}"
          f"{'unified':<10}{'cost($)':<12}{'quality':<10}")
    print("-" * 78)
    for i, r in enumerate(summary, start=1):
        print(f"{i:<5}{r['construction']:<32}{r['n']:<4}"
              f"{r['mean_unified']:<10.4f}"
              f"{r['mean_cost']:<12.2f}{r['mean_quality']:<10.4f}")
    print("=" * 78)


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="construction_trials.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--instances", nargs="*", type=Path, default=None,
                   help="Instance JSON paths. Default: stratified set "
                        "(Manhattan/Paris/SF/Charleston x N=50,100, I003).")
    p.add_argument("--instances-root", type=Path,
                   default=Path("instances/v1"),
                   help="Root used to resolve the default stratified set.")
    p.add_argument("--constructions", nargs="*", default=None,
                   help="Subset of constructions to run. Default: all.")
    p.add_argument("--no-bandit", action="store_true",
                   help="Skip the +bandit_30s post-process arm.")
    p.add_argument("--bandit-seconds", type=float, default=30.0,
                   help="Bandit refine budget (default 30s).")
    p.add_argument("--out", type=Path,
                   default=Path("bench/runs/construction_trials.json"))
    p.add_argument("--list-constructions", action="store_true",
                   help="Print available construction tags and exit.")
    return p


def _resolve_instances(args, root: Path) -> list[Path]:
    if args.instances:
        return [Path(p) for p in args.instances]
    return _stratified_default_set(root)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(name)s] %(message)s")
    args = _argparser().parse_args(argv)

    mods = _import_solvers()
    catalog = _construction_catalog(mods)

    if args.list_constructions:
        for k in catalog:
            print(k)
        return 0

    inst_paths = _resolve_instances(args, args.instances_root)
    if not inst_paths:
        print("error: no instances resolved (try --instances).",
              file=sys.stderr)
        return 2

    if args.constructions:
        keep = set(args.constructions)
        catalog = {k: v for k, v in catalog.items() if k in keep}
        if not catalog:
            print(f"error: --constructions matched none of {list(catalog)}",
                  file=sys.stderr)
            return 2

    from svrptw.config import Settings
    from svrptw.io.instance import load_instance
    from svrptw.metrics import score_solution
    from svrptw.scoring.sub_api import score as unified_score

    settings = Settings()
    rows: list[dict[str, Any]] = []
    arms_per_inst = len(catalog) * (1 if args.no_bandit else 2)
    total_runs = arms_per_inst * len(inst_paths)
    done = 0
    _push_agent("construction-trials", "running", progress=0.0,
                summary=f"0/{total_runs} arms")


    for ip in inst_paths:
        inst = load_instance(ip)
        # Per-instance trial: collect raw rows for all constructions first
        # so we can derive a shared cost_max for normalising the unified
        # cost component across constructions on the same instance.
        per_inst_rows: list[dict[str, Any]] = []
        for tag, fn in catalog.items():
            warm_row = _run_one(inst, fn, settings,
                                tag=tag,
                                want_bandit=False,
                                mods=mods,
                                bandit_seconds=args.bandit_seconds)
            per_inst_rows.append(warm_row)
            done += 1
            _push_progress("construction_trials", done, total_runs)
            _push_agent("construction-trials", "running",
                        progress=done / max(1, total_runs),
                        summary=f"{done}/{total_runs} {tag} {inst.instance_id}")
            if not args.no_bandit:
                bandit_row = _run_one(inst, fn, settings,
                                      tag=f"{tag}_then_bandit_{int(args.bandit_seconds)}s",
                                      want_bandit=True,
                                      mods=mods,
                                      bandit_seconds=args.bandit_seconds)
                per_inst_rows.append(bandit_row)
                done += 1
                _push_progress("construction_trials", done, total_runs)

        # Per-instance cost_max: 1.5x the worst finite cost in the batch.
        finite = [r["warm_cost"] for r in per_inst_rows
                  if r.get("warm_cost") not in (None, float("inf"))]
        finite += [r.get("refined_cost") for r in per_inst_rows
                   if r.get("refined_cost") not in (None, float("inf"))]
        cost_max = (max(finite) * 1.5) if finite else None


        for r in per_inst_rows:
            sol = r.pop("final_solution_obj", None)
            if sol is None:
                rows.append(r)
                continue
            try:
                qs = score_solution(inst, sol)
                us = unified_score(inst, sol, settings=settings,
                                   cost_max=cost_max, quality=qs)
                r["operational_cost"] = us.operational_cost
                r["quality_index"] = us.quality_index
                r["visual_score"] = us.visual_score
                r["unified"] = us.unified
                r["breakdown"] = us.breakdown
                r["inter_route_crossings"] = qs.inter_route_crossings
                r["silhouette_like"] = qs.silhouette_like
                r["load_util_cv"] = qs.load_util_cv
                r["n_routes"] = qs.n_routes
                r["cost_max_used"] = cost_max
            except Exception as e:
                _LOG.warning("scoring failed for %s/%s: %s",
                             r.get("construction"), r.get("instance_id"), e)
                r["scoring_error"] = str(e)
            rows.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "construction_trials.v1",
        "rows": rows,
        "n_instances": len(inst_paths),
        "n_constructions": len(catalog),
        "no_bandit": bool(args.no_bandit),
        "bandit_seconds": float(args.bandit_seconds),
        "instance_paths": [str(p) for p in inst_paths],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    _push_agent("construction-trials", "completed", progress=1.0,
                summary=f"wrote {args.out}")

    _print_leaderboard(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
