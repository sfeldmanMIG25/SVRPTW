"""Benchmark harness.  SPEC-0-BENCH-01.

Runs each solver on each instance at the requested budget points, writes one
row per (solver, instance, budget) into a JSON file.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import auction_gart as auction_gart_mod
from svrptw.solvers.classical import greedy as greedy_mod
from svrptw.solvers.classical import lkh3 as lkh3_mod
from svrptw.solvers.classical import ortools_gart as ortools_gart_mod
from svrptw.solvers.classical import ortools_solver as ortools_mod
from svrptw.solvers.classical import portfolio as portfolio_mod
from svrptw.solvers.classical import pyvrp_solver as pyvrp_mod
from svrptw.solvers.classical import regret_k as regret_k_mod
from svrptw.solvers.common import Solution
from svrptw.solvers.learning import pomo_stub as pomo_mod
from svrptw.solvers.learning.pomo import infer as pomo_infer

SolverFn = Callable[[object, Settings], Solution]


def _hardware() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "nogit"


def _solver_factories() -> dict[str, Callable[..., Solution]]:
    return {
        "greedy":            lambda inst, s, b: greedy_mod.solve(inst, s),
        "ortools@1":         lambda inst, s, b: ortools_mod.solve(inst, s, 1.0),
        "ortools@10":        lambda inst, s, b: ortools_mod.solve(inst, s, 10.0),
        "ortools@60":        lambda inst, s, b: ortools_mod.solve(inst, s, 60.0),
        "ortools_gart@10":   lambda inst, s, b: ortools_gart_mod.solve(inst, s, 10.0),
        "ortools_gart@60":   lambda inst, s, b: ortools_gart_mod.solve(inst, s, 60.0),
        "lkh3@1":            lambda inst, s, b: lkh3_mod.solve(inst, s, 1.0),
        "lkh3@10":           lambda inst, s, b: lkh3_mod.solve(inst, s, 10.0),
        "lkh3@60":           lambda inst, s, b: lkh3_mod.solve(inst, s, 60.0),
        "auction_gart":      lambda inst, s, b: auction_gart_mod.solve(inst, s),
        "portfolio@10":      lambda inst, s, b: portfolio_mod.solve(inst, s, budget_seconds=10.0),
        "portfolio@30":      lambda inst, s, b: portfolio_mod.solve(inst, s, budget_seconds=30.0),
        "regret_k@10":       lambda inst, s, b: regret_k_mod.solve(inst, s, budget_seconds=10.0, k=3),
        "regret_k@30":       lambda inst, s, b: regret_k_mod.solve(inst, s, budget_seconds=30.0, k=3),
        "pyvrp@1":           lambda inst, s, b: pyvrp_mod.solve(inst, s, 1.0),
        "pyvrp@10":          lambda inst, s, b: pyvrp_mod.solve(inst, s, 10.0),
        "pyvrp@30":          lambda inst, s, b: pyvrp_mod.solve(inst, s, 30.0),
        "pyvrp@60":          lambda inst, s, b: pyvrp_mod.solve(inst, s, 60.0),
        "pomo_stub":         lambda inst, s, b: pomo_mod.solve(inst, s),
        "pomo_v1":           lambda inst, s, b: pomo_infer.solve(inst, s, n_starts=32),
        "pomo_v1_greedy":    lambda inst, s, b: pomo_infer.solve(inst, s, n_starts=32, greedy_decode=True),
    }


def run(instances_dir: Path, solvers: list[str], out: Path,
        settings: Settings, n_filter: list[int] | None = None,
        per_city_limit: int | None = None,
        vivrp_backend: str | None = None) -> dict:
    factories = _solver_factories()
    unknown = [s for s in solvers if s not in factories]
    if unknown:
        raise ValueError(f"unknown solver(s): {unknown}.  Available: {list(factories)}")

    inst_paths = sorted(Path(instances_dir).glob("*.json"))
    if n_filter is not None:
        inst_paths = [p for p in inst_paths if any(f"N{n:03d}" in p.name for n in n_filter)]
    if per_city_limit is not None:
        per_city: dict[tuple[str, int], int] = {}
        keep: list[Path] = []
        for p in inst_paths:
            parts = p.stem.split("-")
            key = (parts[1], int(parts[2][1:]))
            per_city[key] = per_city.get(key, 0) + 1
            if per_city[key] <= per_city_limit:
                keep.append(p)
        inst_paths = keep

    rows: list[dict] = []
    sha = _git_sha()
    hw = _hardware()

    total = len(inst_paths) * len(solvers)
    with Progress(TextColumn("[bold blue]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), TimeRemainingColumn()) as prog:
        task = prog.add_task("benchmark", total=total)
        for inst_path in inst_paths:
            inst = load_instance(inst_path)
            for solver_name in solvers:
                t0 = time.perf_counter()
                try:
                    sol = factories[solver_name](inst, settings, None)
                    error = None
                except Exception as e:
                    sol = None
                    error = repr(e)
                elapsed = time.perf_counter() - t0
                row = {
                    "solver": solver_name,
                    "instance_id": inst.instance_id,
                    "n": inst.num_customers,
                    "city": inst.city,
                    "wall_clock_seconds": elapsed,
                    "git_sha": sha,
                    "hardware": hw,
                    "error": error,
                }
                if sol is not None:
                    row.update({k: v for k, v in sol.metrics.items()})
                    row["budget_seconds"] = sol.budget_seconds
                    if vivrp_backend and sol.feasible:
                        try:
                            from svrptw.vivrp import assess
                            from svrptw.viz import render_solution
                            img_dir = out.parent / "vivrp_imgs"
                            img_dir.mkdir(parents=True, exist_ok=True)
                            img = img_dir / f"{inst.instance_id}__{solver_name}.png"
                            render_solution(inst, sol, img)
                            rep = assess(img, summary={
                                "N": inst.num_customers,
                                "K": int(sol.metrics.get("num_vehicles_used", 0)),
                                "missed": int(sol.metrics.get("missed_deliveries", 0)),
                            }, prefer=vivrp_backend, do_zoom=False)
                            row["vivrp_overall"] = rep.overall_score
                            row["vivrp_clustering"] = rep.clustering_score
                            row["vivrp_geometry"] = rep.geometry_score
                            row["vivrp_interpretability"] = rep.interpretability_score
                            row["vivrp_notes"] = rep.notes
                            row["vivrp_model"] = rep.model
                        except Exception as e:
                            row["vivrp_error"] = repr(e)
                rows.append(row)
                prog.advance(task)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "config": settings.model_dump()}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _report(path: Path) -> None:
    with open(path) as f:
        data = json.load(f)
    by_solver_n: dict[tuple[str, int], list[dict]] = {}
    for row in data["rows"]:
        key = (row["solver"], row["n"])
        by_solver_n.setdefault(key, []).append(row)
    print(f"\n{'solver':<14} {'N':>4} {'inst':>5} {'mean cost':>14} {'mean miss':>10} {'mean s':>8}")
    print("-" * 64)
    for (solver, n), rs in sorted(by_solver_n.items()):
        costs = [r["operational_cost"] for r in rs if r.get("operational_cost") is not None]
        misses = [r["missed_deliveries"] for r in rs if r.get("missed_deliveries") is not None]
        secs  = [r["wall_clock_seconds"] for r in rs]
        if not costs:
            continue
        print(f"{solver:<14} {n:>4} {len(rs):>5} {sum(costs)/len(costs):>14.2f} "
              f"{sum(misses)/len(misses):>10.2f} {sum(secs)/len(secs):>8.2f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("--instances", default="instances/v1")
    pr.add_argument("--out", required=True)
    pr.add_argument("--solvers", default="greedy,ortools@1,ortools@10,ortools@60")
    pr.add_argument("--n-filter", default="", help="Comma-separated N sizes to keep")
    pr.add_argument("--per-city-limit", type=int, default=None)
    pr.add_argument("--config", default=None)
    pr.add_argument("--vivrp", choices=["auto", "lmstudio", "local", "gemini", "stub"], default=None,
                    help="If set, score each feasible solution with ViVRP and include columns.")

    pre = sub.add_parser("report")
    pre.add_argument("path")

    args = p.parse_args(argv)

    if args.cmd == "run":
        settings = Settings.from_yaml(args.config) if args.config else Settings()
        nf = [int(x) for x in args.n_filter.split(",") if x.strip()] if args.n_filter else None
        run(Path(args.instances), args.solvers.split(","), Path(args.out),
            settings, nf, args.per_city_limit, vivrp_backend=args.vivrp)
        _report(Path(args.out))
        return 0
    if args.cmd == "report":
        _report(Path(args.path))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
