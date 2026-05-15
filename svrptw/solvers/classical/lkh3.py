"""LKH-3 wrapper for asymmetric CVRPTW.  Writes a CVRPTW problem file,
shells out to C:/LKH/LKH-3.exe, parses the output.

LKH-3 doesn't have a Python binding; we go through files.  The CVRPTW
representation: EXPLICIT FULL_MATRIX edge weights, TIME_WINDOW_SECTION,
DEPOT_SECTION, CAPACITY, DEMAND_SECTION.  See LKH-3 user guide §4.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.common import Route, Solution, evaluate

LKH_EXE = Path(os.environ.get("SVRPTW_LKH_EXE", r"C:\LKH\LKH-3.exe"))
_SCALE = 100  # 1/100-minute resolution (matches OR-Tools)


def _write_problem(inst: Instance, work: Path, max_seconds: int) -> tuple[Path, Path]:
    """Write LKH-3 problem and parameter files. Returns (par_path, problem_path)."""
    N = inst.num_customers
    T = (inst.travel_time * _SCALE).round().astype(np.int64)
    np.fill_diagonal(T, 0)

    prob_path = work / "problem.cvrptw"
    par_path  = work / "problem.par"

    lines = []
    lines.append(f"NAME : {inst.instance_id}")
    lines.append("TYPE : CVRPTW")
    lines.append(f"DIMENSION : {N + 1}")
    lines.append(f"VEHICLES : {inst.num_vehicles}")
    lines.append(f"CAPACITY : {inst.vehicle_capacity}")
    lines.append("EDGE_WEIGHT_TYPE : EXPLICIT")
    lines.append("EDGE_WEIGHT_FORMAT : FULL_MATRIX")
    lines.append("EDGE_WEIGHT_SECTION")
    for i in range(N + 1):
        lines.append(" ".join(str(int(v)) for v in T[i]))
    lines.append("DEMAND_SECTION")
    lines.append("1 0")
    for i, c in enumerate(inst.customers, start=2):
        lines.append(f"{i} {c.demand}")
    lines.append("TIME_WINDOW_SECTION")
    lines.append(f"1 {int(inst.depot.ready * _SCALE)} {int(inst.depot.due * _SCALE)}")
    for i, c in enumerate(inst.customers, start=2):
        lines.append(f"{i} {int(c.ready * _SCALE)} {int(c.due * _SCALE)}")
    lines.append("SERVICE_TIME_SECTION")
    lines.append("1 0")
    for i, c in enumerate(inst.customers, start=2):
        lines.append(f"{i} {int(c.service * _SCALE)}")
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")
    prob_path.write_text("\n".join(lines), encoding="ascii")

    tour_out = work / "tour.txt"
    # SPEC-0-EVAL-01 follow-up: with RUNS=1 + TIME_LIMIT=1 LKH-3 was
    # producing capacity-overloaded routes (1.5× cap). 5 restarts gives
    # it more chances to converge to a capacity-feasible solution.
    # MTSP_MIN_SIZE=1 ensures every used vehicle carries ≥ 1 customer
    # (prevents degenerate zero-route returns).
    par = [
        f"PROBLEM_FILE = {prob_path}",
        f"MTSP_SOLUTION_FILE = {tour_out}",
        "RUNS = 5",
        f"TIME_LIMIT = {max_seconds}",
        "TRACE_LEVEL = 0",
        "MTSP_OBJECTIVE = MINSUM",
        "MTSP_MIN_SIZE = 1",
        f"VEHICLES = {inst.num_vehicles}",
    ]
    par_path.write_text("\n".join(par) + "\n", encoding="ascii")
    return par_path, tour_out


def _parse_tour(tour_path: Path, n_total: int) -> list[list[int]]:
    """Parse LKH-3 MTSP_SOLUTION_FILE output.  Format: one line per route,
    e.g. `Route #1: 1 -> 3 -> 9 -> 5` (LKH-3 uses 1-based ids; depot=1).
    Customer node id i maps to our customer.id == i - 1, since our depot is
    LKH node 1 and our customers map to LKH nodes 2..N+1.
    """
    if not tour_path.exists():
        return []
    routes: list[list[int]] = []
    text = tour_path.read_text(encoding="ascii", errors="ignore")
    for line in text.splitlines():
        # LKH-3 MTSP format: "1 32 35 9 48 11 1 (#5)  Cost: 2571"
        # OR "Route #1: 1 -> 32 -> 35 ..."
        m = re.match(r"\s*(?:Route\s*#?\d+\s*:)?\s*((?:\d+(?:\s*->\s*|\s+))+)", line)
        if not m:
            continue
        if "Cost:" not in line and "Cost =" not in line:
            continue
        ids = re.findall(r"\d+", m.group(1))
        r: list[int] = []
        for tok in ids:
            v = int(tok)
            if v <= 1 or v > n_total:
                continue
            r.append(v - 1)
        if r:
            routes.append(r)
    return routes


def solve(inst: Instance, settings: Settings, budget_seconds: float = 10.0) -> Solution:
    t0 = time.perf_counter()
    if not LKH_EXE.exists():
        # Stub solution
        sol = Solution(instance_id=inst.instance_id, routes=[], solver="lkh3",
                       wall_clock_seconds=0, budget_seconds=budget_seconds,
                       feasible=False, metrics={"error": -1, "operational_cost": float("inf")})
        return sol

    with tempfile.TemporaryDirectory(prefix="lkh3_") as td:
        work = Path(td)
        par_path, tour_path = _write_problem(inst, work, int(max(1, budget_seconds)))
        try:
            res = subprocess.run([str(LKH_EXE), str(par_path)],
                                 cwd=work, capture_output=True, text=True,
                                 timeout=int(budget_seconds * 3 + 30))
            err = res.returncode != 0
            if os.environ.get("SVRPTW_LKH_DEBUG"):
                (work / "stdout.log").write_text(res.stdout or "")
                (work / "stderr.log").write_text(res.stderr or "")
                shutil.copytree(work, Path("lkh3_debug"), dirs_exist_ok=True)
        except subprocess.TimeoutExpired:
            err = True
        elapsed = time.perf_counter() - t0

        raw_routes = _parse_tour(tour_path, inst.num_customers + 1) if not err else []
        routes = [Route(customers=r) for r in raw_routes if r]

    sol = Solution(
        instance_id=inst.instance_id, routes=routes, solver="lkh3",
        wall_clock_seconds=elapsed, budget_seconds=float(budget_seconds),
        feasible=False,
    )
    sol.metrics = evaluate(inst, sol, settings)
    sol.feasible = bool(sol.metrics["feasible"])
    return sol


if __name__ == "__main__":
    import argparse
    import json

    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=10.0)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), args.budget)
    print(json.dumps({"solver": sol.solver, "metrics": sol.metrics,
                      "wall_clock_s": sol.wall_clock_seconds}, indent=2))
