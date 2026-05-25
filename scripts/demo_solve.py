"""Tiny demo: run solve_auto on one instance, push live snapshots + curve."""
from __future__ import annotations

import os
import time
import sys
from pathlib import Path

# Default to the local server unless the env already has one.
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from svrptw.io import load_instance
from svrptw.config import Settings
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
from webui.snapshot import make_on_accept
from webui import client as ui


def main(inst_path: str, budget: float = 10.0) -> int:
    inst = load_instance(inst_path)
    cost_key = "operational_cost"
    routes_key = "num_vehicles_used"
    print(f"instance: {inst.instance_id}, N={inst.num_customers}")

    ui.push_stage("demo-solve", f"live demo: solve_auto on {inst.instance_id}",
                  N=inst.num_customers)
    ui.push_agent("demo-solve", "running",
                  summary=f"live solve_auto on {inst.instance_id}, budget={budget:.0f}s")
    ui.push_log(f"demo solve started on {inst.instance_id} (N={inst.num_customers}, budget={budget:.0f}s)")

    stream = f"warm-{inst.instance_id}"
    on_accept = make_on_accept(inst, stream=stream,
                                every_n_accepts=2, min_improvement=1.0)

    t0 = time.perf_counter()
    sol = solve_auto(inst, Settings(), budget_seconds=budget, on_accept=on_accept)
    dt = time.perf_counter() - t0

    cost = float(sol.metrics[cost_key])
    routes = int(sol.metrics[routes_key])
    print(f"done in {dt:.1f}s; cost={cost:.2f}, routes={routes}")
    ui.push_agent("demo-solve", "completed",
                  summary=f"cost={cost:.1f}, routes={routes}, wall={dt:.1f}s")
    ui.push_log(f"demo solve finished: cost={cost:.1f}, {routes} routes, {dt:.1f}s")
    return 0


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "instances/v1/OSM-Manhattan-N100-I003.json"
    b = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    raise SystemExit(main(p, b))
