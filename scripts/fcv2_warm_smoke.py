"""Smoke: solve_auto(construction='fast_construct_v2') vs default 'pyvrp'."""
from __future__ import annotations
import os, sys, time
from pathlib import Path

sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from svrptw.io import load_instance
from svrptw.config import Settings
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto


def main() -> int:
    inst = load_instance("instances/v1_large/OSM-Manhattan-N0500-I000.json")
    print(f"smoke on {inst.instance_id}")
    for cm in ("pyvrp", "fast_construct_v2"):
        t0 = time.perf_counter()
        sol = solve_auto(inst, Settings(), budget_seconds=75, construction=cm)
        wall = time.perf_counter() - t0
        c = float(sol.metrics["operational_cost"])
        k = int(sol.metrics["num_vehicles_used"])
        print(f"  construction={cm:18s}  cost={c:.1f}  K={k}  wall={wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
