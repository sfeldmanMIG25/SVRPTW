"""Re-run fcv2-warm smoke and compute quality_index on both."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "D:/SVRPTW")
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from svrptw.io import load_instance
from svrptw.config import Settings
from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto
from svrptw.metrics import score_solution


def main() -> int:
    inst = load_instance("instances/v1_large/OSM-Manhattan-N0500-I000.json")
    print(f"smoke + quality on {inst.instance_id} (75s budget each)")
    print(f"{'construction':22s} {'cost':>10s} {'K':>4s} {'q_idx':>6s} {'cross':>6s} {'wall':>6s}")
    for cm in ("pyvrp", "fast_construct_v2"):
        t0 = time.perf_counter()
        sol = solve_auto(inst, Settings(), budget_seconds=75, construction=cm,
                          seed=42)  # paired seed
        wall = time.perf_counter() - t0
        score = score_solution(inst, sol)
        c = float(sol.metrics["operational_cost"])
        k = int(sol.metrics["num_vehicles_used"])
        print(f"{cm:22s} {c:>10.1f} {k:>4d} {score.quality_index:>6.3f} "
              f"{score.inter_route_crossings:>6d} {wall:>5.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
