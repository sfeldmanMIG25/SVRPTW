"""Compare warmstart strategies: PyVRP@5s vs LKH-3@5s vs auction@5s.

The session's key finding was that PyVRP construction is better than
auction_gart at large N. Question: is LKH-3 (which we have locally
at C:/LKH/LKH-3.exe) even better as a warmstart? It's the
state-of-the-art TSP solver and handles CVRP variants.

8 v1 N=200 instances (one per city), 3 warmstart variants, 25s
portfolio bandit refinement after each.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm
from svrptw.solvers.classical import auction_gart as ag
from svrptw.solvers.classical import pyvrp_solver as pv


def main() -> int:
    try:
        from svrptw.solvers.classical import lkh3
        has_lkh = True
    except Exception:
        has_lkh = False

    cities = ["Manhattan", "Paris", "SanFrancisco", "Phoenix",
              "Charleston", "Austin", "Pittsburgh", "Cambridge"]
    paths = [Path(f"instances/v1/OSM-{c}-N200-I000.json") for c in cities]
    rows = []
    for ip in paths:
        if not ip.exists(): continue
        inst = load_instance(str(ip))
        s = Settings()
        # 1) auction warmstart (baseline)
        warm_auc = ag.solve(inst, s)
        # 2) PyVRP warmstart
        t0 = time.perf_counter()
        warm_py = pv.solve(inst, s, budget_seconds=5.0)
        py_wall = time.perf_counter() - t0
        # 3) LKH-3 warmstart (if available)
        warm_lkh = None
        lkh_wall = 0.0
        if has_lkh:
            try:
                t0 = time.perf_counter()
                warm_lkh = lkh3.solve(inst, s)
                lkh_wall = time.perf_counter() - t0
            except Exception as e:
                sys.stdout.write(f"  lkh3 failed on {inst.instance_id}: {e}\n")
                sys.stdout.flush()

        # Refine each with portfolio bandit for ~25s.
        port_auc = pm.solve(inst, s, budget_seconds=25.0, initial_solution=warm_auc)
        port_py  = pm.solve(inst, s, budget_seconds=25.0, initial_solution=warm_py)
        port_lkh = pm.solve(inst, s, budget_seconds=25.0, initial_solution=warm_lkh) if warm_lkh else None

        rec = {
            "instance_id": inst.instance_id,
            "auction_warm_cost":  warm_auc.metrics["operational_cost"],
            "pyvrp_warm_cost":    warm_py.metrics["operational_cost"],
            "lkh_warm_cost":      warm_lkh.metrics["operational_cost"] if warm_lkh else None,
            "pyvrp_warm_wall":    py_wall,
            "lkh_warm_wall":      lkh_wall,
            "port_after_auction": port_auc.metrics["operational_cost"],
            "port_after_pyvrp":   port_py.metrics["operational_cost"],
            "port_after_lkh":     port_lkh.metrics["operational_cost"] if port_lkh else None,
        }
        rows.append(rec)
        lkh_str = f"{rec['port_after_lkh']:.1f}" if rec['port_after_lkh'] is not None else "N/A"
        sys.stdout.write(
            f"  {inst.instance_id:<32}  "
            f"port_auc={rec['port_after_auction']:.1f}  "
            f"port_py={rec['port_after_pyvrp']:.1f}  "
            f"port_lkh={lkh_str}\n"
        )
        sys.stdout.flush()

    Path("bench/runs/warmstart_construction_comparison.json").write_text(json.dumps(rows, indent=2))

    print("\n[summary] mean post-bandit cost by warmstart strategy:")
    print(f"  {'strategy':<20} {'mean cost':>10} {'mean construction wall':>22}")
    aucs = [r["port_after_auction"] for r in rows]
    pys  = [r["port_after_pyvrp"] for r in rows]
    lkhs = [r["port_after_lkh"] for r in rows if r["port_after_lkh"] is not None]
    print(f"  {'auction':<20} {sum(aucs)/len(aucs):>10.1f} {'<1s':>22}")
    print(f"  {'pyvrp@5s':<20} {sum(pys)/len(pys):>10.1f} {sum(r['pyvrp_warm_wall'] for r in rows)/len(rows):>22.2f}")
    if lkhs:
        lkhw = [r['lkh_warm_wall'] for r in rows if r['port_after_lkh'] is not None]
        print(f"  {'lkh-3':<20} {sum(lkhs)/len(lkhs):>10.1f} {sum(lkhw)/len(lkhw):>22.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
