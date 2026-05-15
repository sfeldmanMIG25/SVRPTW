"""Quick multi-instance check: does v0 LogicEnsemble discriminate solvers?

Score 4 solvers across 5 N=50 Manhattan instances. If mean score per
solver collapses to ~one value across all instances, the ensemble
isn't useful as a logic-axis prior. If means separate, we can wire it
into the portfolio bandit.

Bounded: 5 instances * 4 solvers = 20 solves. Greedy is ~ms; pyvrp@30
is the long pole. Total ~3 min.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.logic.ensemble import LogicEnsemble
from svrptw.solvers.classical import (
    auction_gart as ag,
    greedy as gm,
    portfolio as pm,
    pyvrp_solver as pv,
)


def main() -> int:
    ens = LogicEnsemble.load(Path("models/logic/ensemble_v0"))
    settings = Settings()
    paths = sorted(Path("instances/v1").glob("OSM-Manhattan-N050-I*.json"))[:5]

    rows: list[dict] = []
    for ip in paths:
        inst = load_instance(str(ip))
        sols = {
            "portfolio@8": pm.solve(inst, settings, budget_seconds=8.0),
            "pyvrp@30":    pv.solve(inst, settings, budget_seconds=30.0),
            "auction_gart": ag.solve(inst, settings),
            "greedy":       gm.solve(inst, settings),
        }
        for name, sol in sols.items():
            t0 = time.perf_counter()
            score = ens.score(inst, sol)
            inf_ms = (time.perf_counter() - t0) * 1000
            rows.append({
                "instance_id": inst.instance_id,
                "solver": name,
                "operational_cost": sol.metrics["operational_cost"],
                "logic_mean": score.mean,
                "logic_std":  score.std,
                "authoritative": score.authoritative,
                "infer_ms": inf_ms,
                "feasible": bool(sol.metrics.get("feasible", True)),
                "capacity_overload": float(sol.metrics.get("capacity_overload", 0.0)),
            })
            print(f"{inst.instance_id} {name:<13} cost={sol.metrics['operational_cost']:.1f}  "
                  f"logic={score.mean:.3f}±{score.std:.3f}  auth={score.authoritative}  "
                  f"infer={inf_ms:.1f}ms")

    out = Path("bench/runs/ensemble_v0_discrim_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    # Aggregate per-solver mean and spread to see if ensemble separates them.
    by_solver: dict[str, list[float]] = {}
    for r in rows:
        by_solver.setdefault(r["solver"], []).append(r["logic_mean"])
    print("\n[summary] per-solver logic mean across instances:")
    print(f"  {'solver':<13} {'mean':>6} {'min':>6} {'max':>6} {'spread':>7}")
    for s, vals in by_solver.items():
        mn = min(vals); mx = max(vals); avg = sum(vals)/len(vals)
        print(f"  {s:<13} {avg:>6.3f} {mn:>6.3f} {mx:>6.3f} {mx-mn:>7.3f}")
    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
