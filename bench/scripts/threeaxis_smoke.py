"""SPEC-6-PARETO-3AXIS-01 smoke: portfolio vs pyvrp under (cost, time, logic).

Re-solves a handful of v1 N=50 Manhattan instances with both portfolio
and pyvrp, renders each solution, scores with the OpenRouter committee,
then runs pareto3 with the logic axis included.

Bounded scope: only 5 instances × 2 solvers = 10 committee calls × ~20 s
per call = ~3-4 min total. Validates the 3-axis pipeline end-to-end
without overloading free-tier API budgets.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.logic.committee import OpenRouterCommittee
from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv
from svrptw.viz.renderer import render_solution


_PROMPT = """You are a delivery dispatcher reviewing a route plan.
Score in [0, 1]: would you ship this tomorrow?
  1.00 = ship as-is
  0.50 = rework one route
  0.00 = reject; structural problem
Look for: crossings, zig-zags, route imbalance, missing coverage.
Respond with the structured JSON schema only.
"""


def _solve_and_score(solver_name, inst, committee, settings):
    if solver_name == "portfolio@10":
        sol = pm.solve(inst, settings, budget_seconds=10.0)
    else:
        sol = pv.solve(inst, settings, budget_seconds=30.0)
    sol.solver = solver_name
    img = Path(f"bench/runs/threeaxis_imgs/{inst.instance_id}__{solver_name.replace('@','_')}.png")
    img.parent.mkdir(parents=True, exist_ok=True)
    render_solution(inst, sol, img)
    t0 = time.perf_counter()
    label = committee.label(inst, sol, _PROMPT, img)
    judge_s = time.perf_counter() - t0
    return {
        "instance_id": inst.instance_id,
        "solver": solver_name,
        "operational_cost": sol.metrics["operational_cost"],
        "wall_clock_seconds": sol.wall_clock_seconds,
        "logic_score": label.score,
        "logic_score_std": label.score_std,
        "logic_authoritative": label.authoritative,
        "logic_n_responders": label.n_responders,
        "judge_wall_s": judge_s,
        "n": inst.num_customers,
        "capacity_overload": sol.metrics.get("capacity_overload", 0.0),
    }


def main() -> int:
    settings = Settings()
    committee = OpenRouterCommittee(per_model_timeout_s=20.0)
    paths = sorted(Path("instances/v1").glob("OSM-Manhattan-N050-I*.json"))[:5]

    rows = []
    for ip in paths:
        inst = load_instance(str(ip))
        for solver in ["portfolio@10", "pyvrp@30"]:
            print(f"[3axis] solving + scoring {inst.instance_id} {solver}...")
            row = _solve_and_score(solver, inst, committee, settings)
            rows.append(row)
            print(f"  cost={row['operational_cost']:.1f}  wall={row['wall_clock_seconds']:.1f}s  "
                  f"logic={row['logic_score']}  auth={row['logic_authoritative']}  "
                  f"responders={row['logic_n_responders']}  judge={row['judge_wall_s']:.1f}s")

    out = Path("bench/runs/v1_n50_manhattan_3axis_smoke.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out} ({len(rows)} rows)")

    # Run 3-axis pareto3.
    from svrptw.bench.pareto3 import dominance_report
    r2 = dominance_report(rows, challenger="portfolio@10", baseline="pyvrp@30",
                          axes=("operational_cost", "wall_clock_seconds"),
                          minimise=(True, True))
    r3 = dominance_report(rows, challenger="portfolio@10", baseline="pyvrp@30",
                          axes=("operational_cost", "wall_clock_seconds", "logic_score"),
                          minimise=(True, True, False))
    print(f"\n2-axis (cost, time):    dom {r2.headline*100:.1f}% ({r2.n_dominated}/{r2.n_total}), ties {r2.n_ties}")
    print(f"3-axis (cost,time,logic): dom {r3.headline*100:.1f}% ({r3.n_dominated}/{r3.n_total}), ties {r3.n_ties}, logic-dropped {r3.logic_dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
