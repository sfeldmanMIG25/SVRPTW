"""SPEC-6-LOGIC-02 smoke: run the OpenRouter committee on one solution.

Loads Manhattan I000, runs portfolio@10, renders the solution, then
calls the committee. Prints per-model votes + consensus.

Intentionally tiny — purpose is to validate HTTP + structured output +
sanitization + aggregation end-to-end against the live API.
"""
from __future__ import annotations

import time
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.logic.committee import OpenRouterCommittee
from svrptw.solvers.classical import portfolio as pm
from svrptw.viz.renderer import render_solution


_PROMPT = """You are a delivery dispatcher reviewing a route plan.
Look at the rendered route map and decide whether you would ship
this plan tomorrow morning. Score in [0, 1] where:
  - 1.00 = ship as-is, no concerns
  - 0.50 = would rework one route
  - 0.00 = reject; structural problem

Things to weigh:
  - Visual sanity: any obviously crossing or zig-zagging routes?
  - Route balance: are the routes roughly similar length?
  - Coverage: does the plan look like it serves every stop?

Respond with the structured JSON schema only.
"""


def main() -> int:
    inst = load_instance("instances/v1/OSM-Manhattan-N050-I000.json")
    print(f"[smoke] inst={inst.instance_id}")
    t0 = time.perf_counter()
    sol = pm.solve(inst, Settings(), budget_seconds=10.0)
    print(f"[smoke] portfolio@10 done in {time.perf_counter()-t0:.1f}s, "
          f"cost={sol.metrics['operational_cost']:.1f}")

    img_path = Path("bench/runs/vivrp_imgs/smoke_committee.png")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    render_solution(inst, sol, img_path)
    print(f"[smoke] rendered → {img_path}")

    committee = OpenRouterCommittee(per_model_timeout_s=20.0)
    t0 = time.perf_counter()
    label = committee.label(inst, sol, _PROMPT, img_path)
    dt = time.perf_counter() - t0
    print(f"\n[smoke] committee done in {dt:.1f}s")
    if label.score is None:
        print(f"  NO RESPONDERS — committee returned empty")
        print(f"  rationale: {label.rationale}")
        return 1
    print(f"  consensus_score = {label.score:.3f}")
    print(f"  score_std       = {label.score_std:.3f}")
    print(f"  confidence      = {label.confidence:.3f}")
    print(f"  authoritative   = {label.authoritative}")
    print(f"  n_responders    = {label.n_responders}")
    print(f"  rationale:        {label.rationale[:200]}")
    print(f"\n  per-model votes:")
    for m, v in sorted(label.per_model.items(), key=lambda x: -x[1].score):
        print(f"    {m:<55} score={v.score:.2f} latency={v.latency_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
