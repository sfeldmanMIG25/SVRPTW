"""Bench: portfolio-13arm vs portfolio-14arm (with LLM-accepted spatial_centroid_swap).

The LLM-generated `spatial_centroid_swap` cleared paired-seeding
composability at +$4.40 mean_delta on 50% hit rate across 8 mixed
instances. This bench validates that the win generalises beyond the
composability subset on a wider 15-instance Manhattan sample.

Paired seeding: same seed for both 13-arm and 14-arm runs each rep.
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm
from svrptw.solvers.common.llm_operators.spatial_centroid_swap import operator as scs


def main() -> int:
    settings = Settings()
    paths = sorted(Path("instances/v1").glob("OSM-Manhattan-N050-I*.json"))[:5] + \
            sorted(Path("instances/v1").glob("OSM-Austin-N050-I*.json"))[:5] + \
            sorted(Path("instances/v1").glob("OSM-Paris-N050-I*.json"))[:5]
    extra = {"spatial_centroid_swap": scs}
    rows = []
    for ip in paths:
        inst = load_instance(str(ip))
        for rep in range(3):
            base = pm.solve(inst, settings, budget_seconds=10.0, seed=rep)
            aug  = pm.solve(inst, settings, budget_seconds=10.0, seed=rep, extra_arms=extra)
            rows.append({
                "instance_id": inst.instance_id, "rep": rep,
                "base_cost": base.metrics["operational_cost"],
                "aug_cost":  aug.metrics["operational_cost"],
                "delta":     base.metrics["operational_cost"] - aug.metrics["operational_cost"],
            })

    print(f"\n[summary] portfolio_13arm vs portfolio_14arm (15 instances x 3 reps = {len(rows)} pairs)")
    deltas = [r["delta"] for r in rows]
    n = len(deltas)
    mean = sum(deltas) / n
    wins = sum(1 for d in deltas if d > 1.0)
    losses = sum(1 for d in deltas if d < -1.0)
    ties = n - wins - losses
    print(f"  mean delta:    {mean:+.2f}")
    print(f"  wins:          {wins}/{n} ({wins/n*100:.0f}%)")
    print(f"  losses:        {losses}/{n} ({losses/n*100:.0f}%)")
    print(f"  ties (|d|<$1): {ties}/{n}")

    Path("bench/runs/portfolio_with_llm_arm.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
