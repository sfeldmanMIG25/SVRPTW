"""SPEC-8-COUNCIL-02 composability bench — run all current seeds.

Tests whether each seed's improvement on greedy starts (where the
strict single-shot bench accepted them) translates to actual help
when added as a 14th bandit arm against the 13-arm production
portfolio. Expectation: cross_route_2opt rejected (PyVRP-redundant),
gart_guided_ruin likely rejected too (greedy hits only 1/5 → unlikely
to surface useful moves the bandit can't already find).
"""
from __future__ import annotations

import json
from pathlib import Path

from svrptw.council.shadow_bench import composability_run
from svrptw.council.seeds.cross_route_2opt import operator as cross_2opt
from svrptw.council.seeds.gart_guided_ruin import operator as gart_ruin


def main() -> int:
    seeds = {
        "cross_route_2opt": cross_2opt,
        "gart_guided_ruin": gart_ruin,
    }
    summary = []
    for name, op in seeds.items():
        print(f"\n=== composability: {name} ===")
        r = composability_run(name, op, budget_seconds=10.0, n_repeats=3)
        print(f"  mean_delta:    {r['mean_abs_delta']:+.2f}")
        print(f"  hit_rate:      {r['hit_rate']:.2f}")
        print(f"  worst_regress: {r['worst_regress_pct']*100:.2f}%")
        print(f"  accepted:      {r['accepted']}")
        summary.append({"seed": name, "mean_delta": r["mean_abs_delta"],
                        "hit_rate": r["hit_rate"], "accepted": r["accepted"]})
    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['seed']:<22} mean={s['mean_delta']:+8.2f}  "
              f"hit={s['hit_rate']:.2f}  accepted={s['accepted']}")
    Path("bench/runs/council/composability/summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
