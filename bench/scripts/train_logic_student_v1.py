"""Train v1 LogicStudent ensemble on 125 labels (96 original + 30 strong-3).

v0 separated greedy vs strong-3 (Δ 0.05) but couldn't tell portfolio/
pyvrp/auction apart (Δ 0.007). The 30 new pairs are exclusively
strong-vs-strong, so v1 should learn within-strong distinctions.

Smoke validates discrimination on N=50 Manhattan instances.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from svrptw.logic.ensemble import LogicEnsemble, train_ensemble


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pairs = json.load(open("data/logic/pairs_n50_blind_v1_126.json"))
    usable = [
        p for p in pairs
        if p.get("label") is not None
        and not (isinstance(p["label"], float) and math.isnan(p["label"]))
        and p["authoritative_a"] and p["authoritative_b"]
    ]
    print(f"[train] usable pairs: {len(usable)}/{len(pairs)}")

    out_dir = Path("models/logic/ensemble_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    ens, histories = train_ensemble(
        usable,
        instances_dir=Path("instances/v1"),
        n_heads=5,
        val_frac=0.2,
        epochs=80,
        lr=1e-3,
        batch_size=32,
        base_seed=0,
        bootstrap_frac=0.8,
        authoritative_max_std=0.12,
    )
    ens.save(out_dir)
    print(f"[train] saved ensemble to {out_dir}")
    for h in histories:
        if h["val_acc"]:
            print(f"  head {h['head']}: final val_loss={h['val_loss'][-1]:.4f}  "
                  f"final val_acc={h['val_acc'][-1]:.3f}")

    # Discrim smoke: portfolio/pyvrp/auction/greedy on 5 N=50 Manhattan instances.
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import (
        portfolio as pm, pyvrp_solver as pv, greedy as gm, auction_gart as ag
    )
    print("\n[smoke] v1 ensemble discrimination across 5 instances:")
    settings = Settings()
    paths = sorted(Path("instances/v1").glob("OSM-Manhattan-N050-I*.json"))[:5]
    by_solver: dict[str, list[float]] = {}
    for ip in paths:
        inst = load_instance(str(ip))
        sols = {
            "portfolio@8": pm.solve(inst, settings, budget_seconds=8.0),
            "pyvrp@30":    pv.solve(inst, settings, budget_seconds=30.0),
            "auction_gart": ag.solve(inst, settings),
            "greedy":       gm.solve(inst, settings),
        }
        for name, sol in sols.items():
            sc = ens.score(inst, sol)
            by_solver.setdefault(name, []).append(sc.mean)

    print(f"  {'solver':<14} {'mean':>6} {'min':>6} {'max':>6} {'spread':>7}")
    for s, vals in by_solver.items():
        mn = min(vals); mx = max(vals); avg = sum(vals)/len(vals)
        print(f"  {s:<14} {avg:>6.3f} {mn:>6.3f} {mx:>6.3f} {mx-mn:>7.3f}")

    # Inter-solver gaps for strong-3 (the v0 weakness).
    strong = ["portfolio@8", "pyvrp@30", "auction_gart"]
    means = [sum(by_solver[s])/len(by_solver[s]) for s in strong]
    print(f"\n[strong-3 spread] {max(means) - min(means):.4f}  "
          f"(v0 was 0.007; target > 0.03)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
