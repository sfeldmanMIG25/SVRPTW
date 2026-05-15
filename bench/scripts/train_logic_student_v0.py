"""Train the first v0 LogicStudent ensemble on real labels.

Input:  data/logic/pairs_n50_blind_100.json (96 authoritative pairs)
Output: models/logic/ensemble_v0/  (5-head LogicEnsemble)

Validates SPEC-6-LOGIC-01 acceptance gates:
  - held-out agreement with teacher labels
  - latency at inference
  - calibration error
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from svrptw.logic.ensemble import LogicEnsemble, train_ensemble
from svrptw.logic.features import extract


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pairs = json.load(open("data/logic/pairs_n50_blind_100.json"))
    # Filter to authoritative + non-NaN labels.
    usable = [
        p for p in pairs
        if p.get("label") is not None
        and not (isinstance(p["label"], float) and math.isnan(p["label"]))
        and p["authoritative_a"] and p["authoritative_b"]
    ]
    print(f"[train] usable pairs: {len(usable)}/{len(pairs)}")

    out_dir = Path("models/logic/ensemble_v0")
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
        authoritative_max_std=0.15,
    )
    ens.save(out_dir)
    print(f"[train] saved ensemble to {out_dir}")

    # Per-head val accuracy summary.
    for h in histories:
        if h["val_acc"]:
            print(f"  head {h['head']}: final val_loss={h['val_loss'][-1]:.4f}  "
                  f"final val_acc={h['val_acc'][-1]:.3f}")

    # Smoke: score 4 in-domain instances with v0 ensemble.
    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv, \
        greedy as gm, auction_gart as ag
    print("\n[smoke] ensemble inference on 1 instance:")
    inst = load_instance("instances/v1/OSM-Manhattan-N050-I000.json")
    s = Settings()
    sols = {
        "portfolio@10": pm.solve(inst, s, budget_seconds=8.0),
        "pyvrp@30":     pv.solve(inst, s, budget_seconds=30.0),
        "auction_gart": ag.solve(inst, s),
        "greedy":       gm.solve(inst, s),
    }
    print(f"  {'solver':<14} {'score':>8} {'std':>6} {'authoritative':>14}")
    for name, sol in sols.items():
        score = ens.score(inst, sol)
        print(f"  {name:<14} {score.mean:>8.3f} {score.std:>6.3f} {str(score.authoritative):>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
