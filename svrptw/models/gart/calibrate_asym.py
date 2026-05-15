"""Fit the asymmetric-correction beta in SPEC-1-GART-01:

    L_asym ≈ alpha * MST_undirected * (1 + beta * asymmetry_score)

Procedure: generate K random asymmetric distance matrices, query LKH-3 for the
exact-ish ATSP ground truth, query our GartV4Estimator with beta=0 for its
prediction, then fit beta by least squares on:

    log(L_true / (alpha * MST)) ≈ log(1 + beta * asym)  →  linear regression

Output: models/gart/calibration_asym.npz with keys {beta, samples, residuals}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from svrptw.models.gart import get_default_estimator
from svrptw.models.gart.estimator import _asymmetry_score
from svrptw.models.gart.oracle import lkh_atsp_tour_length


def _random_asym_matrix(n: int, asym_strength: float, rng: np.random.Generator) -> np.ndarray:
    """Euclidean base + asymmetric perturbation.  asym_strength in [0, 0.5]
    controls roughly how strong the directional bias is."""
    pts = rng.uniform(0.0, 100.0, size=(n, 2))
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    # Per-edge multiplicative perturbation, independently chosen for each direction.
    noise = rng.uniform(1.0 - asym_strength, 1.0 + asym_strength, size=D.shape)
    D = D * noise
    np.fill_diagonal(D, 0.0)
    return D


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=80, help="number of random asym matrices")
    p.add_argument("--n-range", default="10,30", help="N range as 'lo,hi'")
    p.add_argument("--asym-range", default="0.0,0.5", help="asym strength range")
    p.add_argument("--lkh-budget", type=float, default=3.0, help="seconds per LKH ATSP run")
    p.add_argument("--out", default="models/gart/calibration_asym.npz")
    p.add_argument("--report", default="models/gart/calibration_asym.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    n_lo, n_hi = (int(x) for x in args.n_range.split(","))
    a_lo, a_hi = (float(x) for x in args.asym_range.split(","))

    est = get_default_estimator()
    asyms: list[float] = []
    ratios: list[float] = []     # L_true / (alpha * MST_undirected)
    rows = []
    t0 = time.perf_counter()

    for i in range(args.samples):
        n = int(rng.integers(n_lo, n_hi + 1))
        a = float(rng.uniform(a_lo, a_hi))
        M = _random_asym_matrix(n, a, rng)
        asym_score = _asymmetry_score(M)

        # Ground truth from LKH-3 ATSP.
        L_true = lkh_atsp_tour_length(M, time_limit=args.lkh_budget)
        if not np.isfinite(L_true) or L_true <= 0:
            continue

        # GartV4 prediction with beta=0 (so we measure the bare alpha*MST).
        prev_beta = est.asymmetry_beta
        est.asymmetry_beta = 0.0
        L_gart = est.estimate(
            np.arange(1, n, dtype=np.int64),  # 'customers' are indices 1..n-1; depot is 0
            dist_matrix=M,
        )
        est.asymmetry_beta = prev_beta

        if L_gart <= 0:
            continue
        ratio = L_true / L_gart
        asyms.append(asym_score)
        ratios.append(ratio)
        rows.append({"n": n, "asym_target": a, "asym_score": asym_score,
                     "L_true": L_true, "L_gart_beta0": L_gart, "ratio": ratio})
        elapsed = time.perf_counter() - t0
        print(f"  [{i + 1}/{args.samples}] n={n} asym={asym_score:.3f} "
              f"L_true={L_true:.2f} L_gart={L_gart:.2f} ratio={ratio:.3f}  "
              f"({elapsed:.1f}s)")

    if len(asyms) < 8:
        print("ERROR: too few valid samples to fit beta")
        return 1

    # Fit: ratio - 1 ≈ beta * asym_score (linear in beta).
    a_arr = np.array(asyms)
    r_arr = np.array(ratios)
    beta = float(np.sum((r_arr - 1.0) * a_arr) / np.sum(a_arr ** 2))
    pred = 1.0 + beta * a_arr
    resid = r_arr - pred
    r2 = 1.0 - float(np.sum(resid ** 2) / np.sum((r_arr - r_arr.mean()) ** 2 + 1e-12))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, beta=beta, asyms=a_arr, ratios=r_arr, residuals=resid)
    Path(args.report).write_text(json.dumps({
        "beta": beta, "r2": r2, "samples": len(asyms),
        "rows": rows, "config": {"n_range": args.n_range, "asym_range": args.asym_range,
                                  "lkh_budget": args.lkh_budget, "seed": args.seed},
    }, indent=2))
    print(f"\nFitted beta = {beta:.4f}, R^2 = {r2:.3f}, samples = {len(asyms)}")
    print(f"  saved: {out}\n  report: {args.report}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
