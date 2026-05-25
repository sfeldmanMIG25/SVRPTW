"""iter-5u — Metric K-dependence audit + apply per_route_fixed_cost
mathematically to wholesale JSON.

Two analyses, both on existing data (no new solver runs):

1. K-fixed-cost re-leaderboard: add (n_routes * fixed) to operational_cost
   for each row; recompute the leaderboard. Tests at fixed=$50 and $100.
   Expected: fcv2 falls further behind solve_auto.

2. K-dependence correlation: for each metric (quality_index,
   inter_route_crossings, load_util_cv, mean_tw_buffer_score) compute the
   per-instance Spearman correlation with K across solvers. High |rho|
   = K-dependent metric (artifact-prone).

Usage: PYTHONPATH=. python bench/scripts/iter5u_metric_K_audit.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

WHOLESALE = "bench/runs/wholesale_v1large_full9.json"
OUT = "bench/runs/iter5u_metric_K_audit.json"


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Tie-aware Spearman rank correlation (no scipy dep)."""
    n = len(xs)
    if n < 3:
        return float("nan")
    def rank(arr):
        # average rank for ties
        idx = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[idx[j+1]] == arr[idx[i]]:
                j += 1
            r = (i + j) / 2 + 1
            for k in range(i, j+1):
                ranks[idx[k]] = r
            i = j + 1
        return ranks
    rx, ry = rank(xs), rank(ys)
    mx = sum(rx)/n; my = sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = sum((r-mx)**2 for r in rx) ** 0.5
    dy = sum((r-my)**2 for r in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main() -> int:
    rows = [r for r in json.loads(Path(WHOLESALE).read_text()) if not r.get("_failed")]

    # ---- Analysis 1: re-leaderboard with per_route_fixed_cost
    print("=== Analysis 1: re-leaderboard with per_route_fixed_cost ===")
    for fixed in [0, 50, 100]:
        by_solver = defaultdict(list)
        for r in rows:
            adj = r["operational_cost"] + fixed * r["n_routes"]
            by_solver[r["solver"]].append(adj)
        ordered = sorted(((s, sum(c)/len(c)) for s, c in by_solver.items()),
                         key=lambda x: x[1])
        print(f"\n  per_route_fixed_cost=${fixed}:")
        print(f"    {'rank':>4s} {'solver':22s} {'mean_cost_adj':>14s}")
        for i, (s, mc) in enumerate(ordered, 1):
            print(f"    {i:>4d} {s:22s} {mc:>14.1f}")

    # ---- Analysis 2: per-instance K-dependence correlation
    print("\n=== Analysis 2: K-dependence correlation per instance ===")
    print("  (Spearman rho across solvers within one instance; |rho|>0.7 = K-driven)")
    metrics = ["quality_index", "inter_route_crossings",
               "load_util_cv", "mean_tw_buffer_score"]
    by_inst = defaultdict(list)
    for r in rows:
        # exclude lkh3 which has no feasible solution -- it skews the metric
        if r["solver"] == "lkh3" and r["operational_cost"] > 100000:
            continue
        by_inst[r["instance_id"]].append(r)

    rho_means = {m: [] for m in metrics}
    print(f"\n  {'instance':32s}", end="")
    for m in metrics:
        print(f" {m[:14]:>15s}", end="")
    print()
    for iid in sorted(by_inst):
        rs = by_inst[iid]
        if len(rs) < 4:
            continue
        Ks = [r["n_routes"] for r in rs]
        print(f"  {iid:32s}", end="")
        for m in metrics:
            ys = [r.get(m, 0) for r in rs]
            rho = _spearman(Ks, ys)
            rho_means[m].append(rho)
            print(f" {rho:>+15.3f}", end="")
        print()
    print(f"\n  {'MEAN |rho|':>32s}", end="")
    for m in metrics:
        rhos = [r for r in rho_means[m] if r == r]  # drop nan
        mr = sum(abs(r) for r in rhos)/len(rhos) if rhos else float("nan")
        print(f" {mr:>+15.3f}", end="")
    print()

    payload = {
        "fixed_cost_releaderboard": {
            f"${f}": {s: sum(c)/len(c) for s, c in
                      ((s2, [r["operational_cost"] + f * r["n_routes"]
                             for r in rows if r["solver"] == s2])
                       for s2 in {r["solver"] for r in rows})}
            for f in [0, 50, 100]
        },
        "K_dependence_mean_abs_rho": {
            m: sum(abs(r) for r in rho_means[m] if r == r) / max(1, sum(1 for r in rho_means[m] if r == r))
            for m in metrics
        },
    }
    Path(OUT).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
