"""K-fairness audit: does fcv2's quality lead come from real basin
advantage, or just from refusing to consolidate routes?

Hypothesis: fcv2 uses 1.7-3.3x more routes than solve_auto on v1_large.
More routes => fewer customers per route => fewer crossings, easier to
cluster => higher quality_index. If we cap fcv2 at solve_auto's K, the
quality lead should shrink or disappear.

Test:
1. Get solve_auto's K on each instance from the wholesale JSON.
2. Run fcv2 with inst.num_vehicles overridden to solve_auto's K.
3. Compare cost+quality.
4. Conclusion: if fcv2 still wins quality at K-fair => real basin advantage.
                If quality drops to solve_auto level => K-mismatch artifact.

Usage: PYTHONPATH=. python bench/scripts/fcv2_k_fair_audit.py
"""
from __future__ import annotations

import json
import time
from copy import copy
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.metrics import score_solution
from svrptw.solvers.classical import fast_construct_v2 as fc2


WHOLESALE = "bench/runs/wholesale_v1large_full9.json"
OUT = "bench/runs/fcv2_k_fair_audit.json"
LOG = "bench/runs/fcv2_k_fair_audit.log"


def main() -> int:
    rows = json.loads(Path(WHOLESALE).read_text())
    sa_K = {r["instance_id"]: r["n_routes"] for r in rows
            if not r.get("_failed") and r["solver"] == "solve_auto"}
    fcv2_baseline = {r["instance_id"]: r for r in rows
                     if not r.get("_failed") and r["solver"] == "fast_construct_v2"}

    out_rows = []
    log_lines = ["K-fair audit: fcv2 capped at solve_auto's K vs fcv2 native K"]
    log_lines.append(f"  {'instance':32s}  {'sa_K':>4s}  {'native_K':>8s}  "
                     f"{'native_cost':>11s}  {'native_q':>8s}  "
                     f"{'capK_K':>6s}  {'capK_cost':>9s}  {'capK_q':>7s}  "
                     f"{'d_cost':>7s}  {'d_q':>7s}")
    settings = Settings()

    # N=500 instances only -- quickest, captures the main pattern.
    for iid, K in sorted(sa_K.items()):
        if "N0500" not in iid:
            continue
        inst_path = next((r["instance_path"] for r in rows
                          if r.get("instance_id") == iid), None)
        if not inst_path or not Path(inst_path).exists():
            continue
        inst = load_instance(inst_path)
        # Override num_vehicles to solve_auto's K (force fcv2 to consolidate)
        inst_capped = copy(inst)
        inst_capped.num_vehicles = K

        # Run fcv2 with capped K
        t0 = time.perf_counter()
        sol_cap = fc2.solve(inst_capped, settings, budget_seconds=10.0, seed=0)
        wall_cap = time.perf_counter() - t0

        # Score it (under un-capped settings so cost is comparable)
        score_cap = score_solution(inst_capped, sol_cap)
        cap_cost = float(sol_cap.metrics["operational_cost"])
        cap_K = int(sol_cap.metrics["num_vehicles_used"])
        cap_q = score_cap.quality_index

        baseline = fcv2_baseline.get(iid, {})
        nat_cost = baseline.get("operational_cost", float("nan"))
        nat_q = baseline.get("quality_index", float("nan"))
        nat_K = baseline.get("n_routes", -1)

        log_lines.append(
            f"  {iid:32s}  {K:>4d}  {nat_K:>8d}  {nat_cost:>11.1f}  "
            f"{nat_q:>8.3f}  {cap_K:>6d}  {cap_cost:>9.1f}  {cap_q:>7.3f}  "
            f"{cap_cost - nat_cost:>+7.1f}  {cap_q - nat_q:>+7.3f}"
        )
        out_rows.append({
            "instance_id": iid, "sa_K": K, "fcv2_native_K": nat_K,
            "fcv2_native_cost": nat_cost, "fcv2_native_q": nat_q,
            "fcv2_capK_K": cap_K, "fcv2_capK_cost": cap_cost,
            "fcv2_capK_q": cap_q, "wall_capped": wall_cap,
            "feasible": bool(sol_cap.metrics.get("feasible", True)),
        })

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out_rows, indent=2))
    Path(LOG).write_text("\n".join(log_lines) + "\n")
    for line in log_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
