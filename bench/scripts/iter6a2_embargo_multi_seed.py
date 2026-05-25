"""Multi-seed stability bench for iter-6a-2 embargo (the headline single-term claim).

Tests whether the +$1796/inst mean win from the 6-instance bench is seed-robust
in addition to instance-robust. Runs 3 seeds on 1 instance.
"""
from __future__ import annotations
import json, sys, time, statistics
from copy import deepcopy
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
from svrptw.solvers.common.solution import evaluate

INST = sys.argv[1] if len(sys.argv) > 1 else "instances/v1_large/OSM-Manhattan-N0500-I000.json"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 75.0

PEAK_STARTS = (480, 720)
PEAK_ENDS = (510, 750)
PEN_PER_VISIT = 50.0


def _run_seed(inst, s_base, s_em, seed):
    t0 = time.perf_counter()
    sol_base = pw.solve_auto(inst, s_base, budget_seconds=BUDGET, seed=seed)
    wall_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    sol_em = pw.solve_auto(inst, s_em, budget_seconds=BUDGET, seed=seed)
    wall_e = time.perf_counter() - t0
    c_b = evaluate(inst, sol_base, s_em)["operational_cost"]
    c_e = evaluate(inst, sol_em, s_em)["operational_cost"]
    return {"seed": seed,
            "K_base": int(sol_base.metrics["num_vehicles_used"]),
            "K_em": int(sol_em.metrics["num_vehicles_used"]),
            "cost_base_w_em": c_b, "cost_em_w_em": c_e,
            "net": c_b - c_e, "wall_b": wall_b, "wall_e": wall_e}


def main() -> int:
    inst = load_instance(INST)
    s_base = Settings()
    s_em = deepcopy(s_base)
    s_em.economics.embargo_window_starts = PEAK_STARTS
    s_em.economics.embargo_window_ends = PEAK_ENDS
    s_em.economics.embargo_violation_penalty_per_visit = PEN_PER_VISIT
    print(f"=== iter-6a-2 embargo multi-seed stability ===")
    print(f"  instance: {INST}  budget={BUDGET}s  n_seeds={N_SEEDS}")
    print(f"  windows: {PEAK_STARTS}-{PEAK_ENDS} pen=${PEN_PER_VISIT}/visit")
    runs = []
    for sd in range(N_SEEDS):
        print(f"\n[seed {sd}] solving ...", flush=True)
        r = _run_seed(inst, s_base, s_em, sd)
        runs.append(r)
        print(f"  K {r['K_base']}->{r['K_em']} net=${r['net']:+.1f}", flush=True)
    nets = [r["net"] for r in runs]
    mean_net = statistics.mean(nets)
    sd_net = statistics.stdev(nets) if len(nets) >= 2 else 0.0
    print(f"\n=== aggregate (n={N_SEEDS}) ===")
    print(f"  net under embargo obj: ${mean_net:+.1f} +/- ${sd_net:.1f}/inst")
    print(f"  individual nets:       {[f'${n:+.0f}' for n in nets]}")
    out = Path(f"bench/runs/iter6a2_embargo_multi_seed_{Path(INST).stem}.json")
    out.write_text(json.dumps({"inst": INST, "n_seeds": N_SEEDS, "budget_s": BUDGET,
                                 "runs": runs, "mean_net": mean_net, "sd_net": sd_net,
                                 "verdict_positive": mean_net > 0,
                                 "all_seeds_positive": all(n > 0 for n in nets)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
