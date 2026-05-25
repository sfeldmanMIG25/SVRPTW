"""iter-5v -- shift-overrun bench: does solve_auto's bandit optimize for it?

Tests on Manhattan-N500 (v1_large) at 75 s budget:

  baseline:  solve_auto at default Settings (shift_overrun coefs = 0)
  shifted:   solve_auto with shift_max=300 / shift_overrun_penalty_per_min=1.0

For each, re-evaluate the SAME solution under BOTH cost models so the
comparison is fair:
  - baseline_cost_unshifted: regular operational_cost (the headline)
  - baseline_cost_shifted:   same solution scored with shift term ON
  - shifted_cost_unshifted:  the shifted-solver's solution scored without term
  - shifted_cost_shifted:    the shifted-solver's solution scored with term

Verdict: if the bandit can optimize for shift_overrun, then:
  - shifted_cost_shifted < baseline_cost_shifted (it found a less-overruning solution)
  - the operational cost (shifted_cost_unshifted) may rise a bit, but
    the constraint-aware cost (shifted_cost_shifted) should be lower

Usage: PYTHONPATH=. python bench/scripts/iter5v_shift_overrun_bench.py
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
from svrptw.solvers.common.solution import evaluate

INST = "instances/v1_large/OSM-Manhattan-N0500-I000.json"
CAP_MIN = 300.0      # 5-hour shift cap
PEN_DOLLARS = 1.0     # $1/minute over
BUDGET = 75.0
OUT = "bench/runs/iter5v_shift_overrun_bench.json"


def run_solve(inst, settings, label: str):
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, settings, budget_seconds=BUDGET, seed=0)
    wall = time.perf_counter() - t0
    return sol, wall


def main() -> int:
    inst = load_instance(INST)
    print(f"=== iter-5v shift_overrun bench on {INST} ===")
    print(f"  budget={BUDGET}s  shift_max={CAP_MIN}min  penalty=${PEN_DOLLARS}/min")
    print()

    base_settings = Settings()
    over_settings = deepcopy(base_settings)
    over_settings.economics.shift_max_minutes = CAP_MIN
    over_settings.economics.shift_overrun_penalty_per_min = PEN_DOLLARS

    # 1. baseline solve (no shift term -- existing solve_auto behaviour)
    sol_base, wall_base = run_solve(inst, base_settings, "baseline")
    base_under_base = evaluate(inst, sol_base, base_settings)["operational_cost"]
    base_under_over = evaluate(inst, sol_base, over_settings)["operational_cost"]
    over_minutes_base = base_under_over - base_under_base

    # 2. shift-aware solve (term enabled)
    sol_over, wall_over = run_solve(inst, over_settings, "shifted")
    over_under_base = evaluate(inst, sol_over, base_settings)["operational_cost"]
    over_under_over = evaluate(inst, sol_over, over_settings)["operational_cost"]
    over_minutes_over = over_under_over - over_under_base

    K_base = int(sol_base.metrics["num_vehicles_used"])
    K_over = int(sol_over.metrics["num_vehicles_used"])

    print(f"{'solver':14s}  {'wall':>6s}  {'K':>3s}  "
          f"{'cost_no_shift':>14s}  {'cost_w_shift':>13s}  "
          f"{'overrun_$':>10s}")
    print(f"{'baseline':14s}  {wall_base:>5.1f}s  {K_base:>3d}  "
          f"{base_under_base:>14.1f}  {base_under_over:>13.1f}  "
          f"{over_minutes_base:>10.1f}")
    print(f"{'shifted':14s}  {wall_over:>5.1f}s  {K_over:>3d}  "
          f"{over_under_base:>14.1f}  {over_under_over:>13.1f}  "
          f"{over_minutes_over:>10.1f}")
    print()

    # Verdict
    overrun_reduction = over_minutes_base - over_minutes_over
    cost_increase = over_under_base - base_under_base
    print(f"=== verdict ===")
    print(f"  overrun reduction:  ${overrun_reduction:+.1f}  "
          f"({(overrun_reduction / max(1, over_minutes_base) * 100):+.1f}%)")
    print(f"  ops-cost increase:  ${cost_increase:+.1f}  "
          f"(price paid for shift-awareness)")
    print(f"  K change:           {K_over - K_base:+d}  "
          f"(more routes => less overrun per route)")

    if over_minutes_base < 1.0:
        print(f"  NOTE: baseline had no overrun ({over_minutes_base:.1f}); ",
              "term has nothing to optimize on this instance/cap.")
    elif overrun_reduction > 0:
        print(f"  OK bandit DID optimize for shift_overrun: -${overrun_reduction:.1f} in penalty.")
    else:
        print(f"  ✗ bandit did NOT reduce overrun. Either coefficient too low, ",
              "budget too short, or the operator set lacks a good shift-aware move.")

    out_payload = {
        "instance": INST, "budget_s": BUDGET, "cap_min": CAP_MIN,
        "pen_per_min": PEN_DOLLARS,
        "baseline": {"wall": wall_base, "K": K_base,
                      "cost_no_shift": base_under_base,
                      "cost_w_shift": base_under_over,
                      "overrun_dollars": over_minutes_base},
        "shifted": {"wall": wall_over, "K": K_over,
                     "cost_no_shift": over_under_base,
                     "cost_w_shift": over_under_over,
                     "overrun_dollars": over_minutes_over},
        "overrun_reduction": overrun_reduction,
        "ops_cost_increase": cost_increase,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
