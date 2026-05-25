"""iter-5v -- shift-overrun term: functional smoke + impact estimate.

1. Synthetic check: verify evaluate() applies shift_overrun_penalty linearly.
2. Re-evaluate a couple of existing wholesale solutions under
   shift_max=480, shift_overrun=$1/min to estimate how much cost changes.

Usage: PYTHONPATH=. python bench/scripts/iter5v_shift_overrun_smoke.py
"""
from __future__ import annotations

from copy import deepcopy

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import fast_construct as fc
from svrptw.solvers.common.solution import evaluate

INST = "instances/v1/OSM-Manhattan-N100-I000.json"


def main() -> int:
    inst = load_instance(INST)
    base_settings = Settings()
    sol = fc.solve(inst, base_settings, budget_seconds=2.0)

    # Baseline cost (no shift term)
    base_metrics = evaluate(inst, sol, base_settings)
    base_cost = base_metrics["operational_cost"]

    # With shift term enabled, cap=120 min (forces violation on a few routes),
    # $1/min over
    CAP_MIN = 120.0
    over_settings = deepcopy(base_settings)
    over_settings.economics.shift_max_minutes = CAP_MIN
    over_settings.economics.shift_overrun_penalty_per_min = 1.0
    over_metrics = evaluate(inst, sol, over_settings)
    over_cost = over_metrics["operational_cost"]
    delta_overrun = over_cost - base_cost

    # With shift term enabled at $0.5/min (half coefficient)
    half_settings = deepcopy(base_settings)
    half_settings.economics.shift_max_minutes = CAP_MIN
    half_settings.economics.shift_overrun_penalty_per_min = 0.5
    half_metrics = evaluate(inst, sol, half_settings)
    half_cost = half_metrics["operational_cost"]
    delta_half = half_cost - base_cost

    # Linearity check: penalty at $1 should equal exactly 2x penalty at $0.5
    expected = delta_half * 2.0
    err = abs(delta_overrun - expected)

    print(f"=== iter-5v shift_overrun smoke on {INST} ===")
    print(f"  K (n_routes used): {sol.metrics['num_vehicles_used']}")
    print(f"  baseline cost     (no shift term):           ${base_cost:.2f}")
    print(f"  cost @ shift_max={CAP_MIN:.0f} / penalty=$0.5/min:     ${half_cost:.2f}  (delta=+${delta_half:.2f})")
    print(f"  cost @ shift_max={CAP_MIN:.0f} / penalty=$1.0/min:     ${over_cost:.2f}  (delta=+${delta_overrun:.2f})")
    print(f"  linearity check |2*delta_half - delta_full| = {err:.4f}  ({'OK' if err < 0.01 else 'FAIL'})")

    # Estimate impact on a typical 100-customer instance
    # If delta_overrun > 0, at least one route exceeds 480 min
    # If delta_overrun == 0, no route exceeds 480 min (constraint already satisfied)
    print()
    print(f"=== interpretation ===")
    if delta_overrun > 0.01:
        print(f"  fast_construct produces {delta_overrun:.0f} 'over-shift' minutes total.")
        print(f"  That's the lever: a bandit aware of shift_overrun_penalty would consolidate routes")
        print(f"  to keep all <=480 min. PyVRP at default Settings cannot see this.")
    else:
        print(f"  All routes <= 480 min on this small instance.")
        print(f"  Run the smoke at v1_large N=500 to see real impact (longer routes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
