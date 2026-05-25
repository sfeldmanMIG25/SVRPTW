"""Example 2: solve with multiple operational constraints stacked.

Demonstrates the full constraint catalog by enabling:
  - shift_overrun (8-hour shift cap)
  - driver_breaks (EU 561 4.5h driving cap)
  - embargo (8-8:30am school zone + 12-12:30pm lunch)
  - mixed_fleets (small/large vehicle classes)
  - min_routes (labor minimum 25)

All stack without slowdown thanks to the Phase F lite refactor.

Run: PYTHONPATH=. python examples/02_stacked_constraints.py
"""
from svrptw import Settings, load_instance, solve

inst = load_instance("instances/v1_large/OSM-Manhattan-N0500-I000.json")
settings = Settings()
e = settings.economics

# ─── Driver shift constraint (iter-5v) ────────────────────────────
e.shift_max_minutes = 480.0
e.shift_overrun_penalty_per_min = 1.0

# ─── EU 561 driver breaks (iter-6a-1) ─────────────────────────────
e.driving_max_minutes = 270.0
e.break_violation_penalty_per_min = 2.0

# ─── Hard zones / embargo windows (iter-6a-2) ─────────────────────
e.embargo_window_starts = (480, 720)
e.embargo_window_ends = (510, 750)
e.embargo_violation_penalty_per_visit = 50.0

# ─── Mixed fleets (iter-6a-3) ─────────────────────────────────────
cap = float(inst.vehicle_capacity)
e.vehicle_class_capacities = (cap * 0.5, cap)
e.vehicle_class_fixed_premiums = (5.0, 20.0)
e.vehicle_class_per_mile_premiums = (0.0, 0.1)

# ─── Labor minimum routes (iter-6a-8) ─────────────────────────────
e.min_routes_required = 18
e.under_min_routes_penalty_per_route = 100.0

sol = solve(inst, settings, budget_seconds=75.0)

print(f"Solved with 5 stacked constraints in {sol.wall_clock_seconds:.1f}s")
print(f"  routes used:   {int(sol.metrics['num_vehicles_used'])}")
print(f"  cost (full):   ${sol.metrics['operational_cost']:.2f}")
print(f"  feasible:      {bool(sol.metrics.get('feasible', True))}")

# Decompose the cost by re-scoring under bare settings
from svrptw import evaluate
bare = Settings()
ops_only = evaluate(inst, sol, bare)['operational_cost']
print(f"  ops cost only: ${ops_only:.2f}")
print(f"  constraint penalties: ${sol.metrics['operational_cost'] - ops_only:.2f}")
