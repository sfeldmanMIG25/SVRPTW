"""Example 1: baseline solve, no opt-in constraints.

Run: PYTHONPATH=. python examples/01_quickstart_baseline.py
"""
from svrptw import Settings, load_instance, solve

inst = load_instance("instances/v1_large/OSM-Manhattan-N0500-I000.json")
print(f"Loaded: {inst.instance_id}  N={inst.num_customers}  K_available={inst.num_vehicles}")

settings = Settings()  # bare defaults; no opt-in cost terms
sol = solve(inst, settings, budget_seconds=30.0)

print(f"\nSolved in {sol.wall_clock_seconds:.1f}s")
print(f"  routes used:   {int(sol.metrics['num_vehicles_used'])}")
print(f"  cost:          ${sol.metrics['operational_cost']:.2f}")
print(f"  feasible:      {bool(sol.metrics.get('feasible', True))}")
print(f"  missed:        {sol.metrics.get('missed_deliveries', 0)}")
