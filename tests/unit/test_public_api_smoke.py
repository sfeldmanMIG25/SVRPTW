"""Public API smoke tests for the svrptw 0.1.0 release.

Validates that the top-level import surface is stable and that an
end-to-end `solve` call works with both bare Settings and stacked
constraints.

These tests are intentionally fast (small N=20 synthetic instances) so
they belong in the unit test suite, not the bench harness.
"""
from __future__ import annotations

import pytest

from svrptw.instances_gen.synthetic import generate


def test_top_level_imports_resolve():
    """from svrptw import * — every documented public name resolves."""
    import svrptw
    # Required top-level names
    for name in [
        "solve", "solve_auto",
        "Settings", "Instance", "Customer", "Depot",
        "load_instance",
        "Solution", "Route", "evaluate",
        "__version__",
    ]:
        assert hasattr(svrptw, name), f"public API missing: {name}"
    assert svrptw.__version__ == "0.1.0"


def test_constraints_catalog_resolves_via_lazy_import():
    """from svrptw import constraints — lazy import works."""
    from svrptw import constraints
    assert hasattr(constraints, "CATALOG")
    assert hasattr(constraints, "print_catalog")
    assert hasattr(constraints, "categories")
    assert len(constraints.CATALOG) >= 17  # 17+ documented terms
    cats = constraints.categories()
    assert len(cats) >= 8  # at least baseline + Phase F + iter-5v/x/y + iter-6a-* + legacy


def test_solve_baseline_runs_end_to_end():
    """svrptw.solve(inst) returns a Solution with feasible metrics."""
    from svrptw import solve
    inst = generate(N=20, seed=1001)
    sol = solve(inst, budget_seconds=4.0)
    # Solution shape
    assert sol.metrics["operational_cost"] > 0
    assert sol.metrics["num_vehicles_used"] >= 1
    assert sol.wall_clock_seconds > 0
    # Routes are populated
    served = {c for r in sol.routes for c in r.customers}
    assert len(served) > 0


def test_solve_with_stacked_constraints_runs():
    """svrptw.solve with multiple opt-in constraints completes within budget."""
    from svrptw import solve, Settings
    inst = generate(N=20, seed=1002)
    settings = Settings()
    e = settings.economics
    # Opt in to 5 representative cost terms
    e.shift_max_minutes = 480.0
    e.shift_overrun_penalty_per_min = 1.0
    e.driving_max_minutes = 90.0
    e.break_violation_penalty_per_min = 2.0
    e.embargo_window_starts = (480, 720)
    e.embargo_window_ends = (510, 750)
    e.embargo_violation_penalty_per_visit = 50.0
    e.driver_time_variance_penalty_coef = 0.5
    e.min_routes_required = 3
    e.under_min_routes_penalty_per_route = 100.0

    sol = solve(inst, settings, budget_seconds=4.0)
    # Just confirm it completes and returns a valid Solution
    assert sol.metrics["operational_cost"] > 0
    assert sol.metrics["num_vehicles_used"] >= 1


def test_evaluate_rescores_under_different_settings():
    """svrptw.evaluate re-scores any solution under any cost model."""
    from svrptw import solve, Settings, evaluate
    inst = generate(N=20, seed=1003)
    bare = Settings()
    sol = solve(inst, bare, budget_seconds=4.0)
    cost_bare = evaluate(inst, sol, bare)["operational_cost"]

    # Re-score with shift_overrun term added
    shifted = Settings()
    shifted.economics.shift_max_minutes = 60.0  # tight cap -> guaranteed overrun on small N
    shifted.economics.shift_overrun_penalty_per_min = 5.0
    cost_shifted = evaluate(inst, sol, shifted)["operational_cost"]

    # Shift-aware cost should be >= bare cost (penalty added)
    assert cost_shifted >= cost_bare
    # And should be re-scorable consistently
    cost_shifted_again = evaluate(inst, sol, shifted)["operational_cost"]
    assert cost_shifted == cost_shifted_again
