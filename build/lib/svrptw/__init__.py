"""svrptw — network-aware VRP solver with rich operational constraints.

Production entry point:

    from svrptw import Settings, load_instance, solve

    inst = load_instance("instances/v1_large/OSM-Manhattan-N0500-I000.json")
    settings = Settings()
    # Opt in to whichever constraints apply (see svrptw.constraints):
    settings.economics.shift_max_minutes = 480.0
    settings.economics.shift_overrun_penalty_per_min = 1.0
    sol = solve(inst, settings, budget_seconds=75.0)

    print(f"cost=${sol.metrics['operational_cost']:.2f}  K={sol.metrics['num_vehicles_used']}")

What you get:
- `solve(inst, settings, budget_seconds, ...)`: high-level dispatcher (auto-warms
  with PyVRP HGS construction at large N, refines via LinUCB bandit).
- `Settings`: pydantic config. `Settings.economics` is the cost-model dial; opt
  in to any of the 17 constraint terms (see svrptw.constraints).
- `load_instance(path)`: parse an Instance from JSON.
- `evaluate(inst, sol, settings)`: re-score any solution under any cost model.
- `Solution`, `Route`, `Instance`, `Customer`, `Depot`: dataclasses.

For docs on each constraint term, see `svrptw.constraints`:
    from svrptw import constraints
    constraints.print_catalog()
"""
from __future__ import annotations

# Public API surface. Keep this list short and stable; everything else is
# internal (svrptw.solvers.*, svrptw.metrics.*, svrptw.io.*).
from svrptw.config import Settings
from svrptw.io import Instance, load_instance
from svrptw.io.instance import Customer, Depot
from svrptw.solvers.common.solution import Route, Solution, evaluate

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Top-level dispatchers
    "solve",
    "solve_auto",
    # Config + I/O
    "Settings",
    "Instance",
    "Customer",
    "Depot",
    "load_instance",
    # Solution / scoring
    "Solution",
    "Route",
    "evaluate",
    # Constraint catalog (lazy import via __getattr__)
    "constraints",
]


def solve(inst: "Instance", settings: "Settings | None" = None,
          budget_seconds: float = 30.0, *, seed: int = 0, **kwargs) -> "Solution":
    """Production solver. Auto-dispatches: at small N uses vanilla portfolio
    bandit; at N>=100 uses PyVRP HGS warmstart + LinUCB bandit refinement.

    Args:
        inst: Instance to solve (load via `load_instance`).
        settings: Settings object. Defaults to bare `Settings()` (no opt-in
            constraint terms). Opt in via `settings.economics.X = value`.
        budget_seconds: Wall-clock budget. Scales down to N when very small.
        seed: RNG seed for reproducibility.
        **kwargs: Forwarded to `solve_auto` (e.g. `extra_arms`,
            `plateaus_to_stop`, `bandit_kind`, `policy_artifact`).

    Returns:
        Solution with `.routes`, `.metrics`, `.wall_clock_seconds` populated.
    """
    if settings is None:
        settings = Settings()
    from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto as _sa
    return _sa(inst, settings, budget_seconds=budget_seconds, seed=seed, **kwargs)


def solve_auto(inst: "Instance", settings: "Settings | None" = None,
               budget_seconds: float = 30.0, **kwargs) -> "Solution":
    """Same as `solve()` but allows the full kwargs of `portfolio_pyvrp_warm.solve_auto`."""
    if settings is None:
        settings = Settings()
    from svrptw.solvers.classical.portfolio_pyvrp_warm import solve_auto as _sa
    return _sa(inst, settings, budget_seconds=budget_seconds, **kwargs)


def __getattr__(name: str):
    """Lazy import for `from svrptw import constraints` and other submodules."""
    if name == "constraints":
        import importlib
        return importlib.import_module("svrptw.constraints")
    raise AttributeError(f"module 'svrptw' has no attribute {name!r}")
