"""Phase G3 smoke tests for fast_construct_v2.

The graph machinery is exercised opportunistically when an OSM-cached
instance is available; otherwise the synthetic-instance path delegates
to fast_construct (v1) since N < 200, which is the contract under
test here.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.solvers.classical import fast_construct as fc_v1
from svrptw.solvers.classical.fast_construct_v2 import solve as fc_v2_solve


def test_returns_feasible_at_n100_via_fallback():
    """N=100 < _FALLBACK_N: must delegate to v1 and produce a feasible
    solution with the v2 solver tag.
    """
    inst = generate(N=100, seed=41)
    sol = fc_v2_solve(inst, Settings(), budget_seconds=1.0, seed=0)
    assert sol.solver == "fast_construct_v2"
    assert math.isfinite(sol.metrics["operational_cost"])
    assert sol.metrics["num_vehicles_used"] > 0
    assert sol.feasible, f"expected feasible solution, got {sol.metrics}"


def test_v2_honors_budget_with_first_call_slack():
    """budget=1.0s; allow 50% slack to absorb first-call graph fetch.
    Synthetic instance => v1 fallback; same wall constraint applies.
    """
    inst = generate(N=120, seed=53)
    budget = 1.0
    sol = fc_v2_solve(inst, Settings(), budget_seconds=budget, seed=0)
    assert sol.wall_clock_seconds <= budget * 1.5, (
        f"v2 wall {sol.wall_clock_seconds:.3f}s > {budget*1.5:.3f}s"
    )


def test_v2_dispatches_to_v1_below_threshold():
    """Below the fallback threshold (N < 200) v2 must delegate to v1.
    The v1 polish step is wall-clock-dependent so back-to-back call
    costs may differ slightly; we check both produce comparable finite
    costs and that v2 carries the v2 solver tag.
    """
    inst = generate(N=80, seed=67)
    s = Settings()
    sol_v1 = fc_v1.solve(inst, s, budget_seconds=1.0, seed=0)
    sol_v2 = fc_v2_solve(inst, s, budget_seconds=1.0, seed=0)
    assert sol_v2.solver == "fast_construct_v2"
    assert sol_v1.solver == "fast_construct"
    c1 = float(sol_v1.metrics["operational_cost"])
    c2 = float(sol_v2.metrics["operational_cost"])
    assert math.isfinite(c1) and math.isfinite(c2)
    rel = abs(c1 - c2) / max(abs(c1), 1e-6)
    assert rel < 0.10, (
        f"v1 cost {c1:.3f} vs v2 cost {c2:.3f} rel-diff {rel:.3f}"
    )


def test_v2_solver_tag_set():
    inst = generate(N=60, seed=71)
    sol = fc_v2_solve(inst, Settings(), budget_seconds=0.5, seed=0)
    assert sol.solver == "fast_construct_v2"


def test_v2_merge_polish_imports_present():
    """iter-7-bis Bug 2 fix: the v2 module must import merge_routes for
    the post-community-NN consolidation pass. Guards against accidental
    removal of the merge import that fixed the K-fragmentation bug.
    """
    from svrptw.solvers.classical import fast_construct_v2 as mod
    # merge_routes was added in iter-7-bis; verify the symbol is bound
    # in module scope (used inside solve()).
    assert hasattr(mod, "merge_routes"), (
        "fast_construct_v2 must import merge_routes for post-construction "
        "consolidation polish (iter-7-bis Bug 2 fix)"
    )


def test_v2_solve_signature_unchanged_by_merge_fix():
    """Guard: the iter-7-bis Bug 2 fix only added code inside solve();
    the public signature must stay (inst, settings, budget_seconds=1.0,
    seed=0, *, n_starts=2, G=None). Catches accidental param drift."""
    import inspect
    from svrptw.solvers.classical.fast_construct_v2 import solve
    sig = inspect.signature(solve)
    params = list(sig.parameters.keys())
    assert params[:4] == ["inst", "settings", "budget_seconds", "seed"], (
        f"v2.solve signature drift: got {params}"
    )
    assert "n_starts" in params
    assert "G" in params
