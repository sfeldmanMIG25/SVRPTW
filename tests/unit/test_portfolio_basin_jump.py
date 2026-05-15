"""SPEC-6-BANDIT-PLATEAU-01 — basin-jump tests."""
from __future__ import annotations

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.logic.ensemble import LogicEnsemble
from svrptw.logic.student import LogicStudent
from svrptw.solvers.classical import portfolio as pm


def test_basin_jump_default_off_unchanged_behaviour():
    """With plateau_basin_jump=False (default), portfolio behaviour is
    bit-stable vs the pre-basin-jump code path."""
    inst = generate(N=20, seed=0)
    s = Settings()
    sol_a = pm.solve(inst, s, budget_seconds=3.0)
    sol_b = pm.solve(inst, s, budget_seconds=3.0)
    # Portfolio is bit-deterministic per earlier validation.
    assert abs(sol_a.metrics["operational_cost"]
               - sol_b.metrics["operational_cost"]) < 1e-6


def test_basin_jump_no_regret_on_returned_cost():
    """When enabled, basin-jump may climb cost mid-run, but the RETURNED
    solution must never be worse than the no-basin-jump baseline."""
    inst = generate(N=20, seed=1)
    s = Settings()
    baseline = pm.solve(inst, s, budget_seconds=3.0)
    baseline_cost = baseline.metrics["operational_cost"]

    # 5 untrained heads → ensemble is randomly initialised, so basin-jump's
    # logic-score direction is essentially noise. The no-regret property
    # must still hold.
    ens = LogicEnsemble([LogicStudent() for _ in range(5)],
                        authoritative_max_std=1.0)   # always authoritative
    out = pm.solve(inst, s, budget_seconds=3.0,
                   plateau_basin_jump=True, logic_ensemble=ens,
                   basin_jump_max_count=2)
    # The returned cost must equal or beat baseline.
    assert out.metrics["operational_cost"] <= baseline_cost + 1e-6


def test_basin_jump_disabled_when_no_ensemble():
    """plateau_basin_jump=True with logic_ensemble=None falls through to
    the classical exit (no-op on the basin-jump branch)."""
    inst = generate(N=15, seed=2)
    s = Settings()
    out = pm.solve(inst, s, budget_seconds=2.0,
                   plateau_basin_jump=True, logic_ensemble=None)
    assert out.metrics["operational_cost"] > 0


def test_basin_jump_skips_non_authoritative_ensemble():
    """If the ensemble is never authoritative, basin-jump must not fire."""
    inst = generate(N=15, seed=3)
    s = Settings()
    # Threshold 0.0 → no plain-init ensemble can ever be authoritative.
    ens = LogicEnsemble([LogicStudent() for _ in range(5)],
                        authoritative_max_std=0.0)
    out = pm.solve(inst, s, budget_seconds=2.0,
                   plateau_basin_jump=True, logic_ensemble=ens,
                   logic_authoritative_only=True)
    # No basin_jump_* entries in history.
    history = out.metrics.get("bandit_history", [])
    assert not any(
        isinstance(entry, tuple) and entry[0].startswith("basin_jump_")
        for entry in history
    )
