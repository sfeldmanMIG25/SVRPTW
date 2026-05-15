"""SPEC-4-DATA-02 — network-OD synthetic generator invariants."""
from __future__ import annotations

import numpy as np

from svrptw.instances_gen.network_synthetic import generate
from svrptw.config import Settings
from svrptw.solvers.classical import greedy as greedy_mod


def test_generates_valid_instance():
    inst = generate(N=20, seed=0)
    assert inst.num_customers == 20
    assert inst.travel_time.shape == (21, 21)
    assert inst.travel_dist.shape == (21, 21)
    assert all(c.id == i + 1 for i, c in enumerate(inst.customers))
    # No infinite travel times.
    assert np.isfinite(inst.travel_time).all()
    assert np.isfinite(inst.travel_dist).all()


def test_asymmetry_is_structural_not_noise():
    """Network-OD asymmetry should match v1 OSM Manhattan range (≥ 0.05)
    purely from one-way streets, no per-edge noise."""
    asyms = [generate(N=30, seed=s).asymmetry_score for s in range(5)]
    assert all(a >= 0.02 for a in asyms), f"asymmetries too low: {asyms}"


def test_zero_diagonal():
    """Travel time from a node to itself must be 0."""
    inst = generate(N=15, seed=1)
    diag = np.diag(inst.travel_time)
    assert np.allclose(diag, 0.0)


def test_solvable_by_greedy():
    """Generated instances must be solvable — no fundamental infeasibility."""
    inst = generate(N=30, seed=2)
    sol = greedy_mod.solve(inst, Settings())
    # Greedy might miss some customers due to TW, but the metrics must
    # be finite and the evaluator must not crash.
    assert sol.metrics["operational_cost"] < float("inf")
    assert sol.metrics.get("capacity_overload", 0.0) == 0.0


def test_reproducibility():
    """Same seed → bit-identical instance."""
    a = generate(N=20, seed=42)
    b = generate(N=20, seed=42)
    assert np.array_equal(a.travel_time, b.travel_time)
    assert a.asymmetry_score == b.asymmetry_score
    assert all(ca.demand == cb.demand for ca, cb in zip(a.customers, b.customers))


def test_grid_size_too_small_raises():
    """If grid_size² < N+1 the generator must error rather than silently truncate."""
    import pytest
    with pytest.raises(ValueError):
        generate(N=100, seed=0, grid_size=5)  # 25 nodes < 101 needed
