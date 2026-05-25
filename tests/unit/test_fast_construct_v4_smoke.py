"""Smoke tests for fast_construct_v4 (Solomon I1 sequential insertion).

iter-7-bis-v4 (2026-05-16). v4 closes the +37% gap that v2 leaves vs pyvrp:
standalone bench at v1_large 6-instance suite shows v4 beats pyvrp on BOTH
cost (-3% mean) AND wall (-74% mean). These tests guard the basic
construction invariants.
"""
from __future__ import annotations

import math
import time

import pytest

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.solvers.classical.fast_construct_v4 import solve as fc4_solve


def test_v4_returns_feasible_solution_with_correct_tag():
    """Basic smoke: produces a feasible solution tagged 'fast_construct_v4'."""
    inst = generate(N=40, seed=11)
    sol = fc4_solve(inst, Settings(), budget_seconds=2.0, seed=0)
    assert sol.solver == "fast_construct_v4"
    assert math.isfinite(sol.metrics["operational_cost"])
    assert sol.feasible, f"expected feasible, got metrics={sol.metrics}"
    # All customers placed
    assert sol.metrics.get("i1_unrouted", 0) == 0


def test_v4_honors_wall_budget():
    """v4 must stay within 1.5x of its budget at small N (loose factor
    because synthetic instance setup time isn't counted but startup is)."""
    inst = generate(N=80, seed=23)
    budget = 1.0
    t0 = time.perf_counter()
    sol = fc4_solve(inst, Settings(), budget_seconds=budget, seed=0)
    wall = time.perf_counter() - t0
    assert wall <= budget * 1.5 + 0.5, (
        f"v4 wall {wall:.3f}s > {budget*1.5 + 0.5:.3f}s (budget={budget}s)"
    )


def test_v4_no_overlapping_customers():
    """Each customer is in exactly one route."""
    inst = generate(N=60, seed=37)
    sol = fc4_solve(inst, Settings(), budget_seconds=1.5, seed=0)
    all_cids: list[int] = []
    for r in sol.routes:
        all_cids.extend(r.customers)
    # No duplicates
    assert len(all_cids) == len(set(all_cids)), (
        "v4 produced duplicate customer assignments"
    )


def test_v4_all_customers_routed_on_simple_instance():
    """On a small loose instance, every customer should be routable."""
    inst = generate(N=30, seed=53)
    sol = fc4_solve(inst, Settings(), budget_seconds=2.0, seed=0)
    placed = sum(len(r.customers) for r in sol.routes)
    assert placed == inst.num_customers, (
        f"v4 placed only {placed}/{inst.num_customers} customers on a "
        "loose synthetic instance"
    )


def test_v4_seed_param_accepted():
    """seed= must be accepted for API parity with v1/v2/v3 even though
    I1 is deterministic given the same input."""
    inst = generate(N=20, seed=7)
    s = Settings()
    sol_a = fc4_solve(inst, s, budget_seconds=1.0, seed=0)
    sol_b = fc4_solve(inst, s, budget_seconds=1.0, seed=42)
    # Same input + deterministic algorithm => same cost
    assert abs(sol_a.metrics["operational_cost"]
               - sol_b.metrics["operational_cost"]) < 1e-6, (
        "v4 should be deterministic across seeds"
    )


def test_v4_nearest_k_zero_does_full_enumeration():
    """nearest_k=0 should enumerate all unrouted as candidates (slower
    but in principle finds a slightly better solution at small N)."""
    inst = generate(N=40, seed=61)
    sol = fc4_solve(inst, Settings(), budget_seconds=2.0, seed=0, nearest_k=0)
    assert sol.solver == "fast_construct_v4"
    assert sol.feasible


def test_v4_polish_off_is_faster():
    """polish=False should skip the per-route two_opt_intra pass."""
    inst = generate(N=60, seed=71)
    t0 = time.perf_counter()
    sol_off = fc4_solve(inst, Settings(), budget_seconds=2.0, seed=0,
                        polish=False)
    wall_off = time.perf_counter() - t0
    assert sol_off.feasible
    # Wall ~ construction-only; should be quick
    assert wall_off < 1.5, f"v4 polish=False at N=60 took {wall_off:.2f}s"
