"""Unit tests for the iter-6a-2-refresh conditional operator registration.

The bug it fixes: meta-recipe operators (shift_start / class_shift /
depot_shift) were always registered in the bandit's arm pool, no-opping
correctly when their target constraint axis was inactive but still consuming
bandit-iteration budget. Tested by drift-rebenching embargo:
6/6 +$1797 -> 5/6 +$798 (56% magnitude drop) -> conditional fix -> 6/6 +$1648.

These tests guard the helper that does the pruning so any future regression
on the axis-detection predicates is caught fast.
"""
from __future__ import annotations

import pytest

from svrptw.config import Settings
from svrptw.io import Depot, Instance
from svrptw.instances_gen.synthetic import generate
from svrptw.solvers.classical.portfolio import _OPS, _filter_ops_pool


META_OPS = ("shift_start", "class_shift", "depot_shift")


@pytest.fixture
def inst_single_depot():
    return generate(N=20, seed=0)


@pytest.fixture
def inst_multi_depot():
    base = generate(N=20, seed=0)
    base.depots = [
        Depot(node_id=0, x=base.depot.x, y=base.depot.y,
              ready=base.depot.ready, due=base.depot.due),
        Depot(node_id=1, x=base.depot.x + 1.0, y=base.depot.y + 1.0,
              ready=base.depot.ready, due=base.depot.due),
    ]
    return base


def _pool():
    return dict(_OPS)


def test_baseline_settings_strips_all_three_meta_ops(inst_single_depot):
    """Default Settings + single-depot instance must drop all 3 meta ops."""
    s = Settings()
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    for op in META_OPS:
        assert op not in pool, f"{op} should be pruned under baseline settings"


def test_embargo_keeps_shift_start(inst_single_depot):
    s = Settings()
    s.economics.embargo_window_starts = (480, 720)
    s.economics.embargo_window_ends = (510, 750)
    s.economics.embargo_violation_penalty_per_visit = 50.0
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    assert "shift_start" in pool
    assert "class_shift" not in pool
    assert "depot_shift" not in pool


def test_peak_hour_keeps_shift_start(inst_single_depot):
    s = Settings()
    s.economics.peak_window_starts = (480, 1020)
    s.economics.peak_window_ends = (600, 1140)
    s.economics.peak_hour_wage_multiplier = 1.5
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    assert "shift_start" in pool


def test_peak_multiplier_without_windows_drops_shift_start(inst_single_depot):
    """Half-config (multiplier set but no windows) must NOT activate."""
    s = Settings()
    s.economics.peak_hour_wage_multiplier = 1.5
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    assert "shift_start" not in pool


def test_mixed_fleets_keeps_class_shift(inst_single_depot):
    s = Settings()
    s.economics.vehicle_class_capacities = (50.0, 100.0)
    s.economics.vehicle_class_fixed_premiums = (0.0, 20.0)
    s.economics.vehicle_class_per_mile_premiums = (0.0, 0.0)
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    assert "class_shift" in pool


def test_mixed_fleets_all_zero_premiums_drops_class_shift(inst_single_depot):
    """Class capacities set but ALL premiums zero -> no meaningful class axis."""
    s = Settings()
    s.economics.vehicle_class_capacities = (50.0, 100.0)
    s.economics.vehicle_class_fixed_premiums = (0.0, 0.0)
    s.economics.vehicle_class_per_mile_premiums = (0.0, 0.0)
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    assert "class_shift" not in pool


def test_skills_alone_keeps_class_shift(inst_single_depot):
    """Skills penalty alone (without mixed_fleets premiums) is enough."""
    s = Settings()
    s.economics.skill_mismatch_penalty_per_visit = 100.0
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    assert "class_shift" in pool


def test_multi_depot_keeps_depot_shift(inst_multi_depot):
    s = Settings()
    pool = _filter_ops_pool(_pool(), s, inst_multi_depot)
    assert "depot_shift" in pool


def test_single_depot_drops_depot_shift(inst_single_depot):
    s = Settings()
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    assert "depot_shift" not in pool


def test_one_depot_list_still_drops_depot_shift(inst_single_depot):
    """Edge case: depots set but len==1 -- still effectively single-depot."""
    inst_single_depot.depots = [inst_single_depot.depot]
    s = Settings()
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    assert "depot_shift" not in pool


def test_full_stack_keeps_shift_and_class_drops_depot(inst_single_depot):
    """The combined-stack scenario: all single-depot terms active."""
    s = Settings()
    s.economics.embargo_window_starts = (480, 720)
    s.economics.embargo_window_ends = (510, 750)
    s.economics.embargo_violation_penalty_per_visit = 50.0
    s.economics.vehicle_class_capacities = (50.0, 100.0)
    s.economics.vehicle_class_fixed_premiums = (5.0, 20.0)
    s.economics.skill_mismatch_penalty_per_visit = 100.0
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    assert "shift_start" in pool
    assert "class_shift" in pool
    assert "depot_shift" not in pool


def test_filter_preserves_general_ops(inst_single_depot):
    """Conditional registration must not touch the general-purpose operators."""
    s = Settings()
    pool = _filter_ops_pool(_pool(), s, inst_single_depot)
    for op in (
        "merge_routes", "relocate", "two_opt_intra", "two_opt_star",
        "swap_star", "three_opt_intra", "sisr", "ejection_chain",
        "cyclic_3", "vehicle_kill", "soft_drop", "drop_route",
        "destroy_island", "drop_leg",
    ):
        assert op in pool, f"general op {op} unexpectedly pruned"
