"""Smoke test for the bulk-generated v1_large network instances.

Confirms a generated N=500 OSM instance round-trips through
load_instance with the right shape, no NaN/inf, depot self-distance
of zero, and a plausible first-leg distance/time ratio (5..80 mph).

Skips automatically if the instance is not present (so this test is
no-op on a fresh checkout that hasn't run build_large_batch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from svrptw.io import load_instance

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTANCE = REPO_ROOT / "instances" / "v1_large" / "OSM-Manhattan-N0500-I000.json"


pytestmark = pytest.mark.skipif(
    not INSTANCE.exists(),
    reason="bulk instance not generated; run bench/scripts/build_large_batch.py",
)


def test_shape_and_finite():
    inst = load_instance(INSTANCE)
    n_plus_1 = inst.num_customers + 1
    assert inst.travel_dist.shape == (n_plus_1, n_plus_1)
    assert inst.travel_time.shape == (n_plus_1, n_plus_1)
    assert not np.isnan(inst.travel_dist).any(), "dist has NaN"
    assert not np.isinf(inst.travel_dist).any(), "dist has inf"
    assert not np.isnan(inst.travel_time).any(), "time has NaN"
    assert not np.isinf(inst.travel_time).any(), "time has inf"


def test_depot_self_zero():
    inst = load_instance(INSTANCE)
    assert inst.travel_dist[0, 0] == 0.0
    assert inst.travel_time[0, 0] == 0.0


def test_first_leg_consistency():
    """travel_dist[0,1] > 0, time > 0, ratio in 5..80 mph plausible band."""
    inst = load_instance(INSTANCE)
    d = float(inst.travel_dist[0, 1])
    t_min = float(inst.travel_time[0, 1])
    assert d > 0.0, f"depot->cust1 distance not positive: {d}"
    assert t_min > 0.0, f"depot->cust1 time not positive: {t_min}"
    # mph = miles / (minutes / 60)
    mph = d / (t_min / 60.0)
    assert 5.0 <= mph <= 80.0, f"implausible avg speed {mph:.2f} mph"
