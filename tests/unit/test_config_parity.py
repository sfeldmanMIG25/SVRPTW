"""Verify the typed config reproduces config.py constants byte-for-byte.
Part of SPEC-0-CFG-01 acceptance."""
import sys
from pathlib import Path

# Make the legacy flat module importable from tests.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config as legacy  # noqa: E402
from svrptw.config import Settings  # noqa: E402


def test_defaults_match_legacy():
    s = Settings.from_yaml(ROOT / "svrptw" / "config" / "defaults.yaml")
    assert s.economics.wage_per_hour == legacy.WAGE_COST_PER_MINUTE * 60.0
    assert s.economics.cost_per_mile == legacy.TRANSIT_COST_PER_MILE
    assert s.economics.hard_late_penalty == legacy.HARD_LATE_PENALTY
    assert s.time.day_start_minute == legacy.DEPOT_E_TIME
    assert s.time.day_end_minute == legacy.DEPOT_L_TIME
    assert s.stochastic.travel_lognormal_sigma == legacy.TRAVEL_TIME_LN_SIGMA
    assert s.stochastic.service_normal_sigma == legacy.SERVICE_TIME_SIGMA
    assert s.stochastic.service_mean == legacy.SERVICE_TIME_BASE_MEAN


def test_strict_unknown_key_rejected():
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        Settings(unknown_field=42)


def test_stochastic_disabled_by_default():
    s = Settings.from_yaml(ROOT / "svrptw" / "config" / "defaults.yaml")
    assert s.stochastic.enabled is False
