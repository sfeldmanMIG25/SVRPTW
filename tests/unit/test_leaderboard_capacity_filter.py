"""SPEC-0-EVAL-01 — leaderboard.render() must filter overloaded rows."""
from __future__ import annotations

from svrptw.bench.leaderboard import render


def _row(solver, cost, overload=0.0, n=50, inst="I0"):
    return {
        "solver": solver, "operational_cost": cost,
        "capacity_overload": overload,
        "n": n, "instance_id": f"{inst}-{solver}",
        "missed_deliveries": 0, "wall_clock_seconds": 1.0,
        "num_vehicles_used": 1,
    }


def test_overloaded_rows_dropped_from_leaderboard():
    rows = [
        _row("portfolio", 800.0, overload=0.0),
        _row("lkh3",      300.0, overload=50.0),   # "cheats" — should be dropped
    ]
    md = render(rows)
    # The cheating cost line must not appear; portfolio's cost must.
    assert "800.00" in md
    assert "300.00" not in md
    # The drop count must be flagged in the header.
    assert "dropped **1**" in md


def test_filter_off_when_explicit():
    rows = [
        _row("portfolio", 800.0, overload=0.0),
        _row("lkh3",      300.0, overload=50.0),
    ]
    md = render(rows, filter_capacity_overload=False)
    assert "800.00" in md
    assert "300.00" in md


def test_missing_capacity_overload_field_is_safe():
    """Old bench rows pre-SPEC-0-EVAL-01 don't have the field at all."""
    rows = [
        {"solver": "old", "operational_cost": 500.0, "n": 50,
         "instance_id": "I0", "missed_deliveries": 0,
         "wall_clock_seconds": 1.0, "num_vehicles_used": 1},
    ]
    md = render(rows)  # must not crash on missing field
    assert "500.00" in md
