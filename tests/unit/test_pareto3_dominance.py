"""SPEC-6-PARETO-3AXIS-01 unit tests."""
from __future__ import annotations

from svrptw.bench.pareto3 import _better, dominance_report


def _row(instance, solver, cost, t, n=50, logic=None, auth=True, overload=0.0):
    r = {
        "instance_id": instance, "solver": solver, "n": n,
        "operational_cost": cost, "wall_clock_seconds": t,
        "capacity_overload": overload,
    }
    if logic is not None:
        r["logic_score"] = logic
        r["logic_authoritative"] = auth
    return r


def test_better_tolerance_boundary():
    # 1 % relative tolerance — equal-to-tolerance is NOT strictly better.
    assert _better(99.0, 100.0, minimise=True) is False  # within tol
    assert _better(98.9, 100.0, minimise=True) is True   # outside tol
    assert _better(101.0, 100.0, minimise=False) is False
    assert _better(101.1, 100.0, minimise=False) is True


def test_clean_3axis_dominance():
    rows = [
        _row("I1", "challenger", cost=100.0, t=1.0, logic=0.8),
        _row("I1", "baseline",   cost=110.0, t=10.0, logic=0.7),
    ]
    r = dominance_report(rows, challenger="challenger", baseline="baseline")
    assert r.n_total == 1
    assert r.n_dominated == 1
    assert r.headline == 1.0


def test_tie_when_axes_split():
    """Challenger wins cost+time, baseline wins logic → no dominance."""
    rows = [
        _row("I1", "challenger", cost=100.0, t=1.0, logic=0.5),
        _row("I1", "baseline",   cost=110.0, t=10.0, logic=0.9),
    ]
    r = dominance_report(rows, challenger="challenger", baseline="baseline")
    assert r.n_total == 1
    assert r.n_dominated == 0
    assert r.n_ties == 1


def test_logic_axis_drops_when_not_authoritative():
    """Either side non-authoritative → logic drops, 2-axis dominance applies."""
    rows = [
        _row("I1", "challenger", cost=100.0, t=1.0, logic=0.5, auth=True),
        _row("I1", "baseline",   cost=110.0, t=10.0, logic=0.9, auth=False),
    ]
    r = dominance_report(rows, challenger="challenger", baseline="baseline")
    assert r.logic_dropped == 1
    # Dropping logic → challenger wins on (cost, time).
    assert r.n_dominated == 1


def test_capacity_overload_filter():
    """A row with capacity_overload > 0 is filtered out before comparison."""
    rows = [
        _row("I1", "challenger", cost=100.0, t=1.0, overload=0.0),
        _row("I1", "baseline",   cost=10.0,  t=1.0, overload=5.0),  # cheating
    ]
    r = dominance_report(rows, challenger="challenger", baseline="baseline",
                         axes=("operational_cost", "wall_clock_seconds"),
                         minimise=(True, True))
    # Baseline is filtered → no comparable pair → n_total == 0.
    assert r.n_total == 0


def test_two_axis_fallback_when_logic_missing():
    rows = [
        _row("I1", "challenger", cost=100.0, t=1.0),  # no logic
        _row("I1", "baseline",   cost=110.0, t=10.0),
    ]
    r = dominance_report(rows, challenger="challenger", baseline="baseline")
    assert r.logic_dropped == 1
    assert r.n_dominated == 1


def test_per_n_breakdown():
    rows = [
        _row("A", "challenger", cost=100.0, t=1.0, n=50),
        _row("A", "baseline",   cost=110.0, t=10.0, n=50),
        _row("B", "challenger", cost=200.0, t=1.0, n=100),
        _row("B", "baseline",   cost=190.0, t=10.0, n=100),  # cost ties → no dom
    ]
    r = dominance_report(rows, challenger="challenger", baseline="baseline",
                         axes=("operational_cost", "wall_clock_seconds"),
                         minimise=(True, True))
    assert r.per_n[50]["dom"] == 1
    assert r.per_n[50]["n"] == 1
    assert r.per_n[100]["dom"] == 0


def test_bit_stable_headline():
    rows = [
        _row(f"I{i}", "challenger", cost=100.0 - i, t=1.0)
        for i in range(5)
    ] + [
        _row(f"I{i}", "baseline", cost=110.0 - i, t=10.0)
        for i in range(5)
    ]
    r1 = dominance_report(rows, challenger="challenger", baseline="baseline",
                          axes=("operational_cost", "wall_clock_seconds"),
                          minimise=(True, True))
    r2 = dominance_report(rows, challenger="challenger", baseline="baseline",
                          axes=("operational_cost", "wall_clock_seconds"),
                          minimise=(True, True))
    assert r1.headline == r2.headline
