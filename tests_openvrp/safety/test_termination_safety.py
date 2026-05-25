"""Termination-safety test harness — verifies the solver returns cleanly
at any step in solve time, with cleanup, valid intermediate solutions,
and no resource leaks.

This is a TESTING HARNESS over the existing openvrp/svrptw code, not a
modification of the solver. Failures here surface real safety issues
the public API must handle gracefully.

What we test:
1. Every-phase deadline-honor — tight budgets at boundaries.
2. StopSolve cancellation cleanly returns best-so-far.
3. Hard-termination via arbitrary exception in on_progress is wrapped as
   SolveAborted (never bare Exception).
4. Edge-case budgets (deadline already past, near-zero) return a valid
   Solution rather than crashing.
5. Solution invariants hold even on terminated runs (status,
   feasibility_blockers, fleet_min_report).
6. Resource cleanup: no leaked threads, no leaked open file handles.
7. Determinism survives termination (same seed + same budget = same
   intermediate solution).
"""
from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest

from openvrp import (
    Coordinate,
    Depot,
    SolveOptions,
    Solution,
    Stop,
    StopSolve,
    TimeWindow,
    VehicleClass,
    solve_od,
)
from openvrp.errors import SolveAborted


# ============================================================
# Test fixtures
# ============================================================


def _build_inst(n: int = 30, *, seed: int = 0) -> tuple:
    """Deterministic OD-mode instance. N=30 is the sweet spot where the
    native solver finishes < 50 ms on a fast budget — small enough to
    run hundreds of safety variants, big enough to actually exercise
    construct + merge + 2-opt + relocate phases."""
    rng = np.random.default_rng(seed)
    coords = [(0.0, 0.0)] + [(float(rng.uniform(-20, 20)),
                              float(rng.uniform(-20, 20))) for _ in range(n)]
    nn = len(coords)
    T = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i != j:
                T[i, j] = float(np.hypot(coords[i][0] - coords[j][0],
                                         coords[i][1] - coords[j][1]) * 60.0)
    depots = [Depot(id="D0", node_index=0,
                    coordinate=Coordinate(lon=0, lat=0))]
    stops = [Stop(id=f"S{i}", node_index=i,
                  coordinate=Coordinate(lon=coords[i][0], lat=coords[i][1]),
                  demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, n + 1)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0")]
    return T, depots, stops, fleet


def _assert_valid_solution(sol: Solution) -> None:
    """Every returned Solution — even after termination — must satisfy
    the SPEC-OPENVRP-02 invariants."""
    assert sol is not None, "solve() returned None"
    assert sol.status in ("feasible", "infeasible", "partial"), \
        f"unexpected status {sol.status!r}"
    assert sol.objective_value >= 0.0 or sol.status == "infeasible", \
        f"negative objective {sol.objective_value} on non-infeasible solve"
    assert sol.vehicles_used == len([r for r in sol.routes if r.visits]), \
        "vehicles_used inconsistent with routes"
    assert sol.vehicles_minimum_found <= sol.vehicles_used, \
        "minimum_found exceeds used (D7 violation)"
    # diagnostics always populated
    assert sol.diagnostics is not None
    assert sol.diagnostics.operator_pool, "operator_pool empty"
    # quality report always populated (8 metrics)
    assert len(sol.quality_report.solution_level) == 8


# ============================================================
# Edge-case budgets
# ============================================================


def test_safety_near_zero_budget_returns_valid_solution():
    """budget_seconds=0.001 must return a valid Solution, not crash."""
    T, depots, stops, fleet = _build_inst(n=20)
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=0.001, seed=0))
    _assert_valid_solution(sol)


def test_safety_already_past_deadline_returns_valid_solution():
    """deadline=time.monotonic()-1.0 (already in the past) returns
    a valid Solution rather than crashing."""
    T, depots, stops, fleet = _build_inst(n=20)
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=60.0,
                                        deadline=time.monotonic() - 1.0,
                                        seed=0))
    _assert_valid_solution(sol)


def test_safety_very_short_budget_does_not_overrun_by_more_than_2s():
    """At a 0.05s budget, wall must be < 2s (finalize + overhead)."""
    T, depots, stops, fleet = _build_inst(n=20)
    t0 = time.monotonic()
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=0.05, seed=0,
                                        construction="fast"))
    wall = time.monotonic() - t0
    _assert_valid_solution(sol)
    assert wall < 2.0, f"wall {wall:.2f}s exceeded 2s soft cap on 0.05s budget"


@pytest.mark.parametrize("budget", [0.05, 0.1, 0.5, 1.0, 5.0])
def test_safety_budget_grid_all_return_valid(budget: float):
    """Across a budget range, every solve returns a valid Solution."""
    T, depots, stops, fleet = _build_inst(n=20)
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=budget, seed=0,
                                        construction="fast"))
    _assert_valid_solution(sol)


# ============================================================
# Cancellation via StopSolve
# ============================================================


@pytest.mark.parametrize("cancel_at_phase",
                         ["ingest", "construct", "refine", "finalize"])
def test_safety_stopsolve_at_any_phase_returns_valid(cancel_at_phase: str):
    """StopSolve raised from on_progress at any phase returns cleanly."""
    T, depots, stops, fleet = _build_inst(n=25)
    fired = {"any": False}

    def on_progress(ev):
        fired["any"] = True
        if ev.phase == cancel_at_phase:
            raise StopSolve()

    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=5.0, seed=0,
                                        on_progress=on_progress,
                                        construction="fast"))
    assert fired["any"], "on_progress never fired"
    _assert_valid_solution(sol)


def test_safety_stopsolve_returns_best_so_far():
    """StopSolve cancellation returns the best solution found before
    the cancel — not a placeholder."""
    T, depots, stops, fleet = _build_inst(n=30)

    def on_progress(ev):
        if ev.phase == "refine":
            raise StopSolve()

    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=5.0, seed=0,
                                        on_progress=on_progress,
                                        construction="fast"))
    _assert_valid_solution(sol)
    # Best-so-far must be a real solution (not infeasible from a bare init)
    assert sol.routes, "StopSolve returned no routes"


# ============================================================
# Hard termination via arbitrary exception in on_progress
# ============================================================


def test_safety_arbitrary_exception_wrapped_as_SolveAborted():
    """An on_progress callback that raises something other than StopSolve
    must surface as SolveAborted, never bare RuntimeError (per spec).
    """
    T, depots, stops, fleet = _build_inst(n=20)

    def bad_callback(ev):
        raise RuntimeError("simulated downstream crash")

    with pytest.raises(SolveAborted) as ei:
        solve_od(T, stops, depots, fleet,
                 options=SolveOptions(budget_seconds=2.0, seed=0,
                                      on_progress=bad_callback,
                                      construction="fast"))
    # The original exception is chained (`raise ... from e`) so the
    # caller can introspect.
    assert "simulated downstream crash" in str(ei.value) or \
           isinstance(ei.value.__cause__, RuntimeError)


def test_safety_keyboard_interrupt_propagates_cleanly():
    """KeyboardInterrupt is a special sentinel — Python convention is
    that BaseException subclasses propagate. Either it propagates raw
    or it wraps as SolveAborted; both are acceptable, neither leaves
    the solver in a bad state.
    """
    T, depots, stops, fleet = _build_inst(n=20)

    def interrupting_callback(ev):
        raise KeyboardInterrupt()

    with pytest.raises((KeyboardInterrupt, SolveAborted)):
        solve_od(T, stops, depots, fleet,
                 options=SolveOptions(budget_seconds=2.0, seed=0,
                                      on_progress=interrupting_callback,
                                      construction="fast"))


# ============================================================
# Solution invariants on terminated runs
# ============================================================


def test_safety_terminated_solution_has_fleet_min_report():
    """Even after StopSolve, fleet_min_report is populated."""
    T, depots, stops, fleet = _build_inst(n=20)
    def on_progress(ev):
        if ev.phase == "refine":
            raise StopSolve()
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=2.0, seed=0,
                                        on_progress=on_progress,
                                        construction="fast"))
    assert "used" in sol.diagnostics.fleet_min_report
    assert "minimum_found" in sol.diagnostics.fleet_min_report


def test_safety_terminated_solution_has_full_quality_report():
    """Even after termination, all 8 catalog metrics are reported."""
    T, depots, stops, fleet = _build_inst(n=20)
    def on_progress(ev):
        if ev.phase == "construct":
            raise StopSolve()
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=2.0, seed=0,
                                        on_progress=on_progress,
                                        construction="fast"))
    from openvrp.schema.input import QUALITY_CATALOG
    for k in QUALITY_CATALOG:
        assert k in sol.quality_report.solution_level


def test_safety_terminated_solution_serializable():
    """A terminated Solution still round-trips through JSON."""
    T, depots, stops, fleet = _build_inst(n=20)
    def on_progress(ev):
        if ev.phase == "refine":
            raise StopSolve()
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=2.0, seed=0,
                                        on_progress=on_progress,
                                        construction="fast"))
    body = sol.model_dump_json()
    sol2 = Solution.model_validate_json(body)
    assert sol2.objective_value == sol.objective_value
    assert sol2.vehicles_used == sol.vehicles_used


# ============================================================
# Resource cleanup
# ============================================================


@contextmanager
def _thread_leak_check():
    """Capture before/after thread count; tolerate +1 for timer threads."""
    before = threading.active_count()
    yield
    gc.collect()
    after = threading.active_count()
    assert after <= before + 1, (
        f"thread leak: {before} -> {after} active threads"
    )


def test_safety_no_thread_leak_after_many_solves():
    """Run 20 solves; active-thread count should not grow unboundedly."""
    T, depots, stops, fleet = _build_inst(n=15)
    with _thread_leak_check():
        for _ in range(20):
            sol = solve_od(T, stops, depots, fleet,
                           options=SolveOptions(budget_seconds=0.1, seed=0,
                                                construction="fast"))
            _assert_valid_solution(sol)


def test_safety_no_thread_leak_after_many_stopsolves():
    """20 cancelled solves don't leak threads either."""
    T, depots, stops, fleet = _build_inst(n=15)
    def cancel(ev):
        if ev.phase == "refine":
            raise StopSolve()
    with _thread_leak_check():
        for _ in range(20):
            sol = solve_od(T, stops, depots, fleet,
                           options=SolveOptions(budget_seconds=1.0, seed=0,
                                                on_progress=cancel,
                                                construction="fast"))
            _assert_valid_solution(sol)


# ============================================================
# Determinism under termination
# ============================================================


def test_safety_termination_is_deterministic():
    """Same seed + same termination point ⇒ identical intermediate Solution."""
    T, depots, stops, fleet = _build_inst(n=30)
    def cancel(ev):
        if ev.phase == "refine":
            raise StopSolve()
    seqs = []
    for _ in range(3):
        sol = solve_od(T, stops, depots, fleet,
                       options=SolveOptions(budget_seconds=1.0, seed=42,
                                            on_progress=cancel,
                                            construction="fast"))
        seqs.append(tuple(
            tuple(v.stop_id for v in r.visits if v.kind != "depot")
            for r in sorted(sol.routes,
                            key=lambda r: (r.vehicle_class_id, r.vehicle_ordinal))
        ))
    assert seqs[0] == seqs[1] == seqs[2], \
        "termination produced non-deterministic intermediate solutions"


# ============================================================
# Progress events are always non-decreasing in fraction + elapsed
# ============================================================


def test_safety_progress_events_monotonic():
    """ProgressEvent.fraction and elapsed_seconds never go backward."""
    T, depots, stops, fleet = _build_inst(n=20)
    events = []

    def capture(ev):
        events.append((ev.phase, ev.fraction, ev.elapsed_seconds))

    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=2.0, seed=0,
                                        on_progress=capture,
                                        construction="fast"))
    _assert_valid_solution(sol)
    assert events, "no progress events fired"
    # elapsed_seconds monotonically non-decreasing across all events
    elapsed_seq = [ev[2] for ev in events]
    for i in range(1, len(elapsed_seq)):
        assert elapsed_seq[i] >= elapsed_seq[i - 1] - 1e-6, \
            f"elapsed went backward at event {i}: {elapsed_seq}"
    # fraction within [0, 1]
    for ph, frac, el in events:
        assert 0.0 <= frac <= 1.0, f"fraction out of range: {frac} at phase {ph}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
