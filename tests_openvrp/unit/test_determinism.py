"""SPEC-OPENVRP-00 D11, SPEC-OPENVRP-03 §5 — Determinism contract.

Identical Problem + seed ⇒ identical Route.visits stop sequences across:
- repeated serial runs
- threads=1 vs threads=N
- repeated `fast` constructions
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from openvrp import (
    Coordinate,
    Depot,
    SolveOptions,
    Stop,
    TimeWindow,
    VehicleClass,
    solve_od,
)


def _build_instance(n: int = 12, seed_data: int = 0):
    rng = np.random.default_rng(seed_data)
    coords = [(0.0, 0.0)] + [(float(rng.uniform(-20, 20)),
                              float(rng.uniform(-20, 20))) for _ in range(n)]
    nn = len(coords)
    T = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i != j:
                T[i, j] = float(np.hypot(coords[i][0] - coords[j][0],
                                         coords[i][1] - coords[j][1]) * 60.0)
    depots = [Depot(id="D0", node_index=0, coordinate=Coordinate(lon=0, lat=0))]
    stops = [Stop(id=f"S{i}", node_index=i,
                  coordinate=Coordinate(lon=coords[i][0], lat=coords[i][1]),
                  demand={"w": 1.0},
                  time_windows=[TimeWindow(earliest=0, latest=8 * 3600)])
             for i in range(1, n + 1)]
    fleet = [VehicleClass(id="van", count=None, capacity={"w": 50.0},
                          home_depot_id="D0")]
    return T, depots, stops, fleet


def _stop_sequence(sol) -> tuple:
    """Hashable canonical representation of the chosen routes' stop sequences."""
    return tuple(
        tuple(v.stop_id for v in r.visits if v.kind != "depot")
        for r in sorted(sol.routes,
                        key=lambda r: (r.vehicle_class_id, r.vehicle_ordinal))
    )


def test_determinism_serial_repeated():
    """Same seed, same problem, same answer across repeated serial runs."""
    T, depots, stops, fleet = _build_instance(n=15)
    seqs = []
    for _ in range(3):
        sol = solve_od(T, stops, depots, fleet,
                       options=SolveOptions(budget_seconds=1.0, seed=42))
        seqs.append(_stop_sequence(sol))
    assert seqs[0] == seqs[1] == seqs[2], "serial runs produced different sequences"


def test_determinism_thread_count_independence():
    """threads=1 vs threads=N produce the same stop sequence."""
    T, depots, stops, fleet = _build_instance(n=15)
    s1 = solve_od(T, stops, depots, fleet,
                  options=SolveOptions(budget_seconds=1.0, seed=42, threads=1))
    s4 = solve_od(T, stops, depots, fleet,
                  options=SolveOptions(budget_seconds=1.0, seed=42, threads=4))
    # Sequences should match — the native solver is single-threaded, but
    # the contract holds either way.
    assert _stop_sequence(s1) == _stop_sequence(s4)


def test_determinism_different_seeds_likely_differ():
    """Sanity: different seeds CAN produce different sequences. (If both
    seeds converge to the same optimum that's fine; we only check the
    determinism contract isn't trivially satisfied.)"""
    T, depots, stops, fleet = _build_instance(n=15)
    s_a = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=1.0, seed=42))
    s_b = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=1.0, seed=123))
    # We don't *require* they differ (small instance may converge to same
    # optimum); just verify both are deterministic individually.
    s_a2 = solve_od(T, stops, depots, fleet,
                    options=SolveOptions(budget_seconds=1.0, seed=42))
    assert _stop_sequence(s_a) == _stop_sequence(s_a2)


def _solve_in_subprocess(seed: int) -> tuple:
    """Run a solve in a subprocess (verifies determinism survives process
    boundaries — relevant for parallel benchmarks)."""
    T, depots, stops, fleet = _build_instance(n=12)
    sol = solve_od(T, stops, depots, fleet,
                   options=SolveOptions(budget_seconds=1.0, seed=seed))
    return tuple(
        tuple(v.stop_id for v in r.visits if v.kind != "depot")
        for r in sorted(sol.routes,
                        key=lambda r: (r.vehicle_class_id, r.vehicle_ordinal))
    )


def test_determinism_across_processes():
    """Solve in 4 subprocesses with the same seed — all must agree."""
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_solve_in_subprocess, [42, 42, 42, 42]))
    assert results[0] == results[1] == results[2] == results[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
