"""Smoke tests for ``svrptw.scoring.sub_api``.

Covers:
  - score(inst, sol) with no visual returns a finite UnifiedScore
  - weight isolation: weights={quality_index: 1.0, operational_cost: 0.0}
    yields unified == quality_index
  - compare(inst, sol_a, sol_a) returns winner='tie' and margin~0
  - visual_score=None branch never triggers a VLM call
"""
from __future__ import annotations

import math

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.scoring.sub_api import (
    DEFAULT_WEIGHTS,
    UnifiedScore,
    compare,
    score,
)
from svrptw.solvers.classical import greedy


def _greedy_sol(N: int = 20, seed: int = 13):
    inst = generate(N=N, seed=seed)
    sol = greedy.solve(inst, Settings())
    return inst, sol


def test_score_no_visual_returns_finite():
    inst, sol = _greedy_sol()
    s = score(inst, sol)
    assert isinstance(s, UnifiedScore)
    assert math.isfinite(s.unified)
    assert 0.0 <= s.unified <= 1.0
    assert math.isfinite(s.operational_cost)
    assert 0.0 <= s.quality_index <= 1.0
    # Explicitly: no VLM call took place because visual_image_path is None.
    assert s.visual_score is None
    # Breakdown must include both active components and exclude visual.
    assert "operational_cost" in s.breakdown
    assert "quality_index" in s.breakdown
    assert "visual_score" not in s.breakdown


def test_weight_isolation_quality_only():
    """Setting quality_index weight=1 and others=0 must yield
    unified == quality_index (within float tolerance)."""
    inst, sol = _greedy_sol()
    s = score(inst, sol,
              weights={"quality_index": 1.0,
                       "operational_cost": 0.0,
                       "visual_score": 0.0})
    assert math.isclose(s.unified, s.quality_index, abs_tol=1e-9), (
        f"unified={s.unified} != quality_index={s.quality_index}"
    )


def test_compare_self_is_tie():
    inst, sol = _greedy_sol(N=15, seed=2)
    res = compare(inst, sol, sol)
    assert res["winner"] == "tie", f"self-vs-self winner={res['winner']!r}"
    assert res["margin"] < 1e-6
    # No dissent flags can fire when both sides are byte-identical.
    assert res["dissent_flags"] == []
    # No pair visual was requested.
    assert res["pair_visual_score"] is None


def test_visual_none_no_crash():
    """visual_image_path=None must skip the VLM entirely (no network
    call, no exception). Default weights must renormalise so the
    remaining weights sum to 1."""
    inst, sol = _greedy_sol(N=15, seed=8)
    s = score(inst, sol, visual_image_path=None)
    assert s.visual_score is None
    used = s.raw["weights_used"]
    assert math.isclose(sum(used.values()), 1.0, abs_tol=1e-9), (
        f"renormalized weights {used} do not sum to 1"
    )
    # Default weights must include cost+quality and exclude visual.
    assert "operational_cost" in used
    assert "quality_index" in used
    assert "visual_score" not in used


def test_default_weights_constant_unchanged():
    """Guards against accidental in-place mutation of the module-level
    DEFAULT_WEIGHTS constant by ``score``/``compare`` paths."""
    expected = {"operational_cost": 0.40, "quality_index": 0.40,
                "visual_score": 0.20}
    assert DEFAULT_WEIGHTS == expected
