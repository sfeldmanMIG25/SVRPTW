"""SPEC-6-LOGIC-01 LogicEnsemble tests."""
from __future__ import annotations

from pathlib import Path

import torch

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.logic.ensemble import EnsembleScore, LogicEnsemble
from svrptw.logic.features import FEATURE_DIM
from svrptw.logic.student import LogicStudent
from svrptw.solvers.classical import greedy as greedy_mod


def _make_heads(n: int = 5) -> list[LogicStudent]:
    return [LogicStudent() for _ in range(n)]


def test_ensemble_score_shape_and_range():
    ens = LogicEnsemble(_make_heads(5))
    inst = generate(N=15, seed=0)
    sol = greedy_mod.solve(inst, Settings())
    s = ens.score(inst, sol)
    assert isinstance(s, EnsembleScore)
    assert 0.0 <= s.mean <= 1.0
    assert s.std >= 0.0


def test_ensemble_single_head_zero_std():
    """Single-head ensemble must produce std=0 and authoritative=True."""
    ens = LogicEnsemble([LogicStudent()], authoritative_max_std=0.08)
    inst = generate(N=10, seed=1)
    sol = greedy_mod.solve(inst, Settings())
    s = ens.score(inst, sol)
    assert s.std == 0.0
    assert s.authoritative is True


def test_ensemble_authoritative_flips_on_threshold():
    """High σ across heads → authoritative=False."""
    # Construct 5 heads with deliberately different temperature so that
    # their sigmoid(logit/T) outputs disagree.
    heads = _make_heads(5)
    # Initialise heads with different random weights (already does by default).
    # Force temperatures to differ wildly to amplify disagreement.
    for i, h in enumerate(heads):
        with torch.no_grad():
            h.log_T.copy_(torch.tensor(float(i - 2)))  # T in {0.14, 0.37, 1.0, 2.7, 7.4}
    ens = LogicEnsemble(heads, authoritative_max_std=0.001)
    inst = generate(N=15, seed=2)
    sol = greedy_mod.solve(inst, Settings())
    s = ens.score(inst, sol)
    assert s.authoritative is False, f"expected non-authoritative, got std={s.std}"


def test_ensemble_save_load_roundtrip(tmp_path: Path):
    heads = _make_heads(3)
    ens = LogicEnsemble(heads, authoritative_max_std=0.12)
    inst = generate(N=12, seed=3)
    sol = greedy_mod.solve(inst, Settings())
    before = ens.score(inst, sol)

    ens.save(tmp_path / "ens")
    loaded = LogicEnsemble.load(tmp_path / "ens")
    assert loaded.n_heads == 3
    after = loaded.score(inst, sol)
    # Same heads + same input → same score.
    assert abs(before.mean - after.mean) < 1e-6
    assert abs(before.std - after.std) < 1e-6


def test_ensemble_empty_heads_raises():
    import pytest
    with pytest.raises(ValueError):
        LogicEnsemble([])
