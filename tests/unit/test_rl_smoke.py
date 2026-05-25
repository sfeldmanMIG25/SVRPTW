"""Phase D3-D5 RL smoke tests.

Covers:
  - LoggingBandit collects N transitions when called N times.
  - shaped_reward_terms returns 0 for identical solutions (no shaping).
  - shaped_reward_terms returns positive island_disposal when sol_after
    has fewer islands than sol_before.
  - total_shaped_reward({}) returns base_reward unchanged.
  - MLPBanditPolicy raises a helpful RuntimeError when torch is missing
    (skipped when torch IS available).
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from svrptw.config import Settings
from svrptw.instances_gen.synthetic import generate
from svrptw.solvers.common.solution import Route, Solution, evaluate
from svrptw.solvers.learning.logging_bandit import LoggingBandit
from svrptw.solvers.learning.reward_shaping import (
    DEFAULT_COEFS,
    shaped_reward_terms,
    total_shaped_reward,
)

OPS = ["a", "b", "c", "d"]
DIM = 16


def _ctx() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=DIM).astype(np.float64)


def test_logging_bandit_collects_n_transitions():
    bandit = LoggingBandit(ops=OPS, feature_dim=DIM, seed=1)
    n = 7
    for i in range(n):
        ctx = _ctx()
        op = bandit.choose(ctx, eps=0.5)
        bandit.update(op, ctx, reward=float(i))
    trs = bandit.transitions()
    assert len(trs) == n
    # Spot-check the schema.
    t0 = trs[0]
    assert "context" in t0 and len(t0["context"]) == DIM
    assert "op" in t0 and t0["op"] in OPS
    assert "reward" in t0
    assert "shaped_reward" in t0
    assert "n_pulls_before" in t0
    assert "ts" in t0


def test_logging_bandit_records_shaped_reward_when_passed():
    bandit = LoggingBandit(ops=OPS, feature_dim=DIM, seed=2)
    ctx = _ctx()
    bandit.update("a", ctx, reward=1.0, shaped_reward=2.5)
    t = bandit.transitions()[0]
    assert t["reward"] == 1.0
    assert t["shaped_reward"] == 2.5


def _make_sol(inst, route_lists, settings):
    sol = Solution(
        instance_id=inst.instance_id,
        routes=[Route(customers=list(r)) for r in route_lists],
        solver="hand", wall_clock_seconds=0.0, budget_seconds=0.0,
        feasible=True, metrics={},
    )
    sol.metrics = evaluate(inst, sol, settings)
    return sol


def test_shaped_reward_terms_zero_for_identical_solutions():
    inst = generate(N=10, seed=0)
    s = Settings()
    sol = _make_sol(inst, [
        [c.id for c in inst.customers[:5]],
        [c.id for c in inst.customers[5:]],
    ], s)
    terms = shaped_reward_terms(inst, sol, sol, s)
    assert terms["island_disposal"] == 0.0
    assert terms["isolated_stop_disposal"] == 0.0
    assert terms["leg_shrinkage"] == 0.0


def test_shaped_reward_terms_positive_island_disposal_when_islands_removed():
    inst = generate(N=12, seed=1)
    s = Settings()
    cust_ids = [c.id for c in inst.customers]
    # BEFORE: 4 routes, two of which are 2-customer "islands"
    before = _make_sol(inst, [
        cust_ids[:4],
        cust_ids[4:6],   # island (k=2)
        cust_ids[6:8],   # island (k=2)
        cust_ids[8:],
    ], s)
    # AFTER: 2 routes, no islands
    after = _make_sol(inst, [
        cust_ids[:6],
        cust_ids[6:],
    ], s)
    terms = shaped_reward_terms(inst, before, after, s)
    assert terms["island_disposal"] >= 2.0  # 2 islands disposed


def test_shaped_reward_isolated_stop_disposal():
    inst = generate(N=8, seed=4)
    s = Settings()
    cust_ids = [c.id for c in inst.customers]
    before = _make_sol(inst, [
        [cust_ids[0]],          # solo
        [cust_ids[1]],          # solo
        cust_ids[2:],
    ], s)
    after = _make_sol(inst, [cust_ids], s)  # all in one route
    terms = shaped_reward_terms(inst, before, after, s)
    assert terms["isolated_stop_disposal"] == 2.0


def test_total_shaped_reward_empty_coefs_returns_base():
    base = 7.5
    terms = {
        "island_disposal": 1.0,
        "isolated_stop_disposal": 2.0,
        "leg_shrinkage": 3.0,
    }
    out = total_shaped_reward(base, terms, coefs={})
    assert out == base


def test_total_shaped_reward_default_coefs_applies_weights():
    base = 0.0
    terms = {"island_disposal": 1.0, "isolated_stop_disposal": 0.0,
             "leg_shrinkage": 0.0}
    out = total_shaped_reward(base, terms)  # uses DEFAULT_COEFS
    assert abs(out - DEFAULT_COEFS["island_disposal"]) < 1e-9


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(_torch_available(),
                    reason="torch is installed; cannot exercise the missing-torch path.")
def test_mlp_policy_raises_when_torch_missing():
    # If torch is unavailable, the constructor should raise RuntimeError
    # with a helpful hint mentioning torch and a pip suggestion.
    from svrptw.solvers.learning.policy_mlp import MLPBanditPolicy
    with pytest.raises(RuntimeError) as exc:
        MLPBanditPolicy(ops=OPS, feature_dim=DIM)
    msg = str(exc.value).lower()
    assert "torch" in msg


@pytest.mark.skipif(not _torch_available(), reason="torch missing")
def test_mlp_policy_choose_returns_known_op():
    from svrptw.solvers.learning.policy_mlp import MLPBanditPolicy
    policy = MLPBanditPolicy(ops=OPS, feature_dim=DIM, seed=7,
                             update_every=2)
    op = policy.choose(np.zeros(DIM, dtype=np.float64), eps=0.0)
    assert op in OPS
    # Update twice to trigger one REINFORCE step; must not crash.
    for _ in range(3):
        policy.update(op, np.zeros(DIM, dtype=np.float64), reward=0.5)
    s = policy.stats()
    assert op in s and s[op]["n_pulls"] >= 3
