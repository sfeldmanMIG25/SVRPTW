"""Contextual bandit (LinUCB + Thompson) for adaptive operator selection.

SPEC-3-OPSEL-02.  Each arm = one move operator.  Context = state-feature
vector from `state_features.featurize`.  Reward = (cost_before - cost_after)
/ elapsed_seconds — improvement-per-second.

Ropke-Pisinger ALNS reward decay can be layered on top (caller's choice).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinUCBArm:
    """LinUCB arm state: A = X^T X + I,  b = X^T r.

    Estimator: theta_hat = A^{-1} b.  UCB score for context x:
      x^T theta_hat + alpha * sqrt(x^T A^{-1} x).
    """
    d: int
    alpha: float = 1.0
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    n_pulls: int = 0
    total_reward: float = 0.0

    def __post_init__(self) -> None:
        self.A = np.eye(self.d)
        self.b = np.zeros(self.d)

    def score(self, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mean = float(x @ theta)
        bonus = float(self.alpha * math.sqrt(max(0.0, x @ A_inv @ x)))
        return mean + bonus

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n_pulls += 1
        self.total_reward += reward


class LinUCBBandit:
    """Contextual bandit over a fixed pool of operator names."""

    def __init__(self, ops: list[str], feature_dim: int, alpha: float = 1.0,
                 seed: int = 0):
        self.ops = list(ops)
        self.arms: dict[str, LinUCBArm] = {
            op: LinUCBArm(d=feature_dim, alpha=alpha) for op in ops
        }
        self._rng = np.random.default_rng(seed)

    def choose(self, context: np.ndarray, eps: float = 0.05) -> str:
        """Pick an operator.  With prob `eps` explore uniformly, else greedy UCB."""
        if self._rng.random() < eps:
            return str(self._rng.choice(self.ops))
        scores = {op: arm.score(context) for op, arm in self.arms.items()}
        # Tie-break randomly.
        best = max(scores.values())
        contenders = [op for op, s in scores.items() if s >= best - 1e-9]
        return str(self._rng.choice(contenders))

    def update(self, op: str, context: np.ndarray, reward: float) -> None:
        if op in self.arms:
            self.arms[op].update(context, reward)

    def stats(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for op, arm in self.arms.items():
            mean = arm.total_reward / max(1, arm.n_pulls)
            out[op] = {
                "n_pulls": arm.n_pulls,
                "total_reward": arm.total_reward,
                "mean_reward": mean,
            }
        return out
