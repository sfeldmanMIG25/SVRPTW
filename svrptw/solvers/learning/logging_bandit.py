"""LoggingBandit: wraps LinUCBBandit and records every (state, action, reward)
transition for offline policy learning (Phase D3 of the RL Roadmap).

The wrapper preserves the LinUCBBandit interface — `choose`, `update`,
`stats` — so it drops into pm.solve unchanged. Each call to `update`
appends a transition dict to an in-memory list; callers may also pass a
JSONL path to flush transitions on demand.

Transition shape:
    {
      "context":        list[float],   # 16-dim state vector
      "op":             str,           # operator name
      "reward":         float,         # raw reward passed to update()
      "shaped_reward":  float | None,  # post-shaping (None if not applied)
      "n_pulls_before": int,           # this op's prior pull count
      "ts":             float,         # epoch
    }
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from svrptw.solvers.learning.bandit import LinUCBBandit


class LoggingBandit:
    """LinUCBBandit + a side-channel transition log.

    Drop-in replacement: same `choose(context, eps)` / `update(op, ctx, reward)`
    / `stats()` interface as `LinUCBBandit`. Additionally maintains an
    in-memory `_transitions` list and (optionally) writes transitions to a
    JSONL file on `flush(path)`.

    The shaped-reward parameter on update() is optional — callers that
    apply Phase D4 shaping can pass it so it lands in the log alongside
    the raw reward used by the underlying bandit.
    """

    def __init__(
        self,
        ops: list[str],
        feature_dim: int,
        alpha: float = 1.0,
        seed: int = 0,
        flush_path: str | Path | None = None,
    ) -> None:
        self._inner = LinUCBBandit(ops=ops, feature_dim=feature_dim,
                                   alpha=alpha, seed=seed)
        self.ops = list(ops)
        self.flush_path: str | Path | None = flush_path
        self._transitions: list[dict[str, Any]] = []

    # --- LinUCBBandit interface passthrough ---

    def choose(self, context: np.ndarray, eps: float = 0.05) -> str:
        return self._inner.choose(context, eps=eps)

    def update(self, op: str, context: np.ndarray, reward: float,
               shaped_reward: float | None = None) -> None:
        # Capture pull count BEFORE update so it reflects history.
        n_before = int(self._inner.arms[op].n_pulls) if op in self._inner.arms else 0
        ctx_list = (context.tolist() if isinstance(context, np.ndarray)
                    else list(context))
        self._transitions.append({
            "context": ctx_list,
            "op": op,
            "reward": float(reward),
            "shaped_reward": (None if shaped_reward is None
                              else float(shaped_reward)),
            "n_pulls_before": n_before,
            "ts": time.time(),
        })
        self._inner.update(op, context, reward)

    def stats(self) -> dict[str, dict[str, float]]:
        return self._inner.stats()

    # --- LoggingBandit-specific accessors ---

    def transitions(self) -> list[dict[str, Any]]:
        """Return the in-memory list of recorded transitions."""
        return self._transitions

    def flush(self, path: str | Path | None = None) -> Path:
        """Write the recorded transitions to a JSONL file.

        If `path` is None, uses the constructor-supplied `flush_path`.
        Raises ValueError if neither is set. Returns the path written.
        """
        target = path if path is not None else self.flush_path
        if target is None:
            raise ValueError("LoggingBandit.flush requires a path "
                             "(constructor `flush_path` or argument).")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for t in self._transitions:
                f.write(json.dumps(t) + "\n")
        return target
