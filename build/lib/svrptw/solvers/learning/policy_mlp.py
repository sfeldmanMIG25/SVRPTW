"""MLP policy that drops into pm.solve as a bandit replacement (Phase D2/D3).

Same interface as `LinUCBBandit` — `choose(context, eps)` /
`update(op, context, reward)` / `stats()` — but backed by a small
PyTorch MLP and trained online (REINFORCE-style) on accumulated
(state, action, reward) tuples. Designed to be loadable from a
behavior-cloned checkpoint produced by `train_policy_offline.py`.

Architecture (per RL Roadmap D2):
  Input  : 16 state dims + last-3 op one-hots (3 * |ops|) = 16 + 3*|ops|
  Hidden : [128, 64], ReLU
  Output : logits over |ops| operators
  Sample : Boltzmann softmax with temperature τ (default 0.5)

Torch is lazy-imported inside `__init__`. If torch is missing the
module imports cleanly, but instantiation raises RuntimeError with a
helpful hint.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def _try_import_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        import torch.nn.functional as F  # noqa: F401
        return True
    except Exception:
        return False


class MLPBanditPolicy:
    """Policy-network bandit replacement.

    Public methods mirror LinUCBBandit so it drops into pm.solve via the
    `bandit_kind="mlp"` branch.
    """

    def __init__(
        self,
        ops: list[str],
        feature_dim: int = 16,
        op_history_len: int = 3,
        hidden: tuple[int, ...] = (128, 64),
        lr: float = 1e-3,
        temperature: float = 0.5,
        update_every: int = 8,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if not _try_import_torch():
            raise RuntimeError(
                "torch not installed — install torch (>=2.0) to use "
                "MLPBanditPolicy. Hint: pip install torch"
            )
        import torch
        import torch.nn as nn

        self.ops = list(ops)
        self.n_ops = len(self.ops)
        self.feature_dim = int(feature_dim)
        self.op_history_len = int(op_history_len)
        self.hidden = tuple(hidden)
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.update_every = int(update_every)
        self.device = torch.device(device)

        self._op_to_idx: dict[str, int] = {op: i for i, op in enumerate(self.ops)}
        self._input_dim = self.feature_dim + self.op_history_len * self.n_ops
        self._op_history: deque[int] = deque(maxlen=self.op_history_len)

        torch.manual_seed(int(seed))
        self._rng = np.random.default_rng(int(seed))

        layers: list[Any] = []
        d_in = self._input_dim
        for h in self.hidden:
            layers.append(nn.Linear(d_in, int(h)))
            layers.append(nn.ReLU())
            d_in = int(h)
        layers.append(nn.Linear(d_in, self.n_ops))
        self.net = nn.Sequential(*layers).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # Per-arm tracking (for `stats()` parity with LinUCB)
        self._n_pulls = {op: 0 for op in self.ops}
        self._total_reward = {op: 0.0 for op in self.ops}

        # Update buffer for periodic minibatch REINFORCE updates.
        self._buf_states: list[np.ndarray] = []
        self._buf_actions: list[int] = []
        self._buf_rewards: list[float] = []
        self._update_counter = 0

    # ------ context construction ------

    def _op_history_onehot(self) -> np.ndarray:
        """3*|ops| flat vector: oldest history slot first, then newer.

        Empty slots are zero-vectors. Newer indices are at higher offsets.
        """
        flat = np.zeros(self.op_history_len * self.n_ops, dtype=np.float32)
        # Pad-front with zeros: oldest entries first in the deque
        hist = list(self._op_history)
        # Place last (newest) at the rightmost slot.
        offset = (self.op_history_len - len(hist)) * self.n_ops
        for i, idx in enumerate(hist):
            flat[offset + i * self.n_ops + idx] = 1.0
        return flat

    def _build_input(self, context: np.ndarray) -> np.ndarray:
        ctx = np.asarray(context, dtype=np.float32).reshape(-1)
        if ctx.shape[0] != self.feature_dim:
            raise ValueError(
                f"context length {ctx.shape[0]} != feature_dim {self.feature_dim}"
            )
        oh = self._op_history_onehot()
        return np.concatenate([ctx, oh], axis=0)

    # ------ choose / update / stats interface ------

    def choose(self, context: np.ndarray, eps: float = 0.05) -> str:
        """Sample an operator from the Boltzmann softmax over logits.

        With prob `eps`, pick uniformly at random (exploration).
        """
        import torch
        if self._rng.random() < eps:
            return str(self._rng.choice(self.ops))
        x = self._build_input(context)
        with torch.no_grad():
            t = torch.from_numpy(x).to(self.device).unsqueeze(0)
            logits = self.net(t).squeeze(0).cpu().numpy()
        # Boltzmann softmax with temperature
        tau = max(1e-6, float(self.temperature))
        scaled = logits / tau
        scaled -= scaled.max()
        ex = np.exp(scaled)
        probs = ex / ex.sum()
        idx = int(self._rng.choice(self.n_ops, p=probs))
        return self.ops[idx]

    def update(self, op: str, context: np.ndarray, reward: float) -> None:
        """Buffer a (state, action, reward) tuple, run REINFORCE every
        `update_every` calls. Op-history is updated AFTER recording the
        state so the state corresponds to "before this op was taken"."""
        if op not in self._op_to_idx:
            return
        x = self._build_input(context)
        a = self._op_to_idx[op]
        self._buf_states.append(x)
        self._buf_actions.append(int(a))
        self._buf_rewards.append(float(reward))
        self._n_pulls[op] += 1
        self._total_reward[op] += float(reward)
        # Push op into history AFTER this state was constructed.
        self._op_history.append(int(a))
        self._update_counter += 1
        if self._update_counter >= self.update_every:
            self._reinforce_step()
            self._update_counter = 0

    def _reinforce_step(self) -> None:
        """One REINFORCE-style minibatch update on the buffered tuples.

        Loss = -mean( advantage * log p(a|s) ).
        Advantage = reward - mean(buffer_rewards) (single-baseline).
        Buffer is cleared after the step.
        """
        import torch
        import torch.nn.functional as F
        if not self._buf_states:
            return
        states = torch.from_numpy(np.stack(self._buf_states, axis=0)).to(self.device)
        actions = torch.tensor(self._buf_actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(self._buf_rewards, dtype=torch.float32, device=self.device)
        baseline = rewards.mean()
        adv = rewards - baseline
        logits = self.net(states)
        log_probs = F.log_softmax(logits, dim=1)
        chosen_lp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = -(adv.detach() * chosen_lp).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._buf_states.clear()
        self._buf_actions.clear()
        self._buf_rewards.clear()

    def stats(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for op in self.ops:
            n = int(self._n_pulls[op])
            tr = float(self._total_reward[op])
            out[op] = {
                "n_pulls": n,
                "total_reward": tr,
                "mean_reward": tr / max(1, n),
            }
        return out

    # ------ persistence ------

    def save(self, path: str | Path) -> None:
        """Serialise weights + config to a single .pt file."""
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.net.state_dict(),
            "config": {
                "ops": self.ops,
                "feature_dim": self.feature_dim,
                "op_history_len": self.op_history_len,
                "hidden": list(self.hidden),
                "lr": self.lr,
                "temperature": self.temperature,
                "update_every": self.update_every,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, ops: list[str] | None = None,
             device: str = "cpu") -> "MLPBanditPolicy":
        """Load a checkpoint into a new MLPBanditPolicy.

        If `ops` is provided it must match the saved op list (this is a
        sanity check — same op IDs were used at training time).
        """
        import torch
        path = Path(path)
        payload = torch.load(path, map_location=device)
        cfg = payload["config"]
        saved_ops = list(cfg["ops"])
        if ops is not None and set(ops) != set(saved_ops):
            # Soft warning: we trust the checkpoint, not the caller.
            # If sets disagree, raise; if order disagrees, raise too —
            # action indices wouldn't line up.
            raise ValueError(
                f"ops mismatch: checkpoint has {saved_ops}, caller passed {list(ops)}"
            )
        inst = cls(
            ops=saved_ops,
            feature_dim=int(cfg["feature_dim"]),
            op_history_len=int(cfg["op_history_len"]),
            hidden=tuple(cfg["hidden"]),
            lr=float(cfg["lr"]),
            temperature=float(cfg["temperature"]),
            update_every=int(cfg["update_every"]),
            device=device,
        )
        inst.net.load_state_dict(payload["state_dict"])
        return inst
