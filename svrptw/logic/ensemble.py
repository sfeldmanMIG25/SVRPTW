"""SPEC-6-LOGIC-01 — LogicEnsemble: 5 distilled student heads + σ-based
authoritative flag.

Each head is a LogicStudent trained on a bootstrap resample of the
preference-pair set (with a different RNG seed). At inference, the
ensemble returns:
  - score.mean: float in [0,1]  — point estimate
  - score.std:  float           — disagreement signal
  - score.authoritative: bool   — std ≤ threshold (default 0.08)

Downstream consumers (e.g. SPEC-6-BANDIT-PLATEAU-01 basin-jump) must
treat non-authoritative predictions as advisory.

Scaling: heads are tiny MLPs (~30k params each), so 5 of them inference
in ≤ 250 ms cumulative on CPU — well within the solver's hot-loop budget
when basin-jump is rare.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from svrptw.io import Instance
from svrptw.logic.features import FEATURE_DIM, extract
from svrptw.logic.student import LogicStudent, StudentConfig, train
from svrptw.solvers.common import Solution


_LOG = logging.getLogger("svrptw.logic.ensemble")


@dataclass
class EnsembleScore:
    mean: float
    std: float
    authoritative: bool

    def to_dict(self) -> dict:
        return asdict(self)


class LogicEnsemble:
    """N independent LogicStudent heads, ensembled at inference."""

    def __init__(self, heads: list[LogicStudent], authoritative_max_std: float = 0.08):
        if not heads:
            raise ValueError("LogicEnsemble needs at least one head")
        self._heads = heads
        self._max_std = float(authoritative_max_std)

    @property
    def n_heads(self) -> int:
        return len(self._heads)

    def score(self, inst: Instance, sol: Solution) -> EnsembleScore:
        feats = extract(inst, sol).unsqueeze(0)
        per_head: list[float] = []
        for h in self._heads:
            h.eval()
            with torch.no_grad():
                logit = h(feats).squeeze(0)
                T = max(h.temperature, 1e-3)
                per_head.append(float(torch.sigmoid(logit / T)))
        arr = np.array(per_head, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        return EnsembleScore(
            mean=mean, std=std, authoritative=(std <= self._max_std)
        )

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for i, h in enumerate(self._heads):
            torch.save({
                "state_dict": h.state_dict(),
                "cfg": asdict(h.cfg),
                "log_T": float(h.log_T),
            }, dir_path / f"head_{i:02d}.pt")
        # Save ensemble-level config.
        (dir_path / "ensemble.json").write_text(
            f'{{"n_heads": {len(self._heads)}, "max_std": {self._max_std}}}',
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir_path: Path,
             authoritative_max_std: Optional[float] = None) -> "LogicEnsemble":
        dir_path = Path(dir_path)
        head_paths = sorted(dir_path.glob("head_*.pt"))
        if not head_paths:
            raise FileNotFoundError(f"no head_*.pt in {dir_path}")
        heads: list[LogicStudent] = []
        for p in head_paths:
            blob = torch.load(p, weights_only=False)
            cfg = StudentConfig(**blob["cfg"])
            h = LogicStudent(cfg)
            h.load_state_dict(blob["state_dict"])
            with torch.no_grad():
                h.log_T.copy_(torch.tensor(blob["log_T"]))
            heads.append(h)
        max_std = authoritative_max_std if authoritative_max_std is not None else 0.08
        if (dir_path / "ensemble.json").exists():
            import json
            cfg = json.loads((dir_path / "ensemble.json").read_text(encoding="utf-8"))
            if authoritative_max_std is None:
                max_std = float(cfg.get("max_std", 0.08))
        return cls(heads, authoritative_max_std=max_std)


def train_ensemble(
    pairs: list[dict],
    instances_dir: Path,
    *,
    n_heads: int = 5,
    val_frac: float = 0.2,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    base_seed: int = 0,
    bootstrap_frac: float = 0.8,
    authoritative_max_std: float = 0.08,
) -> tuple[LogicEnsemble, list[dict]]:
    """Train `n_heads` independent LogicStudents on bootstrap resamples.

    Returns (ensemble, list-of-per-head-histories). Each head sees an
    80 %-with-replacement bootstrap of the pair set and a different
    random seed for initialisation and val/train split. The bootstrap
    is the standard recipe for ensemble σ to approximate
    epistemic uncertainty (Lakshminarayanan 2017).
    """
    rng = np.random.default_rng(base_seed)
    histories = []
    heads: list[LogicStudent] = []
    n = len(pairs)
    if n == 0:
        raise RuntimeError("no pairs to train on")

    for k in range(n_heads):
        torch.manual_seed(base_seed + k * 977)
        head_seed = int(rng.integers(0, 1_000_000))
        # Bootstrap sample with replacement.
        idx = rng.choice(n, size=int(n * bootstrap_frac), replace=True)
        sample = [pairs[i] for i in idx.tolist()]
        student, hist = train(
            sample,
            instances_dir=instances_dir,
            val_frac=val_frac,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=head_seed,
        )
        heads.append(student)
        histories.append({"head": k, **hist})

    return LogicEnsemble(heads, authoritative_max_std=authoritative_max_std), histories
