"""SPEC-6-LOGIC-01 — LogicStudent: distilled cheap-inference judge.

3-layer MLP over 32-d solution features; Bradley-Terry loss on
preference pairs; temperature-scaled calibration so the scalar in
[0,1] is interpretable as P(dispatcher ships it).

Architecture intentionally small — must hit <50 ms on CPU at N=200.
Image-embedding path (256-d frozen ViT) is queued as a follow-up
extension; current path uses only the 32-d engineered features.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from svrptw.io import Instance
from svrptw.logic.features import FEATURE_DIM, extract
from svrptw.solvers.common import Solution


@dataclass
class StudentConfig:
    feature_dim: int = FEATURE_DIM
    hidden: int = 128
    dropout: float = 0.1
    n_layers: int = 3      # input → hidden → hidden → 1


class LogicStudent(nn.Module):
    """Compact MLP scoring a (instance, solution) pair in [0, 1]."""

    def __init__(self, cfg: Optional[StudentConfig] = None):
        super().__init__()
        self.cfg = cfg or StudentConfig()
        layers: list[nn.Module] = []
        in_dim = self.cfg.feature_dim
        for i in range(self.cfg.n_layers - 1):
            layers += [nn.Linear(in_dim, self.cfg.hidden), nn.GELU(),
                       nn.Dropout(self.cfg.dropout)]
            in_dim = self.cfg.hidden
        layers += [nn.Linear(in_dim, 1)]
        self.body = nn.Sequential(*layers)
        # Temperature for calibration after Bradley-Terry training.
        self.log_T = nn.Parameter(torch.zeros(()), requires_grad=False)

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_T))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, F). Returns raw score logit (B,) — pre-sigmoid."""
        return self.body(feats).squeeze(-1)

    def score(self, inst: Instance, sol: Solution) -> float:
        """Calibrated P(dispatcher ships) in [0, 1]."""
        self.eval()
        with torch.no_grad():
            x = extract(inst, sol).unsqueeze(0)
            logit = self.forward(x).squeeze(0)
            return float(torch.sigmoid(logit / max(self.temperature, 1e-3)))

    def score_batch(self, items: list[tuple[Instance, Solution]]) -> list[float]:
        self.eval()
        with torch.no_grad():
            feats = torch.stack([extract(i, s) for (i, s) in items], dim=0)
            logits = self.forward(feats)
            return torch.sigmoid(logits / max(self.temperature, 1e-3)).tolist()


def bradley_terry_loss(
    student: LogicStudent, feats_a: torch.Tensor, feats_b: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """P(A ≻ B) = sigmoid(s_A - s_B). BCE over labels.

    labels: (B,) float — 1.0 if A≻B, 0.0 if B≻A, 0.5 if tie.
    """
    s_a = student(feats_a)
    s_b = student(feats_b)
    log_p_a = F.logsigmoid(s_a - s_b)
    log_p_b = F.logsigmoid(s_b - s_a)
    # Symmetric cross-entropy:  -[y log P(A≻B) + (1-y) log P(B≻A)]
    return -(labels * log_p_a + (1.0 - labels) * log_p_b).mean()


def fit_temperature(
    student: LogicStudent, feats_a: torch.Tensor, feats_b: torch.Tensor,
    labels: torch.Tensor, *, max_iters: int = 200, lr: float = 0.05,
) -> None:
    """Single-parameter temperature calibration on a held-out val set.

    Minimises BCE on the *calibrated* P(A≻B) = sigmoid((s_A - s_B)/T).
    Updates `student.log_T` in place; body parameters unchanged.
    """
    with torch.no_grad():
        s_a = student(feats_a)
        s_b = student(feats_b)
    log_T = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.LBFGS([log_T], max_iter=max_iters, lr=lr,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        log_p_a = F.logsigmoid((s_a - s_b) / T)
        log_p_b = F.logsigmoid((s_b - s_a) / T)
        loss = -(labels * log_p_a + (1.0 - labels) * log_p_b).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        student.log_T.copy_(log_T.data)


def train(
    pairs: list[dict],
    instances_dir: Path,
    *,
    val_frac: float = 0.2,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[LogicStudent, dict]:
    """End-to-end training from a `dataset.build_pairs` JSON list.

    Returns (trained_student, train_history). Skips pairs with NaN
    labels or missing instance files. Validation is by `instance_id`
    so the student doesn't see val instances during training.
    """
    from svrptw.io import load_instance
    from svrptw.logic.dataset import _reconstruct_solution

    rng = np.random.default_rng(seed)

    # Materialise features for each pair. Skip incomplete rows.
    feats_a_list = []
    feats_b_list = []
    labels_list = []
    iids = []
    for p in pairs:
        if p.get("label") is None or (isinstance(p["label"], float) and (p["label"] != p["label"])):
            continue
        iid = p["instance_id"]
        ipath = instances_dir / f"{iid}.json"
        if not ipath.exists():
            continue
        inst = load_instance(str(ipath))
        sol_a = _reconstruct_solution(inst, {"solver": p["solver_a"]})
        sol_b = _reconstruct_solution(inst, {"solver": p["solver_b"]})
        if sol_a is None or sol_b is None:
            continue
        feats_a_list.append(extract(inst, sol_a))
        feats_b_list.append(extract(inst, sol_b))
        labels_list.append(float(p["label"]))
        iids.append(iid)

    if not labels_list:
        raise RuntimeError("no usable pairs after reconstruction")

    feats_a = torch.stack(feats_a_list)
    feats_b = torch.stack(feats_b_list)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    iids_arr = np.array(iids)

    # Split by instance_id.
    uniq_iids = sorted(set(iids))
    rng.shuffle(uniq_iids)
    n_val = max(1, int(len(uniq_iids) * val_frac))
    val_iids = set(uniq_iids[:n_val])
    train_mask = np.array([iid not in val_iids for iid in iids])
    val_mask = ~train_mask

    student = LogicStudent()
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    n_train = int(train_mask.sum())
    for epoch in range(epochs):
        student.train()
        perm = rng.permutation(n_train)
        train_idx = np.where(train_mask)[0][perm]
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[s:s + batch_size]
            fa = feats_a[batch_idx]
            fb = feats_b[batch_idx]
            y = labels[batch_idx]
            loss = bradley_terry_loss(student, fa, fb, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history["train_loss"].append(epoch_loss / max(1, n_batches))

        # Val
        student.eval()
        with torch.no_grad():
            val_idx = np.where(val_mask)[0]
            fa = feats_a[val_idx]
            fb = feats_b[val_idx]
            y = labels[val_idx]
            val_loss = float(bradley_terry_loss(student, fa, fb, y))
            preds = (student(fa) > student(fb)).float()
            # Accuracy vs binary labels (treat 0.5 as either-direction tolerated).
            tol = (y == 0.5).float()
            acc = ((preds == y).float() + tol * 0.5).mean().item()
            history["val_loss"].append(val_loss)
            history["val_acc"].append(float(acc))

    # Calibrate temperature on val set.
    val_idx = np.where(val_mask)[0]
    fit_temperature(student, feats_a[val_idx], feats_b[val_idx], labels[val_idx])

    return student, history
