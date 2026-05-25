"""Phase D3 — train MLPBanditPolicy offline by behavior cloning.

Loads JSONL transitions produced by `collect_bandit_logs.py`, fits a
Boltzmann-softmax MLP via reward-weighted cross-entropy (negative
rewards are clipped to 0 so they don't drag the policy toward
operators that hurt). Saves the result to bench/runs/policy_v1.pt.

Usage:
    python bench/scripts/train_policy_offline.py
    python bench/scripts/train_policy_offline.py --epochs 30 \\
        --transitions bench/runs/bandit_transitions.jsonl \\
        --output bench/runs/policy_v1.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

# Bootstrap import path for `python bench/scripts/train_policy_offline.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from svrptw.solvers.learning.policy_mlp import MLPBanditPolicy


def _load_transitions(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _build_xy(transitions: list[dict], ops: list[str],
              op_history_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-transition rows into (X, y, w) arrays.

    X: (T, feature_dim + op_history_len * |ops|)
    y: (T,) action indices
    w: (T,) reward weights (>= 0; negatives clipped to 0).
    """
    op_to_idx = {op: i for i, op in enumerate(ops)}
    n_ops = len(ops)
    feature_dim = len(transitions[0]["context"]) if transitions else 16
    in_dim = feature_dim + op_history_len * n_ops

    # Reconstruct op-history per instance — transitions carry "instance"
    # if produced by collect_bandit_logs.py.
    by_inst: dict[str, list[dict]] = {}
    for t in transitions:
        by_inst.setdefault(t.get("instance", "_default"), []).append(t)

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    w_rows: list[float] = []
    for _, tlist in by_inst.items():
        history: deque[int] = deque(maxlen=op_history_len)
        for t in tlist:
            ctx = np.asarray(t["context"], dtype=np.float32)
            if ctx.shape[0] != feature_dim:
                continue
            op = t["op"]
            if op not in op_to_idx:
                continue
            a = op_to_idx[op]
            oh = np.zeros(op_history_len * n_ops, dtype=np.float32)
            hist = list(history)
            offset = (op_history_len - len(hist)) * n_ops
            for i, idx in enumerate(hist):
                oh[offset + i * n_ops + idx] = 1.0
            X_rows.append(np.concatenate([ctx, oh], axis=0))
            y_rows.append(a)
            r = t.get("shaped_reward")
            if r is None:
                r = t["reward"]
            w_rows.append(max(0.0, float(r)))
            history.append(a)
    return (np.stack(X_rows, axis=0) if X_rows else np.zeros((0, in_dim), dtype=np.float32),
            np.asarray(y_rows, dtype=np.int64),
            np.asarray(w_rows, dtype=np.float32))


def _train(policy: MLPBanditPolicy, X: np.ndarray, y: np.ndarray, w: np.ndarray,
           epochs: int, batch_size: int, seed: int = 0) -> list[float]:
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    losses: list[float] = []
    if n == 0:
        print("No usable transitions; skipping training.", file=sys.stderr)
        return losses
    # Normalise weights for stability (avoid divide-by-zero).
    w_norm = w.copy()
    if w_norm.sum() <= 0:
        w_norm = np.ones_like(w_norm)
    w_norm = w_norm / max(1e-9, w_norm.mean())
    Xt = torch.from_numpy(X).to(policy.device)
    yt = torch.from_numpy(y).to(policy.device)
    wt = torch.from_numpy(w_norm).to(policy.device)
    for epoch in range(int(epochs)):
        idx = np.arange(n)
        rng.shuffle(idx)
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, n, batch_size):
            b = idx[s: s + batch_size]
            bi = torch.from_numpy(b).to(policy.device)
            xb = Xt.index_select(0, bi)
            yb = yt.index_select(0, bi)
            wb = wt.index_select(0, bi)
            logits = policy.net(xb)
            log_probs = F.log_softmax(logits, dim=1)
            chosen_lp = log_probs.gather(1, yb.unsqueeze(1)).squeeze(1)
            loss = -(wb * chosen_lp).mean()
            policy.optim.zero_grad()
            loss.backward()
            policy.optim.step()
            ep_loss += float(loss.item())
            n_batches += 1
        ep_loss /= max(1, n_batches)
        losses.append(ep_loss)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[train] epoch {epoch+1}/{epochs}  loss={ep_loss:.4f}",
                  file=sys.stderr)
    return losses


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transitions", type=str,
                   default="bench/runs/bandit_transitions.jsonl")
    p.add_argument("--output", type=str, default="bench/runs/policy_v1.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--op-history-len", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    tr_path = Path(args.transitions)
    if not tr_path.is_absolute():
        tr_path = repo_root / tr_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not tr_path.exists():
        print(f"Transitions file not found: {tr_path}", file=sys.stderr)
        return 2
    transitions = _load_transitions(tr_path)
    if not transitions:
        print(f"No transitions in {tr_path}", file=sys.stderr)
        return 2


    # Discover ops from the data; preserve insertion order so action
    # indices match the model output dimension.
    seen_ops: list[str] = []
    seen_set: set[str] = set()
    for t in transitions:
        op = t["op"]
        if op not in seen_set:
            seen_set.add(op)
            seen_ops.append(op)
    feature_dim = len(transitions[0]["context"])
    print(f"[train] {len(transitions)} transitions, "
          f"{len(seen_ops)} ops, feature_dim={feature_dim}", file=sys.stderr)
    print(f"[train] op pickup distribution (raw):", file=sys.stderr)
    counts = Counter(t["op"] for t in transitions)
    for op, c in counts.most_common():
        print(f"  {op:20s} {c}", file=sys.stderr)

    policy = MLPBanditPolicy(
        ops=seen_ops, feature_dim=feature_dim,
        op_history_len=args.op_history_len,
        lr=args.lr, temperature=args.temperature,
        device=args.device, seed=args.seed,
    )
    X, y, w = _build_xy(transitions, seen_ops, args.op_history_len)
    print(f"[train] X={X.shape} y={y.shape} w_mean={float(w.mean()):.3f}",
          file=sys.stderr)
    losses = _train(policy, X, y, w, epochs=args.epochs,
                    batch_size=args.batch_size, seed=args.seed)
    policy.save(out_path)
    print(f"[train] saved policy -> {out_path}")
    if losses:
        print(f"[train] final_loss={losses[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
