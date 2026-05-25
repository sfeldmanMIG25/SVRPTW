"""Iterable training-data pipeline for POMO.

Generates synthetic instances on the fly — no disk cache for small N,
optional disk cache for large N.  Worker-safe via split seeding.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from svrptw.instances_gen.synthetic import generate as _generate
from svrptw.io import Instance


@dataclass
class POMOInstanceConfig:
    N_choices: tuple[int, ...] = (50,)
    asymmetry: float = 0.05
    tw_tightness: float = 0.3
    # When N >= sparse_threshold, omit the dense matrix from training-time
    # instances (use sparse-k neighbour features only).  Inference and
    # evaluation reconstruct the matrix as needed.
    sparse_threshold: int = 2000
    sparse_k: int = 64


class POMOInstanceDataset(torch.utils.data.IterableDataset):
    """Infinite stream of fresh synthetic Instances.

    Curriculum is encoded as `N_choices`: each iteration samples one N
    uniformly from the list.  Pass `[50]` for fixed-N training,
    `[50, 100, 200]` for mixed-N curriculum.

    Each worker gets a disjoint seed stream via `worker_info.id`.
    """

    def __init__(self, cfg: POMOInstanceConfig, base_seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.base_seed = base_seed

    def _worker_seed_base(self) -> int:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.base_seed
        return self.base_seed + info.id * 1_000_003

    def __iter__(self) -> Iterator[Instance]:
        import random
        rng = random.Random(self._worker_seed_base())
        i = 0
        while True:
            N = rng.choice(self.cfg.N_choices)
            seed = self._worker_seed_base() + i
            include_matrix = N < self.cfg.sparse_threshold
            sparse_k = self.cfg.sparse_k if N >= self.cfg.sparse_threshold else None
            yield _generate(
                N=N, seed=seed,
                asymmetry=self.cfg.asymmetry,
                tw_tightness=self.cfg.tw_tightness,
                include_matrix=include_matrix,
                sparse_k=sparse_k,
            )
            i += 1


def estimate_gpu_memory_mb(N: int, K: int, batch: int, dim: int = 128,
                           layers: int = 4) -> float:
    """Rough memory budget for the POMO forward pass.

    Components (float32 = 4 bytes):
      - encoder activations: O(batch * N * dim * layers)
      - edge feats         : O(batch * N * N * 2)
      - decoder logits      : O(batch * K * N) per step
      - tour memory         : O(batch * K * N) ints

    Returns peak estimate in MB (heuristic; actual depends on autograd).
    """
    enc = 4 * batch * (N + 1) * dim * layers * 4    # *4 for fwd+bwd headroom
    edge = 4 * batch * (N + 1) * (N + 1) * 2
    dec = 4 * batch * K * (N + 1)
    tour = 4 * batch * K * (N + 1)
    return (enc + edge + dec + tour) / (1024 * 1024)


def pick_safe_config(N: int, gpu_mem_gb: float = 7.5) -> dict:
    """Choose K, batch, dim, layers that fit within `gpu_mem_gb` (leaves
    0.5 GB headroom on an 8 GB card).  Conservative; favour stability
    over throughput."""
    budget_mb = gpu_mem_gb * 1024
    # Prefer K = min(N, 32) for stability per POMO research.
    K = min(N, 32)
    for batch, dim, layers in [
        (16, 128, 4), (8, 128, 4), (4, 128, 4),
        (4, 128, 2), (2, 128, 2), (1, 64, 2), (1, 32, 2),
    ]:
        est = estimate_gpu_memory_mb(N, K, batch, dim, layers)
        if est < budget_mb:
            return {"batch": batch, "K": K, "dim": dim, "layers": layers,
                    "est_mb": round(est, 1)}
    return {"batch": 1, "K": 8, "dim": 32, "layers": 2,
            "est_mb": round(estimate_gpu_memory_mb(N, 8, 1, 32, 2), 1)}


if __name__ == "__main__":
    # Smoke: print memory budget across the curriculum.
    print(f"{'N':>6} {'est_mb':>10} {'config'}")
    for N in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        cfg = pick_safe_config(N)
        print(f"{N:>6} {cfg['est_mb']:>10.1f} {cfg}")
