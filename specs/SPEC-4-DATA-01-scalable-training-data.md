# SPEC-4-DATA-01 — Scalable training data for POMO/learning

```
ID:            SPEC-4-DATA-01
Title:         Synthetic instance generator + dataset pipeline for training up to N=10,000
Owner role:    ML Engineer
Status:        FROZEN
Inputs:        N range, seed, asymmetry strength, output dir (optional)
Outputs:       svrptw.instances_gen.synthetic + svrptw.solvers.learning.pomo.dataset
```

## Goal

Train POMO at sizes well beyond the 160-instance v1 OSM set. We need:
- Cheap synthetic instances (no OSM Dijkstra), parametric difficulty.
- N up to **10,000** without OOM.
- Streaming / on-the-fly generation so disk doesn't bottleneck.
- Memory profile that doesn't grow worse than O(N²) for the matrix.

## Memory math

Float32 travel-time matrix at N+1 nodes:

| N | matrix size | matrices held in mem (B=8 batch × 2 streams) |
|---|---|---|
| 50 | 0.01 MB | 0.16 MB |
| 100 | 0.04 MB | 0.6 MB |
| 200 | 0.16 MB | 2.6 MB |
| 500 | 1.0 MB | 16 MB |
| 1000 | 4 MB | 64 MB |
| 5000 | 100 MB | 1.6 GB |
| 10000 | 400 MB | 6.4 GB |

At N=10,000 even one matrix is 400 MB — batches > 1 are out without sparsification.

## Strategy by scale

**N ≤ 1000 (small).** Dense per-instance matrices in GPU. Standard MatNet
encoder. Batch ≥ 4.

**N = 1000–5000 (medium).** Dense matrix on CPU, transfer one at a time
to GPU. Batch = 1. Use sparse k-nearest attention (k=64) in the encoder
to keep activations bounded. Encoder layers reduced 4 → 2.

**N > 5000 (large).** **Hierarchical / windowed.** Two paths:
1. **Cluster-then-route**: K-means partition into ~N/200 sub-instances, solve each with the small-scale POMO, reassemble. Loses 1-3% gap, fits in any GPU.
2. **Local-attention transformer (GLOP-style, Ye et al. 2024)**: only attend within k-nearest, never materialize the full (N,N) attention matrix.

The synthetic generator must support both — emit instances with optional cluster IDs (for path 1) and per-node k-neighbour lists (for path 2).

## API

```python
# svrptw/instances_gen/synthetic.py
def generate(N: int, seed: int = 0, *,
             asymmetry: float = 0.05,
             demand_range: tuple[int, int] = (1, 10),
             tw_tightness: float = 0.3,
             include_matrix: bool = True,
             sparse_k: int | None = None,
            ) -> Instance:
    """Euclidean random points + asymmetric noise.  Matrix included by
    default; pass `sparse_k` to attach only the k-nearest neighbour list
    per node (omit the dense (N+1, N+1) matrix from memory)."""
```

```python
# svrptw/solvers/learning/pomo/dataset.py
class POMOInstanceDataset(torch.utils.data.IterableDataset):
    def __init__(self, N_choices: list[int], seed: int = 0,
                 asymmetry: float = 0.05): ...
    def __iter__(self):
        # Infinite stream of fresh synthetic Instances; one per yield.
        # Worker-safe with split seeding.
```

## Acceptance

- `python -m svrptw.instances_gen.synthetic --N 10000 --seed 0`
  produces a `synthetic-N10000-S0` Instance in < 10 seconds and
  uses < 800 MB peak RAM.
- The dataset yields valid Instances at N=50, 200, 1000 with no leaks
  across 1000 iterations.
- POMO training script (`train.py`) accepts `--source synthetic
  --n-curriculum 50,100,200,500` and walks the curriculum.

## Non-goals

- Real road-network instances at N > 500 (still OSMnx, separate spec).
- Solving N=10,000 instances with POMO directly — the spec is about
  *training data*. The hierarchical inference path is its own follow-up.

## Dependencies

- SPEC-0-CFG-01 (Settings.seed).
- SPEC-0-INST-01 (the Instance dataclass we extend).
