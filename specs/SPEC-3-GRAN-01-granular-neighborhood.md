# SPEC-3-GRAN-01 — Granular neighborhood

```
ID:            SPEC-3-GRAN-01
Title:         k-nearest-neighbor restriction for local-search moves
Owner role:    OR Engineer
Status:        FROZEN
Inputs:        Instance, k (default 20)
Outputs:       svrptw.solvers.common.neighbors.GranularNeighborhood
```

## Why

Per Track-2 research (HGS-VRPTW, PyVRP 0.9+): restricting all move
neighborhoods to the k=20 *time-aware* nearest customers per customer
turns O(N²) move evaluation into O(N·k). At N=500 that's a 25× cut.

It also tends to *improve* solution quality because random long-distance
moves rarely improve a tight TW solution — they just waste search budget.

## Behavior

```python
class GranularNeighborhood:
    def __init__(self, inst: Instance, k: int = 20):
        # Precompute, per customer c, the list of (k) closest customers by
        # symmetrized time-aware distance: min(T[c,i], T[i,c]).
        ...

    def neighbors_of(self, cid: int) -> np.ndarray:
        """Return array of k customer ids closest to cid by time-aware distance."""

    def union_of(self, cids: Iterable[int]) -> set[int]:
        """Union of neighbor lists for a set of customers — useful for
        ruin-recreate operators."""
```

Cached per `id(inst)` (same pattern as the cust_by_id cache).

## Acceptance

- For an instance with N=500, `neighbors_of(any)` returns exactly 20 ids
  and the call costs < 5 μs.
- `relocate`, `swap_intra`, `swap_inter`, `cross_exchange`, and SISR's
  reinsertion all accept an optional `neighbors` argument; passing one
  cuts their O(N²) inner loops to O(N·k).
- On N=500 instances, `auction_gart` with k=20 finishes in ≤ 30 s
  (currently 67 s) without operational-cost regression.
- The single-customer relocate move sees its candidate set shrink from
  N to ~k.

## Non-goals

- Dynamic k tuning. Constant per solve.
- Asymmetry-aware separate "in-neighbours" vs "out-neighbours" lists.
  Symmetric (min of both directions) is the standard simplification
  per HGS/PyVRP.

## Dependencies

- SPEC-0-CFG-01 (Settings.solver.params can carry `k`).
