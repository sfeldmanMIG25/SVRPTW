"""Granular neighborhood — k nearest customers per customer, time-aware.

SPEC-3-GRAN-01.  Used to prune move-evaluation neighborhoods from O(N) to
O(k), giving the biggest single-line speedup at N=500.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from svrptw.io import Instance

_NEIGHBOR_CACHE: dict[tuple[int, int], GranularNeighborhood] = {}


class GranularNeighborhood:
    __slots__ = ("k", "_neighbors")

    def __init__(self, inst: Instance, k: int = 20):
        self.k = k
        # Use the symmetrised time-aware distance: min(T[i,j], T[j,i]).
        # This is the standard simplification for granular neighbourhoods
        # (HGS, PyVRP); the move evaluators handle asymmetry separately
        # when computing actual move costs.
        T = inst.travel_time
        sym = np.minimum(T, T.T)
        np.fill_diagonal(sym, np.inf)
        # For each customer (rows 1..N), pick the k smallest entries.
        n_total = T.shape[0]
        kk = min(k, n_total - 2)  # exclude self and depot
        self._neighbors = np.empty((n_total, kk), dtype=np.int64)
        for cid in range(n_total):
            # Exclude depot (column 0) as a "neighbor" — moves never target it.
            row = sym[cid].copy()
            row[0] = np.inf
            idx = np.argpartition(row, kk)[:kk]
            # Sort the small partition.
            self._neighbors[cid] = idx[np.argsort(row[idx])]

    def neighbors_of(self, cid: int) -> np.ndarray:
        """Return up to k customer ids closest to `cid` (excluding depot and self).
        Returned ids are 1..N (depot is index 0)."""
        return self._neighbors[cid]

    def union_of(self, cids: Iterable[int]) -> set[int]:
        """Union of neighbor lists for a set of customers."""
        out: set[int] = set()
        for c in cids:
            out.update(self._neighbors[int(c)].tolist())
        return out


def get_neighbors(inst: Instance, k: int = 20) -> GranularNeighborhood:
    """Cached per (instance-id, k) — building the matrix is O(N² log k)."""
    key = (id(inst), k)
    gn = _NEIGHBOR_CACHE.get(key)
    if gn is None:
        gn = GranularNeighborhood(inst, k=k)
        _NEIGHBOR_CACHE[key] = gn
    return gn
