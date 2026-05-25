"""GART v4 tour-length estimator wrapper.  SPEC-1-GART-01.

The underlying LightGBM model lives in
    D:/VRP-Advanced-Estimator-Integration/estimators/lgbm_model_v4/
We import its inference module directly (its weights are the contract; the
Python entry point is just convenience), then wrap it in a TourLengthEstimator
protocol that takes either raw 2-D coordinates or (k,) node indices into a
travel-time matrix.

Asymmetric calibration (`alpha · MST · (1 + beta · asym_score)`) is implemented
behind ``asymmetric_correction=True``.  ``beta`` is currently 0 and will be
fitted in a follow-up calibration pass against LKH-3 ATSP ground truth (see
SPEC-1-GART-01 §asymmetric extension).
"""
from __future__ import annotations

import functools
import sys
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

_GART_REPO = Path("D:/VRP-Advanced-Estimator-Integration/estimators/lgbm_model_v4")
# feature_engineering.py lives in the upstream training repo (single source of truth);
# the integration repo's estimator imports it by name.
_FEATURE_ENG_REPO = Path("D:/Area-and-Distribution-Free-Estimator-for-TSP/lgbm_model_v4")
_LOCK = threading.Lock()


class TourLengthEstimator(Protocol):
    def estimate(self, nodes: np.ndarray, dist_matrix: np.ndarray | None = None) -> float: ...
    def estimate_batch(self, batch: list[np.ndarray], dist_matrix: np.ndarray | None = None) -> np.ndarray: ...
    def estimate_marginal(self, current: np.ndarray, candidate: int | np.ndarray,
                          dist_matrix: np.ndarray | None = None) -> float: ...


def _mst_length_undirected(M: np.ndarray) -> float:
    """Undirected MST length from a (possibly asymmetric) distance matrix.
    For asymmetric M we symmetrize by min(M, M.T) — using the shorter
    direction for each pair, which is the natural undirected lower bound."""
    n = M.shape[0]
    if n < 2:
        return 0.0
    sym = np.minimum(M, M.T)
    # Prim's algorithm.
    in_tree = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf)
    min_edge[0] = 0.0
    total = 0.0
    for _ in range(n):
        u = int(np.argmin(np.where(in_tree, np.inf, min_edge)))
        if not np.isfinite(min_edge[u]):
            break
        in_tree[u] = True
        total += min_edge[u]
        for v in range(n):
            if not in_tree[v] and sym[u, v] < min_edge[v]:
                min_edge[v] = sym[u, v]
    return float(total)


def _nearest_insertion_delta(current_idx: np.ndarray, candidate_idx: int, D: np.ndarray) -> float:
    """Cost to insert `candidate_idx` into the cycle formed by `current_idx`
    using nearest-insertion (O(k))."""
    if len(current_idx) == 0:
        return 2.0 * D[0, candidate_idx]  # depot -> cand -> depot
    cycle = np.concatenate([[0], current_idx, [0]])
    best = np.inf
    for i in range(len(cycle) - 1):
        a, b = cycle[i], cycle[i + 1]
        delta = D[a, candidate_idx] + D[candidate_idx, b] - D[a, b]
        if delta < best:
            best = delta
    return float(best)


class GartV4Estimator:
    # Calibrated against 500 random asymmetric matrices vs LKH-3 ATSP ground
    # truth.  See svrptw/models/gart/calibrate_asym.py and the report at
    # models/gart/calibration_asym.json.
    DEFAULT_ASYM_BETA = 0.3505

    def __init__(self, model_dir: str | Path | None = None,
                 asymmetric_correction: bool = True,
                 asymmetry_beta: float | None = None,
                 cache_size: int = 100_000):
        model_dir = Path(model_dir) if model_dir is not None else _GART_REPO
        for extra in (_FEATURE_ENG_REPO, model_dir):
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        # Late-import: this file pulls in pandas/lightgbm; keep cold-start cheap.
        from lgbm_estimator_v4 import TSP_V4_LGBM_Estimator  # type: ignore
        self._inner = TSP_V4_LGBM_Estimator(model_dir=str(model_dir))
        self.asymmetric_correction = asymmetric_correction
        self.asymmetry_beta = asymmetry_beta if asymmetry_beta is not None else self.DEFAULT_ASYM_BETA
        self._estimate_cached = functools.lru_cache(maxsize=cache_size)(self._estimate_uncached)

    # ---------- core ----------

    def _coords_from_nodes(self, nodes: np.ndarray, coords: np.ndarray) -> np.ndarray:
        # nodes is an array of integer indices; coords (M, 2) is the full coord table.
        return coords[nodes]

    def _alpha_for_coords(self, coords: np.ndarray, grid_size: float = 100.0) -> float:
        if coords.shape[0] < 3:
            return float("nan")
        out = self._inner.estimate(coords, 2, grid_size)
        return float(out["alpha"])

    def estimate_from_coords(self, coords: np.ndarray, grid_size: float = 100.0) -> float:
        if coords.shape[0] < 3:
            return 0.0
        out = self._inner.estimate(coords, 2, grid_size)
        return float(out["estimate"])

    # ---------- public Protocol ----------

    def estimate(self, nodes: np.ndarray, dist_matrix: np.ndarray | None = None,
                 coords: np.ndarray | None = None) -> float:
        nodes = np.asarray(nodes)
        if nodes.size == 0:
            return 0.0
        if nodes.size == 1 and dist_matrix is not None:
            return 2.0 * float(dist_matrix[0, int(nodes[0])])

        if dist_matrix is not None:
            return self._estimate_with_matrix(tuple(int(x) for x in nodes), dist_matrix)
        if coords is not None:
            return self.estimate_from_coords(coords)
        raise ValueError("estimate() needs either dist_matrix or coords")

    def _estimate_uncached(self, nodes_tuple: tuple[int, ...], matrix_id: int) -> float:
        # The matrix_id is required so the cache invalidates across distinct matrices;
        # the actual matrix is supplied via a thread-local stash in _estimate_with_matrix.
        M = _MATRIX_STASH.get(matrix_id)
        if M is None:
            raise RuntimeError("matrix stash miss — internal cache invariant broken")
        nodes = np.array(nodes_tuple, dtype=np.int64)
        # For k = 1, 2 the optimal closed tour is exact and cheap.
        if nodes.size == 1:
            c = int(nodes[0])
            return float(M[0, c] + M[c, 0])
        if nodes.size == 2:
            a, b = int(nodes[0]), int(nodes[1])
            f = M[0, a] + M[a, b] + M[b, 0]
            r = M[0, b] + M[b, a] + M[a, 0]
            return float(min(f, r))
        # k >= 3: stitch the asymmetric depot legs onto an alpha · MST core.
        nodes_with_depot = np.concatenate([[0], nodes])
        sub = M[np.ix_(nodes_with_depot, nodes_with_depot)]
        mst = _mst_length_undirected(sub)
        if mst == 0.0:
            return float(M[0, nodes].sum() + M[nodes, 0].sum()) / nodes.size
        # We need an alpha; the underlying model wants 2-D coords.  We approximate by
        # multidimensional scaling on the symmetrized matrix for the alpha-feature path.
        coords = _mds_2d(np.minimum(sub, sub.T))
        alpha = self._alpha_for_coords(coords)
        if not np.isfinite(alpha):
            alpha = 1.4  # mid-range fallback
        L = float(alpha * mst)
        if self.asymmetric_correction:
            asym = _asymmetry_score(sub)
            L *= (1.0 + self.asymmetry_beta * asym)
        return L

    def _estimate_with_matrix(self, nodes_tuple: tuple[int, ...], M: np.ndarray) -> float:
        mid = id(M)
        with _LOCK:
            _MATRIX_STASH[mid] = M
        try:
            return self._estimate_cached(nodes_tuple, mid)
        finally:
            pass  # leave the stash populated; LRU will recycle slots as needed

    def estimate_batch(self, batch: list[np.ndarray], dist_matrix: np.ndarray | None = None) -> np.ndarray:
        return np.array([self.estimate(b, dist_matrix=dist_matrix) for b in batch], dtype=np.float64)

    def estimate_marginal(self, current: np.ndarray, candidate: int | np.ndarray,
                          dist_matrix: np.ndarray | None = None) -> float:
        if dist_matrix is None:
            full = np.concatenate([np.asarray(current), np.atleast_1d(candidate)])
            return self.estimate(full) - self.estimate(np.asarray(current))
        return _nearest_insertion_delta(np.asarray(current, dtype=np.int64),
                                        int(candidate), dist_matrix)


# Lightweight thread-shared stash so the lru_cache key can be hashable.
_MATRIX_STASH: dict[int, np.ndarray] = {}


def _mds_2d(D_sym: np.ndarray) -> np.ndarray:
    """Classical MDS to recover 2-D coordinates from a symmetric distance matrix."""
    n = D_sym.shape[0]
    D2 = D_sym ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    w2 = np.clip(w[idx], 0.0, None)
    coords = V[:, idx] * np.sqrt(w2)
    if not np.isfinite(coords).all():
        return np.zeros((n, 2), dtype=np.float64)
    return coords.astype(np.float64)


def _asymmetry_score(M: np.ndarray) -> float:
    n = M.shape[0]
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    a, b = M[iu], M.T[iu]
    denom = np.maximum(np.maximum(a, b), 1e-9)
    return float(np.mean(np.abs(a - b) / denom))


_default: GartV4Estimator | None = None


def get_default_estimator() -> GartV4Estimator:
    global _default
    if _default is None:
        _default = GartV4Estimator()
    return _default


def _cli() -> int:
    import argparse
    import json

    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--customers", required=True, help="comma-separated customer ids (1-indexed)")
    args = p.parse_args()
    inst = load_instance(args.instance)
    ids = [int(x) for x in args.customers.split(",") if x.strip()]
    est = get_default_estimator()
    L = est.estimate(np.array(ids, dtype=np.int64), dist_matrix=inst.travel_time)
    print(json.dumps({"instance": inst.instance_id, "customers": ids, "estimate_minutes": L}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
