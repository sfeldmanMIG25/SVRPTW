"""Synthetic VRPTW instance generator — Euclidean + asymmetric noise.

Designed for ML training data: cheap to generate, parametric difficulty,
memory-aware (skip the dense matrix on request and emit only sparse
k-nearest neighbour structure for huge N).

CLI:
    python -m svrptw.instances_gen.synthetic --N 1000 --seed 0
    python -m svrptw.instances_gen.synthetic --N 10000 --seed 0 --sparse-k 64
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from svrptw.io import (
    BreakRegime,
    Charger,
    Customer,
    Depot,
    Instance,
    VehicleSpec,
    save_instance,
)

_VEHICLE_CLASSES = {
    "van":     {"capacity_range": (40, 60),   "fuel": None,  "rate": None, "weight": 0.55},
    "box":     {"capacity_range": (80, 120),  "fuel": None,  "rate": None, "weight": 0.30},
    "tractor": {"capacity_range": (180, 220), "fuel": None,  "rate": None, "weight": 0.10},
    "ev":      {"capacity_range": (50, 80),   "fuel": 150.0, "rate": 5.0,  "weight": 0.05},
}


def _build_vehicle_fleet(num_vehicles: int, rng: np.random.Generator,
                         include_ev: bool = False) -> list[VehicleSpec]:
    if include_ev:
        scaled = {k: v["weight"] for k, v in _VEHICLE_CLASSES.items()}
        scaled["ev"] = 0.30
        s = sum(scaled.values())
        weights = {k: v / s for k, v in scaled.items()}
    else:
        s = sum(v["weight"] for k, v in _VEHICLE_CLASSES.items() if k != "ev")
        weights = {k: v["weight"] / s for k, v in _VEHICLE_CLASSES.items() if k != "ev"}
    classes = list(weights.keys())
    probs = np.array([weights[k] for k in classes], dtype=np.float64)
    vehs: list[VehicleSpec] = []
    for _ in range(num_vehicles):
        cls = classes[int(rng.choice(len(classes), p=probs))]
        meta = _VEHICLE_CLASSES[cls]
        cap_lo, cap_hi = meta["capacity_range"]
        vehs.append(VehicleSpec(
            vclass=cls,
            capacity=int(rng.integers(cap_lo, cap_hi + 1)),
            fuel_capacity=meta["fuel"],
            recharge_rate=meta["rate"],
        ))
    return vehs


def _build_breaks(regime: str, n_customers: int,
                  rng: np.random.Generator) -> BreakRegime | None:
    if regime == "none":
        return None
    if regime == "EU561":
        drive_limit, break_dur = 270, 45
    elif regime == "FMCSA":
        drive_limit, break_dur = 480, 30
    else:
        return None
    n_rest = max(5, int(0.10 * n_customers))
    rest_ids = sorted(rng.choice(np.arange(1, n_customers + 1),
                                  size=min(n_rest, n_customers),
                                  replace=False).tolist())
    return BreakRegime(
        name=regime, drive_limit_min=drive_limit,
        break_duration_min=break_dur,
        rest_node_ids=tuple(int(x) for x in rest_ids),
    )


def _build_chargers(num: int, n_customers: int,
                    rng: np.random.Generator) -> list[Charger] | None:
    if num <= 0:
        return None
    ids = sorted(rng.choice(np.arange(1, n_customers + 1),
                            size=min(num, n_customers), replace=False).tolist())
    return [Charger(node_id=int(i), rate=float(rng.uniform(3.0, 8.0)),
                    types=("ev",)) for i in ids]


def _build_dense_matrices(pts: np.ndarray, asymmetry: float,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_minutes, D_miles) as (N+1, N+1) float32 dense arrays.

    Euclidean distance scaled to ~5 mi/unit; travel time at ~25 mph =
    distance/25 hr = distance * 2.4 min/mi.  Per-edge multiplicative
    noise produces asymmetry.
    """
    # Vectorised pairwise Euclidean.
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diff * diff).sum(-1)).astype(np.float32)
    D *= 0.5   # half-mile per unit coord
    # Travel time in minutes at ~25 mph = 2.4 min/mi.
    T = D * 2.4
    # Per-edge multiplicative noise → asymmetry.
    if asymmetry > 0:
        noise = rng.uniform(1.0 - asymmetry, 1.0 + asymmetry,
                            size=T.shape).astype(np.float32)
        T = T * noise
    np.fill_diagonal(T, 0.0)
    np.fill_diagonal(D, 0.0)
    return T, D


def _build_sparse_neighbors(pts: np.ndarray, k: int) -> dict:
    """Compute k-nearest-neighbours per point WITHOUT materialising the
    full (N, N) distance matrix.  Returns a dict
    {'indices': (N, k) int32, 'dists_min': (N, k) float32} usable by
    sparse-attention encoders.

    Uses scipy.spatial.cKDTree for O(N log N) queries; falls back to a
    chunked brute-force search if scipy unavailable.
    """
    n = pts.shape[0]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        d, idx = tree.query(pts, k=min(k + 1, n))
        # Drop self (column 0)
        return {"indices": idx[:, 1:].astype(np.int32),
                "dists_min": (d[:, 1:] * 0.5).astype(np.float32)}
    except Exception:
        # Chunked brute force fallback
        chunk = max(1, min(n, 10_000_000 // max(1, n)))
        indices = np.zeros((n, k), dtype=np.int32)
        dists = np.zeros((n, k), dtype=np.float32)
        for i_start in range(0, n, chunk):
            i_end = min(n, i_start + chunk)
            block = pts[i_start:i_end][:, None, :] - pts[None, :, :]
            d_block = np.sqrt((block * block).sum(-1))
            # Exclude self
            for r in range(i_end - i_start):
                d_block[r, i_start + r] = np.inf
            part_idx = np.argpartition(d_block, k, axis=1)[:, :k]
            for r in range(i_end - i_start):
                pi = part_idx[r]
                order = np.argsort(d_block[r, pi])
                indices[i_start + r] = pi[order]
                dists[i_start + r] = d_block[r, pi[order]] * 0.5
        return {"indices": indices, "dists_min": dists}


def generate(N: int, seed: int = 0, *,
             asymmetry: float = 0.05,
             demand_range: tuple[int, int] = (1, 10),
             tw_tightness: float = 0.3,
             coord_range: float = 100.0,
             vehicle_factor: float = 0.30,
             capacity_buffer: float = 1.40,
             include_matrix: bool = True,
             sparse_k: int | None = None,
             hetero_fleet: bool = False,
             breaks_regime: str = "none",
             num_chargers: int = 0) -> Instance:
    """Generate one synthetic Instance at the requested size.

    Memory:
      - dense matrix (T + D): O((N+1)^2) float32 = 8 (N+1)² bytes.
      - sparse mode: O(N * sparse_k) — usable up to N=10,000 in <100 MB.

    `include_matrix=False` skips the dense matrix entirely (only sparse
    or external matrix-attached usage).  In that case the loader's
    travel_time/travel_dist are (1, 1) zero arrays — placeholders.
    """
    rng = np.random.default_rng(seed)
    n = N + 1   # +1 for depot
    pts = rng.uniform(0.0, coord_range, size=(n, 2)).astype(np.float32)
    pts[0] = np.array([coord_range / 2, coord_range / 2], dtype=np.float32)  # depot center

    if include_matrix:
        T, D = _build_dense_matrices(pts, asymmetry, rng)
    else:
        T = np.zeros((1, 1), dtype=np.float32)
        D = np.zeros((1, 1), dtype=np.float32)

    sparse_extra = None
    if sparse_k is not None and sparse_k > 0 and N >= sparse_k:
        sparse_extra = _build_sparse_neighbors(pts, sparse_k)

    # Day window 480-960 (default settings).
    day_start, day_end = 480, 960
    dem_lo, dem_hi = demand_range

    # TW per customer: width = tw_tightness * (day_end - day_start) ± jitter,
    # centered uniformly in the feasible window.
    width_med = tw_tightness * (day_end - day_start)
    customers: list[Customer] = []
    total_demand = 0
    for i in range(1, n):
        width = float(rng.uniform(max(20.0, width_med * 0.6), width_med * 1.4))
        if include_matrix:
            depot_to = float(T[0, i])
            depot_back = float(T[i, 0])
        else:
            # Approximate from coordinates if matrix omitted.
            d_to = float(np.linalg.norm(pts[i] - pts[0])) * 0.5 * 2.4
            depot_to = depot_back = d_to * float(rng.uniform(1.0 - asymmetry, 1.0 + asymmetry))
        service = int(rng.integers(5, 16))
        earliest = day_start + depot_to
        latest = day_end - depot_back - service
        if earliest >= latest:
            # Infeasible solo; widen day or skip
            ready = day_start
            due = day_end
        else:
            center = float(rng.uniform(earliest + width / 2, max(earliest + width / 2, latest - width / 2)))
            ready = int(max(day_start, center - width / 2))
            due = int(min(day_end, center + width / 2))
        demand = int(rng.integers(dem_lo, dem_hi + 1))
        total_demand += demand
        customers.append(Customer(
            id=i, node_id=i, x=float(pts[i, 0]), y=float(pts[i, 1]),
            demand=demand, ready=ready, due=due, service=service,
        ))

    num_vehicles = max(2, int(math.ceil(N * vehicle_factor)))
    capacity = int(math.ceil(capacity_buffer * total_demand / num_vehicles))

    # Asymmetry score (only if we have the matrix).
    if include_matrix and N >= 2:
        iu = np.triu_indices(n, k=1)
        a, b = T[iu], T.T[iu]
        denom = np.maximum(np.maximum(a, b), 1e-6)
        asym_score = float(np.mean(np.abs(a - b) / denom))
    else:
        asym_score = float(asymmetry)

    inst = Instance(
        instance_id=f"SYNTH-N{N:05d}-S{seed}",
        city="Synthetic",
        num_customers=N,
        num_vehicles=num_vehicles,
        vehicle_capacity=capacity,
        depot=Depot(node_id=0, x=float(pts[0, 0]), y=float(pts[0, 1]),
                    ready=day_start, due=day_end),
        customers=customers,
        travel_time=T, travel_dist=D,
        asymmetry_score=asym_score,
        seed=seed,
        schema_version="1.0",
    )
    # Attach sparse neighbour structure as a side-channel attribute if asked.
    if sparse_extra is not None:
        inst.__dict__["_sparse_neighbors"] = sparse_extra

    # SPEC-0-INST-03 enrichments — all optional, all leave legacy solvers untouched.
    if hetero_fleet:
        include_ev = num_chargers > 0
        inst.vehicles = _build_vehicle_fleet(num_vehicles, rng, include_ev=include_ev)
    if breaks_regime and breaks_regime != "none":
        inst.breaks = _build_breaks(breaks_regime, N, rng)
    if num_chargers > 0:
        inst.chargers = _build_chargers(num_chargers, N, rng)
    return inst


def _peak_rss_mb() -> float:
    """Best-effort current RSS in MB (Windows + POSIX)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return -1.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--asym", type=float, default=0.05)
    p.add_argument("--tw-tightness", type=float, default=0.3)
    p.add_argument("--sparse-k", type=int, default=None)
    p.add_argument("--no-matrix", action="store_true",
                   help="Skip dense matrix — only sparse + coord features.")
    p.add_argument("--hetero-fleet", action="store_true",
                   help="Heterogeneous fleet (van/box/tractor; EV when --chargers).")
    p.add_argument("--breaks", choices=["none", "EU561", "FMCSA"], default="none")
    p.add_argument("--chargers", type=int, default=0,
                   help="Number of charger stations (enables EV vehicle class).")
    p.add_argument("--out", default=None, help="Optional output dir to save instance.")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    inst = generate(N=args.N, seed=args.seed, asymmetry=args.asym,
                    tw_tightness=args.tw_tightness,
                    include_matrix=not args.no_matrix,
                    sparse_k=args.sparse_k,
                    hetero_fleet=args.hetero_fleet,
                    breaks_regime=args.breaks,
                    num_chargers=args.chargers)
    elapsed = time.perf_counter() - t0
    rss = _peak_rss_mb()
    has_sparse = "_sparse_neighbors" in inst.__dict__
    matrix_bytes = inst.travel_time.nbytes + inst.travel_dist.nbytes
    sparse_bytes = 0
    if has_sparse:
        s = inst.__dict__["_sparse_neighbors"]
        sparse_bytes = s["indices"].nbytes + s["dists_min"].nbytes
    print(f"{inst.instance_id}: N={inst.num_customers}, K={inst.num_vehicles}, "
          f"asym={inst.asymmetry_score:.3f}, "
          f"matrix={matrix_bytes / 1e6:.1f} MB, sparse={sparse_bytes / 1e6:.1f} MB, "
          f"build_time={elapsed:.2f}s, rss={rss:.1f} MB")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        save_instance(inst, out)
        print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
