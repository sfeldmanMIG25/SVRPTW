"""SPEC-4-DATA-02 — network-OD synthetic generator.

Builds a directed grid-with-arterials road graph, runs Dijkstra to get
shortest-path T (time) and D (distance) matrices, then samples customer
locations as graph nodes. The asymmetry is *structured* (one-way
streets) rather than i.i.d. noise on Euclidean distances.

Minimal viable first pass:
- grid-with-arterials topology only (other topologies queued).
- Dense matrix output (sparse comes later for N=10k).
- Drop-in API compatible with `svrptw.io.Instance`.

Why this exists: POMO v3 trained on the Euclidean+noise synthetic
generator transfers worse to v1 OSM than POMO trained on v1 OSM
directly. The hypothesis is that road-network geometry (hub-and-spoke,
one-ways, structured asymmetry) isn't captured by Euclidean+i.i.d.-
noise. See `bench/figures/pomo_v3_vs_v4.md` for the failed transfer
story and `specs/SPEC-4-DATA-02-od-network-synthetic.md` for the spec.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from svrptw.io import Customer, Depot, Instance


def _build_grid_arterials(
    grid_size: int,
    one_way_frac: float,
    arterial_speed_mult: float,
    rng: np.random.Generator,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Build a directed grid graph with one-way streets and arterials.

    Returns:
        (adj_time, adj_dist) sparse adjacency matrices, and
        (coords) array of (grid_size**2, 2) node coordinates.

    Time and distance differ because arterials have higher speed (lower
    travel-time per unit distance) — modelling traffic-light vs freeway.
    """
    n_nodes = grid_size * grid_size
    coords = np.zeros((n_nodes, 2), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            coords[r * grid_size + c] = [c, r]

    # Build edges (4-neighbour grid), then maybe one-way them.
    edges_time: list[tuple[int, int, float]] = []
    edges_dist: list[tuple[int, int, float]] = []

    block_distance = 1.0
    block_time = 3.0  # ~3 minutes per block — realistic city street pace

    # Pre-pick arterial rows/columns: every 4th row and column is a fast arterial.
    arterial_rows = set(range(0, grid_size, 4))
    arterial_cols = set(range(0, grid_size, 4))

    for r in range(grid_size):
        for c in range(grid_size):
            u = r * grid_size + c
            # right neighbour
            if c + 1 < grid_size:
                v = u + 1
                is_arterial = (r in arterial_rows)
                d = block_distance
                t = block_time / (arterial_speed_mult if is_arterial else 1.0)
                # Possibly make one-way (forward only or reverse only).
                if rng.random() < one_way_frac:
                    if rng.random() < 0.5:
                        edges_time.append((u, v, t))
                        edges_dist.append((u, v, d))
                    else:
                        edges_time.append((v, u, t))
                        edges_dist.append((v, u, d))
                else:
                    edges_time.extend([(u, v, t), (v, u, t)])
                    edges_dist.extend([(u, v, d), (v, u, d)])
            # down neighbour
            if r + 1 < grid_size:
                v = u + grid_size
                is_arterial = (c in arterial_cols)
                d = block_distance
                t = block_time / (arterial_speed_mult if is_arterial else 1.0)
                if rng.random() < one_way_frac:
                    if rng.random() < 0.5:
                        edges_time.append((u, v, t))
                        edges_dist.append((u, v, d))
                    else:
                        edges_time.append((v, u, t))
                        edges_dist.append((v, u, d))
                else:
                    edges_time.extend([(u, v, t), (v, u, t)])
                    edges_dist.extend([(u, v, d), (v, u, d)])

    # Convert to CSR sparse matrices.
    rows_t = np.array([e[0] for e in edges_time])
    cols_t = np.array([e[1] for e in edges_time])
    data_t = np.array([e[2] for e in edges_time], dtype=np.float32)
    rows_d = np.array([e[0] for e in edges_dist])
    cols_d = np.array([e[1] for e in edges_dist])
    data_d = np.array([e[2] for e in edges_dist], dtype=np.float32)

    adj_time = csr_matrix((data_t, (rows_t, cols_t)), shape=(n_nodes, n_nodes))
    adj_dist = csr_matrix((data_d, (rows_d, cols_d)), shape=(n_nodes, n_nodes))
    return adj_time, adj_dist, coords


def generate(
    N: int,
    seed: int = 0,
    *,
    grid_size: int = 20,
    one_way_frac: float = 0.15,
    arterial_speed_mult: float = 1.8,
    demand_range: tuple[int, int] = (1, 10),
    tw_tightness: float = 0.3,
    vehicle_factor: float = 0.30,
    capacity_buffer: float = 1.40,
) -> Instance:
    """Generate a network-OD synthetic instance.

    `grid_size` controls the underlying road graph. The actual number
    of customers sampled is `N`; the depot is the central node.

    Asymmetry is structural (one-way streets + arterial speed) rather
    than i.i.d. noise. Expected asymmetry score: 0.04-0.08 at
    one_way_frac=0.15 — matches the v1 OSM Manhattan distribution.
    """
    rng = np.random.default_rng(seed)
    if grid_size * grid_size < N + 1:
        raise ValueError(f"grid_size={grid_size} too small for N={N}; need at least sqrt(N+1)")

    adj_time, adj_dist, coords = _build_grid_arterials(
        grid_size, one_way_frac, arterial_speed_mult, rng,
    )

    # Pick the central node as depot, then sample N customer nodes
    # uniformly without replacement from the rest.
    n_nodes = grid_size * grid_size
    depot_node = (grid_size // 2) * grid_size + (grid_size // 2)
    other_nodes = np.array([i for i in range(n_nodes) if i != depot_node])
    customer_nodes = rng.choice(other_nodes, size=N, replace=False)

    # Used nodes (depot + customers) in instance-index order.
    used_nodes = np.concatenate([[depot_node], customer_nodes])
    n_used = len(used_nodes)  # = N + 1

    # All-pairs shortest paths from each used node to every other.
    # scipy's dijkstra with `indices=used_nodes` returns (n_used, n_nodes).
    T_full, _ = dijkstra(
        csgraph=adj_time, directed=True, indices=used_nodes,
        return_predecessors=True,
    )
    D_full, _ = dijkstra(
        csgraph=adj_dist, directed=True, indices=used_nodes,
        return_predecessors=True,
    )
    # Slice to the (n_used, n_used) sub-matrix.
    T = T_full[:, used_nodes].astype(np.float32)
    D = D_full[:, used_nodes].astype(np.float32)

    # Sanity check: no inf (graph should be strongly connected enough).
    if np.isinf(T).any():
        # Some nodes unreachable. Fall back: replace inf with 3× max finite.
        finite_max = float(T[np.isfinite(T)].max())
        T = np.where(np.isinf(T), finite_max * 3, T).astype(np.float32)
        D = np.where(np.isinf(D), float(D[np.isfinite(D)].max()) * 3, D).astype(np.float32)

    # Customers + depot.
    day_start, day_end = 480, 960
    width_med = tw_tightness * (day_end - day_start)
    customers: list[Customer] = []
    total_demand = 0
    dem_lo, dem_hi = demand_range
    for i in range(1, n_used):
        depot_to = float(T[0, i])
        depot_back = float(T[i, 0])
        service = int(rng.integers(5, 16))
        earliest = day_start + depot_to
        latest = day_end - depot_back - service
        width = float(rng.uniform(max(20.0, width_med * 0.6), width_med * 1.4))
        if earliest >= latest:
            ready = day_start
            due = day_end
        else:
            center = float(rng.uniform(earliest + width / 2,
                                        max(earliest + width / 2, latest - width / 2)))
            ready = int(max(day_start, center - width / 2))
            due = int(min(day_end, center + width / 2))
        demand = int(rng.integers(dem_lo, dem_hi + 1))
        total_demand += demand
        node = int(used_nodes[i])
        x = float(coords[node, 0])
        y = float(coords[node, 1])
        customers.append(Customer(
            id=i, node_id=node, x=x, y=y,
            demand=demand, ready=ready, due=due, service=service,
        ))

    num_vehicles = max(2, int(math.ceil(N * vehicle_factor)))
    capacity = int(math.ceil(capacity_buffer * total_demand / num_vehicles))

    iu = np.triu_indices(n_used, k=1)
    a, b = T[iu], T.T[iu]
    denom = np.maximum(np.maximum(a, b), 1e-6)
    asym_score = float(np.mean(np.abs(a - b) / denom))

    dep_node = int(used_nodes[0])
    depot = Depot(
        node_id=dep_node, x=float(coords[dep_node, 0]), y=float(coords[dep_node, 1]),
        ready=day_start, due=day_end,
    )
    inst = Instance(
        instance_id=f"NET-N{N:05d}-S{seed}",
        city="grid-arterials",
        num_customers=N,
        num_vehicles=num_vehicles,
        vehicle_capacity=capacity,
        depot=depot,
        customers=customers,
        travel_time=T,
        travel_dist=D,
        asymmetry_score=asym_score,
        seed=seed,
    )
    return inst
