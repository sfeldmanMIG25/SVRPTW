"""OD construction + per-leg geometry reconstruction (SPEC-OPENVRP-06 §3, §4).

The single most important contract here is D14: the OD the solver
optimizes over and the geometry it reports come from the same graph.
``compute_od`` returns time/distance matrices PLUS the predecessor
arrays from scipy.sparse.csgraph.dijkstra; ``trace_path`` reconstructs
the node-by-node shortest path for a single (a→b) pair in O(path length).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openvrp.errors import MissingExtra
from openvrp.network.ingest import LoadedNetwork


@dataclass
class ODBundle:
    """OD matrices + per-source predecessor arrays for traceback.

    Cached lookups (built once at compute_od time) make trace_path O(path
    length) instead of re-scanning the predecessor table per call.
    """
    time_seconds: np.ndarray    # (k, k) — square over k sources
    distance_meters: np.ndarray
    index_of: dict[str, int]    # stop_or_depot_id -> matrix index 0..k-1
    snapped_node: dict[str, int]
    predecessors: np.ndarray    # (unique_sources, |V|) — from scipy.sparse.csgraph.dijkstra
    node_list: list[int]        # all graph node ids (the |V| order)
    node_xy: dict[int, tuple[float, float]]
    # Cached for trace_path (built by ``compute_od``).
    node_index_of: dict[int, int] = field(default_factory=dict)
    src_row_of_node: dict[int, int] = field(default_factory=dict)


def compute_od(loaded: LoadedNetwork,
               snapped: dict[str, int]) -> ODBundle:
    """Build (time, distance) OD over the snapped sources.

    Returns predecessors arrays so geometry reconstruction is a simple
    traceback per leg.
    """
    try:
        import networkx as nx
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
    except ImportError as e:
        raise MissingExtra("network", reason=str(e)) from e

    # Build node-ordering and CSR for edge travel time
    g: nx.DiGraph = loaded.graph
    nodes = list(g.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows: list[int] = []
    cols: list[int] = []
    t_data: list[float] = []
    d_data: list[float] = []
    for u, v, data in g.edges(data=True):
        rows.append(node_idx[u])
        cols.append(node_idx[v])
        t_data.append(float(data.get("travel_time", data.get("length", 1.0))))
        d_data.append(float(data.get("length", 1.0)))
    T_sparse = csr_matrix((t_data, (rows, cols)), shape=(n, n))
    D_sparse = csr_matrix((d_data, (rows, cols)), shape=(n, n))

    # Sources = unique snapped node indices
    source_ids = list(snapped.values())
    source_node_indices = [node_idx[s] for s in source_ids]
    # Map stop_id -> matrix index (the snap order, deduplicated below)
    # If two stops snap to the same node, both share the same OD row.
    unique_source_indices: list[int] = []
    seen: dict[int, int] = {}
    for ni in source_node_indices:
        if ni not in seen:
            seen[ni] = len(unique_source_indices)
            unique_source_indices.append(ni)
    # Dijkstra (with predecessors) from each unique source
    t_mat, t_preds = dijkstra(T_sparse, directed=True,
                              indices=unique_source_indices,
                              return_predecessors=True)
    d_mat, _ = dijkstra(D_sparse, directed=True,
                        indices=unique_source_indices,
                        return_predecessors=False)
    # Vectorized slicing into the (k, k) result matrix — replaces a
    # ~k^2 Python loop (sec-scale at k=1000).
    sid_list = list(snapped.keys())
    row_idx: np.ndarray = np.array([seen[node_idx[snapped[s]]] for s in sid_list], dtype=np.int64)
    col_idx: np.ndarray = np.array([node_idx[snapped[s]] for s in sid_list], dtype=np.int64)
    out_t = t_mat[row_idx][:, col_idx].astype(np.float64, copy=False)
    out_d = d_mat[row_idx][:, col_idx].astype(np.float64, copy=False)
    # Replace inf (disconnected) with sentinel — we'll surface in
    # geometry_failures; keep the inf so evaluators mark them as missed.
    index_of = {sid: i for i, sid in enumerate(sid_list)}
    # Pre-build trace_path lookups so each call is O(path length).
    node_index_of = {n: i for i, n in enumerate(nodes)}
    src_row_of_node = {snapped[s]: seen[node_idx[snapped[s]]] for s in sid_list}
    return ODBundle(
        time_seconds=out_t,
        distance_meters=out_d,
        index_of=index_of,
        snapped_node=dict(snapped),
        predecessors=np.asarray(t_preds, dtype=np.int64),
        node_list=nodes,
        node_xy=dict(loaded.node_xy),
        node_index_of=node_index_of,
        src_row_of_node=src_row_of_node,
    )


def trace_path(bundle: ODBundle, src_id: str, dst_id: str) -> list[tuple[float, float]] | None:
    """Reconstruct the node-by-node polyline of the shortest path from
    ``src_id`` to ``dst_id`` (each by snapped stop id).

    Returns the ordered ``[(lon, lat), ...]`` polyline, or ``None`` if
    the predecessor chain is broken (disconnected). O(path length) per
    call thanks to the cached node_index_of / src_row_of_node lookups
    built in ``compute_od``.
    """
    if src_id not in bundle.snapped_node or dst_id not in bundle.snapped_node:
        return None
    src_node = bundle.snapped_node[src_id]
    dst_node = bundle.snapped_node[dst_id]
    if not bundle.node_index_of or not bundle.src_row_of_node:
        # Backwards-compat fallback (older ODBundle without caches).
        node_index = {n: i for i, n in enumerate(bundle.node_list)}
        src_idx = node_index[src_node]; dst_idx = node_index[dst_node]
        nodes = bundle.node_list
        src_row = -1
        for row in range(bundle.predecessors.shape[0]):
            if bundle.predecessors[row, src_idx] == -9999:
                src_row = row
                break
        if src_row < 0:
            return None
    else:
        src_idx = bundle.node_index_of[src_node]
        dst_idx = bundle.node_index_of[dst_node]
        src_row = bundle.src_row_of_node.get(src_node, -1)
        if src_row < 0:
            return None
        nodes = bundle.node_list

    # Walk backward dst -> src using predecessors
    path: list[int] = [dst_idx]
    cur = dst_idx
    cap = bundle.predecessors.shape[1] + 1
    while cur != src_idx:
        nxt = int(bundle.predecessors[src_row, cur])
        if nxt < 0 or nxt == -9999:
            return None
        path.append(nxt)
        cur = nxt
        cap -= 1
        if cap <= 0:
            return None
    path.reverse()
    return [bundle.node_xy[nodes[i]] for i in path if nodes[i] in bundle.node_xy]


def reconstruct_route_geometry(bundle: ODBundle, route_visits: list[Any],
                               from_stop_attr: str = "stop_id"
                               ) -> tuple[list[Any], list[str]]:
    """Build a list of (from_id, to_id, polyline, length_m, duration_s)
    tuples for every consecutive pair in ``route_visits``.

    Returns ``(legs, failures)`` where ``failures`` lists "(a, b)" pairs
    whose predecessor chain was broken (disconnected component).
    """
    legs: list[tuple[str, str, list[tuple[float, float]], float, float]] = []
    failures: list[str] = []
    if not route_visits or bundle is None:
        return legs, failures
    for i in range(len(route_visits) - 1):
        a = getattr(route_visits[i], from_stop_attr, None)
        b = getattr(route_visits[i + 1], from_stop_attr, None)
        if a is None or b is None or a == b:
            continue
        if a not in bundle.snapped_node or b not in bundle.snapped_node:
            continue
        poly = trace_path(bundle, a, b)
        if poly is None or len(poly) < 2:
            failures.append(f"{a}->{b}")
            continue
        ai = bundle.index_of[a]
        bi = bundle.index_of[b]
        legs.append((a, b, poly,
                     float(bundle.distance_meters[ai, bi]),
                     float(bundle.time_seconds[ai, bi])))
    return legs, failures


__all__ = ["ODBundle", "compute_od", "trace_path"]
