"""Graph-aware solution quality (Phase G2).

The Euclidean metrics in `svrptw.metrics.quality` are O(N**2) on intra/
inter pairwise distances, segment crossings, and convex-hull SAT. At
N >= 1000 the bandit's evaluate() loop becomes unworkable (~25-100ms
per call vs ~5-20ms at N=200).

This module provides a drop-in replacement that operates on the OSM
driving graph already cached for road-routing visualisation. Per-route
signals come from O(|graph_edges|) Louvain communities and per-node
betweenness centrality (cached on disk alongside the graph). Cross-route
crossings are estimated by edge-set intersection along shortest paths
between consecutive customers, which is O(|E_route|) per pair.

Public API mirrors the Euclidean version:

    score_solution_graph(inst, sol, *, G=None) -> GraphSolutionQualityScore

If `inst` looks non-geographic (Solomon/Homberger), the call falls back
to the Euclidean metrics suite.
"""
from __future__ import annotations

import hashlib
import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from svrptw.io.instance import Instance
    from svrptw.solvers.common.solution import Solution

_LOG = logging.getLogger("svrptw.metrics.graph_quality")
_CACHE = Path(__file__).resolve().parents[2] / "webui" / "cache" / "networks"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GraphRouteScore:
    route_idx: int
    n_customers: int
    bc_mean: float           # mean betweenness centrality of route's nodes
    bc_std: float            # std (low = tight band = compact route)
    community_purity: float  # frac of customers in dominant Louvain community
    network_length: float    # sum of edge lengths along route's shortest paths


@dataclass
class GraphSolutionQualityScore:
    n_routes: int
    routes: list[GraphRouteScore] = field(default_factory=list)
    mean_bc_std: float = 0.0
    mean_community_purity: float = 0.0
    cross_route_edge_overlap: int = 0
    crossings_via_edge_overlap: int = 0   # legacy-name alias
    quality_index_graph: float = 0.0
    n_customers_served: int = 0
    n_unrouted: int = 0
    served_frac: float = 0.0
    load_util_cv: float = 0.0
    fallback_reason: str = ""             # "" if graph-aware was used


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_geographic(inst: "Instance") -> bool:
    """True iff the instance's customer x/y look like WGS84 coords.
    Mirrors `webui.basemap.is_geographic`; duplicated to avoid the
    webui import for callers that only want the metric.
    """
    try:
        xs = [c.x for c in inst.customers]
        ys = [c.y for c in inst.customers]
    except Exception:
        return False
    if not xs or not ys:
        return False
    if not (-180 <= min(xs) <= max(xs) <= 180):
        return False
    if not (-90 <= min(ys) <= max(ys) <= 90):
        return False
    if (max(xs) - min(xs)) > 5.0 or (max(ys) - min(ys)) > 5.0:
        return False
    return True


def _bbox_hash(bbox: tuple[float, float, float, float]) -> str:
    return hashlib.sha1(
        f"{bbox[0]:.5f},{bbox[1]:.5f},{bbox[2]:.5f},{bbox[3]:.5f}".encode()
    ).hexdigest()[:12]


def _instance_bbox(inst: "Instance",
                   padding_frac: float = 0.05) -> tuple[float, float, float, float]:
    xs = [c.x for c in inst.customers] + [inst.depot.x]
    ys = [c.y for c in inst.customers] + [inst.depot.y]
    pad_x = (max(xs) - min(xs)) * padding_frac
    pad_y = (max(ys) - min(ys)) * padding_frac
    return (min(xs) - pad_x, min(ys) - pad_y,
            max(xs) + pad_x, max(ys) + pad_y)


def _maybe_load_graph(inst: "Instance"):
    """Lazy fetch the OSM driving graph for inst's bbox.
    Returns (G, cache_key) or (None, "") on failure.
    """
    try:
        from webui.basemap import instance_bbox
        from webui.road_network import get_network
    except Exception:
        return None, ""
    try:
        bbox = instance_bbox(inst)
    except Exception:
        return None, ""
    G = get_network(bbox)
    if G is None:
        return None, ""
    return G, _bbox_hash(bbox)


def _aux_cache_path(cache_key: str, suffix: str) -> Path:
    """Side-car cache file beside the graphml. Caller must ensure
    the parent dir exists (we do it once at module init)."""
    return _CACHE / f"drive__{cache_key}.{suffix}.pkl"


# Per-process caches: avoids re-loading pickled BC + community dicts
# on every score call. Keyed by cache_key (bbox hash).
_BC_CACHE: dict = {}
_COMM_CACHE: dict = {}


def _compute_or_load_betweenness(G, cache_key: str) -> dict:
    """Per-node betweenness centrality. Sampled (k=200) for big graphs.
    Cached to disk + in-process by graph cache key.
    """
    import networkx as nx
    if cache_key and cache_key in _BC_CACHE:
        return _BC_CACHE[cache_key]
    aux = _aux_cache_path(cache_key, "bc") if cache_key else None
    if aux is not None and aux.exists():
        try:
            with aux.open("rb") as f:
                bc = pickle.load(f)
            _BC_CACHE[cache_key] = bc
            return bc
        except Exception as e:
            _LOG.debug("bc cache load failed: %s", e)
    try:
        n = G.number_of_nodes()
        k = min(n, 200) if n > 1000 else None
        bc = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)
    except Exception as e:
        _LOG.warning("betweenness_centrality failed: %s", e)
        return {}
    if aux is not None:
        _CACHE.mkdir(parents=True, exist_ok=True)
        try:
            with aux.open("wb") as f:
                pickle.dump(bc, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            _LOG.debug("bc cache write failed: %s", e)
    if cache_key:
        _BC_CACHE[cache_key] = bc
    return bc


def _compute_or_load_communities(G, cache_key: str) -> dict:
    """Louvain communities on the undirected projection of G.
    Returns dict node_id -> community_id. Cached by graph hash.
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    if cache_key and cache_key in _COMM_CACHE:
        return _COMM_CACHE[cache_key]
    aux = _aux_cache_path(cache_key, "louvain") if cache_key else None
    if aux is not None and aux.exists():
        try:
            with aux.open("rb") as f:
                out = pickle.load(f)
            _COMM_CACHE[cache_key] = out
            return out
        except Exception as e:
            _LOG.debug("louvain cache load failed: %s", e)
    try:
        UG = G.to_undirected() if G.is_directed() else G
        if UG.is_multigraph():
            UG = nx.Graph(UG)
        comms = louvain_communities(UG, seed=42)
    except Exception as e:
        _LOG.warning("louvain_communities failed: %s", e)
        return {}
    out: dict = {}
    for cid, members in enumerate(comms):
        for node in members:
            out[node] = cid
    if aux is not None:
        _CACHE.mkdir(parents=True, exist_ok=True)
        try:
            with aux.open("wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            _LOG.debug("louvain cache write failed: %s", e)
    if cache_key:
        _COMM_CACHE[cache_key] = out
    return out


# Per-instance cache of customer-id -> osm-node-id (built once).
_CUST_NODE_CACHE: dict = {}
# Per-instance cache of customer-id -> demand.
_CUST_DEMAND_CACHE: dict = {}
# Per-graph cache of the node set (set(G.nodes)) -- ~12k entries.
_GRAPH_NODESET_CACHE: dict = {}


def _customer_node_ids(inst: "Instance") -> dict[int, int]:
    """Map customer.id (1..N) -> osm node_id, when present.
    Cached per instance.
    """
    cached = _CUST_NODE_CACHE.get(inst.instance_id)
    if cached is not None:
        return cached
    out: dict[int, int] = {}
    for c in inst.customers:
        nid = getattr(c, "node_id", None)
        if nid is None:
            continue
        try:
            out[int(c.id)] = int(nid)
        except Exception:
            continue
    _CUST_NODE_CACHE[inst.instance_id] = out
    return out


# Per-(graph,instance) cache of pairwise (a,b)->(edge_set, length).
# Keyed by (id(G), instance_id, frozenset of customer node_ids + depot).
# Cleared opportunistically by `_pair_cache_key` rebuild.
_PAIR_CACHE: dict = {}


# Cache the (graph_key, instance_id) -> nodes_tuple mapping so we
# don't rebuild the sorted set on every score call.
_PAIR_KEY_CACHE: dict = {}


def _pair_cache_key(graph_key: str, inst: "Instance",
                    cust_to_node: dict[int, int],
                    depot_node: Optional[int]) -> tuple:
    short = (graph_key, inst.instance_id)
    cached = _PAIR_KEY_CACHE.get(short)
    if cached is not None:
        return cached
    nodes = tuple(sorted(set(cust_to_node.values()) | (
        {depot_node} if depot_node is not None else set()
    )))
    full = (graph_key, inst.instance_id, nodes)
    _PAIR_KEY_CACHE[short] = full
    return full


# Per-process graph cache so repeat score calls don't re-load the
# graphml from disk. Keyed by bbox hash. Sized loosely -- one graph
# per city is the expected steady-state.
_GRAPH_CACHE: dict = {}


def _get_cached_graph(inst: "Instance"):
    """Lazily fetch + in-process cache the OSM driving graph for inst.
    Returns (G, cache_key) or (None, '') on failure.
    """
    try:
        from webui.basemap import instance_bbox
    except Exception:
        return None, ""
    try:
        key = _bbox_hash(instance_bbox(inst))
    except Exception:
        return None, ""
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached, key
    G, _ = _maybe_load_graph(inst)
    if G is None:
        return None, key
    _GRAPH_CACHE[key] = G
    return G, key


def _build_pair_table(G, nodes: tuple) -> dict:
    """Single-source Dijkstra from each node in `nodes`, restricted to
    target nodes in `nodes`. Returns dict (a, b) -> (edge_set, length).
    O(K * (V + E log V)) where K = len(nodes); K << V at N <= 1000.
    """
    import networkx as nx
    out: dict = {}
    node_set = set(nodes)
    for a in nodes:
        if a not in G:
            continue
        try:
            lengths, paths = nx.single_source_dijkstra(
                G, a, weight="length"
            )
        except Exception:
            continue
        for b in nodes:
            if b == a or b not in paths:
                continue
            path = paths[b]
            edges: set[tuple[int, int]] = set()
            for u, v in zip(path[:-1], path[1:]):
                key = (u, v) if u <= v else (v, u)
                edges.add(key)
            out[(a, b)] = (edges, float(lengths.get(b, 0.0)))
    return out


def _route_edge_set(G, cust_to_node: dict[int, int],
                    depot_node: Optional[int],
                    customers: list[int],
                    *, pair_table: Optional[dict] = None,
                    ) -> tuple[set[tuple[int, int]], float]:
    """For a route's customer sequence, look up cached edge sets between
    consecutive nodes. Returns (edge_set, total_length).
    """
    edges: set[tuple[int, int]] = set()
    total_len = 0.0
    if not customers or pair_table is None:
        return edges, 0.0
    sequence: list[int] = []
    if depot_node is not None:
        sequence.append(depot_node)
    for cid in customers:
        nid = cust_to_node.get(cid)
        if nid is not None:
            sequence.append(nid)
    if depot_node is not None:
        sequence.append(depot_node)
    for a, b in zip(sequence[:-1], sequence[1:]):
        cached = pair_table.get((a, b))
        if cached is None:
            continue
        e, ln = cached
        edges |= e
        total_len += ln
    return edges, total_len


def _route_load(inst: "Instance", customers: list[int],
                cust_demand: dict[int, int]) -> float:
    return float(sum(cust_demand.get(c, 0) for c in customers))


def _cv(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    if m == 0:
        return 0.0
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) / m


def _fallback_to_euclidean(inst: "Instance", sol: "Solution",
                           reason: str) -> GraphSolutionQualityScore:
    """Wrap the Euclidean SolutionQualityScore in the graph-aware shape.
    Used when the instance is non-geographic OR the OSM graph is missing.
    """
    from svrptw.metrics import score_solution
    qs = score_solution(inst, sol)
    out = GraphSolutionQualityScore(
        n_routes=qs.n_routes,
        routes=[
            GraphRouteScore(
                route_idx=r.route_idx,
                n_customers=r.n_customers,
                bc_mean=0.0,
                bc_std=0.0,
                community_purity=1.0,           # neutral default
                network_length=0.0,
            )
            for r in qs.routes
        ],
        mean_bc_std=0.0,
        mean_community_purity=1.0,
        cross_route_edge_overlap=qs.inter_route_crossings,
        crossings_via_edge_overlap=qs.inter_route_crossings,
        quality_index_graph=qs.quality_index,
        n_customers_served=qs.n_customers_served,
        n_unrouted=qs.n_unrouted,
        served_frac=(qs.n_customers_served / max(1, inst.num_customers)),
        load_util_cv=qs.load_util_cv,
        fallback_reason=reason,
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_solution_graph(inst: "Instance", sol: "Solution",
                         *, G=None) -> GraphSolutionQualityScore:
    """Graph-aware quality score for a (Instance, Solution) pair.

    On Euclidean / Solomon instances (no WGS84 coords), or when the OSM
    graph cannot be loaded, falls back to the Euclidean metrics suite
    so the return shape is stable regardless of instance type.

    Pass ``G`` to skip the lazy graph fetch (useful when the caller
    already holds the graph).
    """
    served_ids = sol.visited_customer_ids() if hasattr(sol, "visited_customer_ids") else set()
    n_served = len(served_ids)
    if not _is_geographic(inst):
        return _fallback_to_euclidean(inst, sol, reason="non_geographic")
    cust_to_node = _customer_node_ids(inst)
    if not cust_to_node:
        return _fallback_to_euclidean(inst, sol, reason="no_node_ids")
    cache_key = ""
    if G is None:
        G, cache_key = _get_cached_graph(inst)
        if G is None:
            return _fallback_to_euclidean(inst, sol, reason="graph_unavailable")
    else:
        # Best-effort cache key from the instance bbox so we can
        # cache betweenness + louvain alongside the graphml.
        try:
            cache_key = _bbox_hash(_instance_bbox(inst))
        except Exception:
            cache_key = ""
        # Stash the externally-supplied graph in the per-process cache
        # so subsequent calls re-use it.
        if cache_key and cache_key not in _GRAPH_CACHE:
            _GRAPH_CACHE[cache_key] = G

    bc = _compute_or_load_betweenness(G, cache_key)
    comms = _compute_or_load_communities(G, cache_key)
    depot_node = getattr(inst.depot, "node_id", None)
    cust_demand = _CUST_DEMAND_CACHE.get(inst.instance_id)
    if cust_demand is None:
        cust_demand = {c.id: c.demand for c in inst.customers}
        _CUST_DEMAND_CACHE[inst.instance_id] = cust_demand
    cap = max(1.0, float(inst.vehicle_capacity))

    # Per-(graph, instance) pair table -- expensive to build on first
    # call, but every subsequent score_solution_graph(inst, *) on the
    # same instance reuses it. Skip when too few node ids to be useful.
    pair_key = _pair_cache_key(cache_key, inst, cust_to_node, depot_node)
    pair_table = _PAIR_CACHE.get(pair_key)
    if pair_table is None:
        nodes_for_table: tuple = pair_key[2]
        pair_table = _build_pair_table(G, nodes_for_table)
        _PAIR_CACHE[pair_key] = pair_table

    # Cache the graph node set per cache_key -- ~12k entries for
    # Manhattan, building once per process avoids repeated O(V) cost.
    G_node_set = _GRAPH_NODESET_CACHE.get(cache_key)
    if G_node_set is None:
        G_node_set = set(G.nodes)
        if cache_key:
            _GRAPH_NODESET_CACHE[cache_key] = G_node_set

    routes_out: list[GraphRouteScore] = []
    edge_sets: list[set[tuple[int, int]]] = []
    bc_stds: list[float] = []
    purities: list[float] = []
    utils: list[float] = []
    bc_get = bc.get
    comms_get = comms.get if comms else None
    cust_to_node_get = cust_to_node.get
    cust_demand_get = cust_demand.get
    sqrt = math.sqrt
    for ridx, route in enumerate(sol.routes):
        custs = route.customers
        if not custs:
            continue
        # Per-node betweenness for this route's stops (skips off-graph).
        nodes: list[int] = []
        bc_sum = 0.0
        bc_n = 0
        bcs_local: list[float] = []
        for c in custs:
            n = cust_to_node_get(c)
            if n is None:
                continue
            nodes.append(n)
            if n in G_node_set:
                v = float(bc_get(n, 0.0))
                bcs_local.append(v)
                bc_sum += v
                bc_n += 1
        if bc_n > 0:
            bc_mean = bc_sum / bc_n
            v2 = 0.0
            for v in bcs_local:
                d = v - bc_mean
                v2 += d * d
            bc_std = sqrt(v2 / bc_n)
        else:
            bc_mean = bc_std = 0.0
        # Community purity: dominant Louvain community fraction.
        purity = 1.0
        if comms_get and nodes:
            counts: dict = {}
            for n in nodes:
                cid = comms_get(n)
                if cid is None:
                    continue
                counts[cid] = counts.get(cid, 0) + 1
            if counts:
                vals = counts.values()
                purity = max(vals) / max(1, sum(vals))
        # Edge set + network length via cached pair-table.
        edges, net_len = _route_edge_set(
            G, cust_to_node, depot_node, custs,
            pair_table=pair_table,
        )
        edge_sets.append(edges)
        routes_out.append(GraphRouteScore(
            route_idx=ridx,
            n_customers=len(route.customers),
            bc_mean=bc_mean,
            bc_std=bc_std,
            community_purity=purity,
            network_length=net_len,
        ))
        bc_stds.append(bc_std)
        purities.append(purity)
        load = 0.0
        for c in custs:
            load += cust_demand_get(c, 0)
        utils.append(min(1.0, load / cap))


    # Cross-route edge overlap via pairwise set intersection.
    overlap = 0
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            overlap += len(edge_sets[i] & edge_sets[j])

    n_routes = len(routes_out)
    mean_bc_std = (sum(bc_stds) / len(bc_stds)) if bc_stds else 0.0
    mean_purity = (sum(purities) / len(purities)) if purities else 1.0
    served_frac = n_served / max(1, inst.num_customers)
    util_cv = _cv(utils)
    # Normalize bc_std to roughly [0, 1]: typical city-graph BC values
    # span 0 .. ~0.05; std rarely exceeds ~0.02. Clamp to be safe.
    bc_norm = max(0.0, min(1.0, mean_bc_std / 0.02))
    overlap_norm = overlap / max(1, n_routes)
    quality_index_graph = min(1.0, (
        0.30 * (1.0 - bc_norm)
        + 0.30 * mean_purity
        + 0.20 * (1.0 / (1.0 + overlap_norm))
        + 0.10 * served_frac
        + 0.10 * max(0.0, 1.0 - min(1.0, util_cv))
    ))

    return GraphSolutionQualityScore(
        n_routes=n_routes,
        routes=routes_out,
        mean_bc_std=mean_bc_std,
        mean_community_purity=mean_purity,
        cross_route_edge_overlap=overlap,
        crossings_via_edge_overlap=overlap,
        quality_index_graph=float(quality_index_graph),
        n_customers_served=n_served,
        n_unrouted=inst.num_customers - n_served,
        served_frac=served_frac,
        load_util_cv=util_cv,
        fallback_reason="",
    )


__all__ = [
    "GraphRouteScore",
    "GraphSolutionQualityScore",
    "score_solution_graph",
]
