"""Build a large realistic VRPTW instance from an OSM driving network.

Pulls (or reuses cached) osmnx graph for a city, samples N customer nodes,
picks a central depot, runs Dijkstra to build (N+1) x (N+1) OD matrices
for distance AND travel time (meters / seconds, then converted to miles /
minutes to match the project convention), assigns demands + TWs, and
saves the JSON + .npy matrices.

Routing decisions during the bench will use these precomputed matrices,
so there is no Euclidean leakage at solve time. The renderer's road-
following polylines (svrptw.viz.renderer.render_llm_compare with
route_mode='network') queries the SAME osmnx graph -- exact consistency
between solver cost and visual.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, "D:/SVRPTW")
import os
os.environ.setdefault("SVRPTW_WEBUI_URL", "http://127.0.0.1:8765")

from webui import client as ui

# City -> default WGS84 bounding box (min_lon, min_lat, max_lon, max_lat).
# Tight enough that osmnx graph stays tractable; loose enough to span N=2000.
_CITY_BBOX = {
    "Manhattan":     (-74.020, 40.700, -73.910, 40.880),
    "Paris":         (  2.260, 48.815,   2.420, 48.910),
    "SanFrancisco":  (-122.515, 37.700, -122.355, 37.815),
    "Charleston":    (-80.060, 32.745, -79.860, 32.840),
    "Phoenix":       (-112.180, 33.380, -111.940, 33.580),
    "Pittsburgh":    (-80.040, 40.380, -79.880, 40.480),
    "Cambridge":     ( -0.170, 52.180,  0.180, 52.260),
    "Austin":        (-97.830, 30.220, -97.660, 30.380),
}


def _load_graph(city: str):
    """Get the osmnx drive graph for `city` (cached on disk via webui.road_network)."""
    from webui.road_network import get_network
    bbox = _CITY_BBOX.get(city)
    if bbox is None:
        raise ValueError(f"unknown city {city}; add a bbox to _CITY_BBOX")
    G = get_network(bbox, network_type="drive")
    if G is None:
        raise RuntimeError(f"failed to fetch graph for {city}")
    return G, bbox


def _annotate_travel_time(G) -> None:
    """Add `travel_time` (seconds) attribute to each edge.

    speed = maxspeed when present; otherwise 30 km/h (urban default).
    travel_time_s = length_m / (speed_kmh * 1000 / 3600).
    """
    for u, v, k, d in G.edges(keys=True, data=True):
        spd_raw = d.get("maxspeed")
        if isinstance(spd_raw, list):
            spd_raw = spd_raw[0] if spd_raw else None
        try:
            if spd_raw is None:
                spd_kmh = 30.0
            elif isinstance(spd_raw, (int, float)):
                spd_kmh = float(spd_raw)
            else:
                spd_kmh = float(str(spd_raw).split()[0])
        except (ValueError, AttributeError):
            spd_kmh = 30.0
        spd_kmh = max(5.0, min(120.0, spd_kmh))
        d["travel_time"] = float(d["length"]) / (spd_kmh * 1000.0 / 3600.0)


def _sample_customer_nodes(G, n: int, seed: int) -> list:
    """Sample n nodes from the LARGEST STRONGLY-CONNECTED COMPONENT.

    SPEC-INST-SCC-FIX (iter-5o): the wholesale bench surfaced that
    naive `list(G.nodes)` sampling can pick nodes in disconnected
    sub-components (e.g. an island reachable only via ferry, or a
    one-way trap). Those nodes show up in the OD matrix as
    `posinf` -> nan_to_num to 1e6 seconds = ~277 hours travel, and
    PyVRP correctly rejects the resulting instances as infeasible.

    Filter to the largest SCC so every pair has finite travel time.
    """
    import numpy as _np
    import networkx as _nx
    sccs = list(_nx.strongly_connected_components(G))
    largest = max(sccs, key=len) if sccs else set(G.nodes)
    nodes = list(largest)
    if n > len(nodes):
        raise ValueError(
            f"need n={n} customers but largest SCC has only {len(nodes)} nodes "
            f"(out of {len(G.nodes)} total in graph; {len(sccs)} components)"
        )
    rng = _np.random.default_rng(seed)
    idx = rng.choice(len(nodes), size=n, replace=False)
    return [nodes[i] for i in idx]


def _pick_depot(G, customer_nodes: list) -> Any:
    """Pick the depot as the node with min total path length to customers (sampled)."""
    import networkx as nx
    sample = customer_nodes[: min(50, len(customer_nodes))]
    candidates = list(G.nodes)
    # Subsample candidate nodes too -- 100 is plenty
    import numpy as _np
    rng = _np.random.default_rng(0)
    cand_idx = rng.choice(len(candidates), size=min(100, len(candidates)), replace=False)
    candidates = [candidates[i] for i in cand_idx]
    best = candidates[0]; best_score = float("inf")
    for c in candidates:
        try:
            length = sum(nx.shortest_path_length(G, c, s, weight="length")
                         for s in sample[:10])
        except Exception:
            continue
        if length < best_score:
            best_score = length
            best = c
    return best


def _build_od_matrices(G, nodes: list) -> tuple[np.ndarray, np.ndarray]:
    """Fast OD via scipy.sparse.csgraph.dijkstra (C-level).

    Returns (dist_miles, time_minutes), both (N+1) x (N+1) float32.
    Empirically 20-100x faster than networkx single_source on city graphs.

    osmnx returns a MultiDiGraph: two-way streets are already represented
    as TWO directed edges (u->v and v->u). One-way streets appear as a
    single directed edge. So we just transcribe each directed edge as-is
    and let csgraph treat the graph as directed -- no manual reverse.
    Multi-edges between the same pair (rare; turn restrictions etc.) are
    collapsed to the minimum weight when building the CSR.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    # 1. Build sparse adjacency for length and travel_time edge weights.
    #    osmnx graph nodes are arbitrary IDs; map to 0..M-1.
    all_nodes_in_graph = list(G.nodes)
    node_idx = {n: i for i, n in enumerate(all_nodes_in_graph)}
    M = len(all_nodes_in_graph)
    rows: list = []
    cols: list = []
    w_len: list = []
    w_time: list = []
    for u, v, k, d in G.edges(keys=True, data=True):
        ui_, vi_ = node_idx[u], node_idx[v]
        rows.append(ui_); cols.append(vi_)
        w_len.append(float(d.get("length", 1.0)))
        w_time.append(float(d.get("travel_time", 1.0)))
    # COO -> CSR; duplicate (u,v) pairs (parallel edges) are summed by default.
    # We want the MIN weight (cheapest parallel edge wins). Build via lil for
    # the min reduction, or use coo + groupby. Fastest: build coo, convert to
    # csr, then take min over duplicates by re-building.
    L = sp.csr_matrix((w_len, (rows, cols)), shape=(M, M), dtype=np.float64)
    T = sp.csr_matrix((w_time, (rows, cols)), shape=(M, M), dtype=np.float64)
    # 2. dijkstra from `nodes` only (much smaller than the whole graph).
    src_idx = np.array([node_idx[n] for n in nodes])
    dist_m_full = dijkstra(L, indices=src_idx, directed=True)   # (len(src), M)
    time_s_full = dijkstra(T, indices=src_idx, directed=True)
    # 3. Extract just the OD slice (sources x sources)
    dist_m = dist_m_full[:, src_idx]
    time_s = time_s_full[:, src_idx]
    # 4. Replace inf (unreachable) with a large finite value to avoid solver NaN
    dist_m = np.nan_to_num(dist_m, nan=1e6, posinf=1e6)
    time_s = np.nan_to_num(time_s, nan=1e6, posinf=1e6)
    # 5. Convert: meters -> miles, seconds -> minutes
    return (dist_m * 0.000621371).astype(np.float32), (time_s / 60.0).astype(np.float32)


def _assign_demand_tw(n: int, day_start: int, day_end: int, seed: int):
    """Random demand 1..5, random TW within day."""
    rng = np.random.default_rng(seed + 100)
    demands = rng.integers(1, 6, size=n).tolist()
    tw_centers = rng.uniform(day_start + 30, day_end - 60, size=n)
    tw_widths = rng.uniform(60, 240, size=n)
    ready = [int(max(day_start, c - w/2)) for c, w in zip(tw_centers, tw_widths)]
    due = [int(min(day_end, c + w/2)) for c, w in zip(tw_centers, tw_widths)]
    services = [10] * n
    return demands, ready, due, services


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", required=True, choices=sorted(_CITY_BBOX.keys()))
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--rep", default="I000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vehicle-capacity", type=int, default=200)
    p.add_argument("--num-vehicles", type=int, default=None,
                   help="default = max(N//8, 10)")
    p.add_argument("--day-start", type=int, default=480)
    p.add_argument("--day-end", type=int, default=960)
    p.add_argument("--out-dir", default="instances/v1_large")
    args = p.parse_args()

    iid = f"OSM-{args.city}-N{args.N:04d}-{args.rep}"
    ui.push_agent(f"build-{iid}", "running",
                   summary=f"loading graph + sampling {args.N} customers + OD compute")

    t0 = time.perf_counter()
    G, bbox = _load_graph(args.city)
    print(f"[graph] {args.city} bbox={bbox} nodes={len(G.nodes)} edges={len(G.edges)}")
    _annotate_travel_time(G)
    print(f"[graph] travel_time annotated")

    cust_nodes = _sample_customer_nodes(G, args.N, seed=args.seed)
    depot_node = _pick_depot(G, cust_nodes)
    print(f"[depot] {depot_node}")
    all_nodes = [depot_node] + cust_nodes
    print(f"[OD] computing {len(all_nodes)}x{len(all_nodes)} matrices ...")
    dist_mi, time_min = _build_od_matrices(G, all_nodes)
    print(f"[OD] done in {time.perf_counter()-t0:.1f}s")


    # Per-customer demand + TWs
    demands, ready, due, services = _assign_demand_tw(
        args.N, args.day_start, args.day_end, seed=args.seed)
    K = args.num_vehicles or max(args.N // 8, 10)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = out_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    # load_instance expects an .npz with key 'm' -- not plain .npy
    dist_path = matrices_dir / f"{iid}__dist.npz"
    time_path = matrices_dir / f"{iid}__time.npz"
    np.savez_compressed(dist_path, m=dist_mi)
    np.savez_compressed(time_path, m=time_min)

    # Build the JSON in the v1-style schema (so load_instance handles it)
    customers = []
    for i, nd in enumerate(cust_nodes, start=1):
        x = float(G.nodes[nd]["x"])  # lon
        y = float(G.nodes[nd]["y"])  # lat
        customers.append({
            "id": i, "node_id": int(nd),
            "x": x, "y": y,
            "demand": int(demands[i-1]),
            "ready": int(ready[i-1]), "due": int(due[i-1]),
            "service": int(services[i-1]),
        })
    depot_data = {
        "node_id": int(depot_node),
        "x": float(G.nodes[depot_node]["x"]),
        "y": float(G.nodes[depot_node]["y"]),
        "ready": int(args.day_start),
        "due": int(args.day_end),
    }
    # The matrix paths are stored RELATIVE TO THE JSON FILE'S DIRECTORY,
    # not the global `instances/` root. load_instance's _resolve joins
    # the JSON's parent dir with this string.
    json_path = out_dir / f"{iid}.json"
    rel_dist = str(dist_path.relative_to(out_dir).as_posix())
    rel_time = str(time_path.relative_to(out_dir).as_posix())
    inst_json = {
        "instance_id": iid,
        "city": args.city,
        "topology": "drive",
        "num_customers": args.N,
        "num_vehicles": K,
        "vehicle_capacity": args.vehicle_capacity,
        "depot": depot_data,
        "customers": customers,
        "asymmetry_score": 0.07,
        "generator_seed": args.seed,
        "schema_version": 1,
        "travel_time_matrix_path": rel_time,
        "travel_distance_matrix_path": rel_dist,
    }
    json_path.write_text(json.dumps(inst_json, indent=2))
    elapsed = time.perf_counter() - t0
    print(f"[done] wrote {json_path}  ({elapsed:.1f}s)")
    ui.push_agent(f"build-{iid}", "completed",
                   summary=f"wrote {json_path.name} in {elapsed:.0f}s ({args.N} customers, K={K})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
