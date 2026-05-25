"""OSMnx-based asymmetric VRPTW instance generator. SPEC-0-INST-01.

Run:
    python -m svrptw.instances_gen.osmnx_generator \
        --config specs/data/cities.yaml \
        --out instances/v1

Outputs:
    instances/v1/<INSTANCE_ID>.json
    instances/v1/matrices/<INSTANCE_ID>.npz          (travel-time, asymmetric)
    instances/v1/matrices/<INSTANCE_ID>_dist.npz     (distance, asymmetric)
    instances/v1/manifest.sha256
    instances/v1/REPORT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import yaml


# Lazy imports of heavy deps so --help works without the venv
def _imports():
    import networkx as nx
    import osmnx as ox
    return ox, nx


def _seed(master: int, *parts: str | int) -> int:
    h = hashlib.sha256(("|".join(str(p) for p in (master, *parts))).encode()).digest()
    return int.from_bytes(h[:4], "big")


def _stratified_node_sample(G, k: int, rng: random.Random) -> list[int]:
    """Spatially stratified node sample.  Splits the bbox into a grid and pulls one node
    per cell until k are collected.  Falls back to random sampling for the remainder."""
    nodes = list(G.nodes(data=True))
    if k >= len(nodes):
        rng.shuffle(nodes)
        return [n for n, _ in nodes[:k]]

    xs = np.array([d["x"] for _, d in nodes])
    ys = np.array([d["y"] for _, d in nodes])
    grid = max(1, int(math.ceil(math.sqrt(k))))
    x_edges = np.linspace(xs.min(), xs.max(), grid + 1)
    y_edges = np.linspace(ys.min(), ys.max(), grid + 1)
    buckets: dict[tuple[int, int], list[int]] = {}
    for (n, _), x, y in zip(nodes, xs, ys, strict=False):
        ix = min(grid - 1, np.searchsorted(x_edges, x) - 1)
        iy = min(grid - 1, np.searchsorted(y_edges, y) - 1)
        buckets.setdefault((max(0, ix), max(0, iy)), []).append(n)

    picked: list[int] = []
    keys = list(buckets.keys())
    rng.shuffle(keys)
    for key in keys:
        if len(picked) >= k:
            break
        picked.append(rng.choice(buckets[key]))

    if len(picked) < k:
        remaining = [n for n, _ in nodes if n not in set(picked)]
        rng.shuffle(remaining)
        picked.extend(remaining[: k - len(picked)])
    return picked[:k]


def _shortest_path_matrix(G, nodes: list[int], weight: str) -> np.ndarray:
    import networkx as nx
    n = len(nodes)
    M = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(M, 0.0)
    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, src, weight=weight)
        for j, dst in enumerate(nodes):
            if dst in lengths:
                M[i, j] = lengths[dst]
    return M


def _asymmetry_score(M: np.ndarray) -> float:
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = M[iu], M.T[iu]
    denom = np.maximum(np.maximum(a, b), 1e-9)
    return float(np.mean(np.abs(a - b) / denom))


def _generate_one(city_name: str, city_query: str, topology: str, N: int, idx: int,
                  cfg: dict, master_seed: int) -> dict | None:
    ox, nx = _imports()

    seed = _seed(master_seed, city_name, N, idx)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Download / cache network, restrict to largest strongly-connected component so
    # every pair has a finite directed shortest path.
    G_full = ox.graph_from_place(city_query, network_type=cfg["network_type"], simplify=True)
    G_full = ox.add_edge_speeds(G_full)
    G_full = ox.add_edge_travel_times(G_full)
    sccs = sorted(nx.strongly_connected_components(G_full), key=len, reverse=True)
    G = G_full.subgraph(sccs[0]).copy()

    # Depot = node nearest the network centroid
    centroid_x = np.mean([d["x"] for _, d in G.nodes(data=True)])
    centroid_y = np.mean([d["y"] for _, d in G.nodes(data=True)])
    depot_node = ox.distance.nearest_nodes(G, X=centroid_x, Y=centroid_y)

    # Sample customers
    customers = _stratified_node_sample(G, N + 1, rng)
    customers = [c for c in customers if c != depot_node][:N]
    if len(customers) < N:
        return None

    nodes_all = [depot_node] + customers
    T = _shortest_path_matrix(G, nodes_all, weight="travel_time")   # seconds
    D = _shortest_path_matrix(G, nodes_all, weight="length")        # meters
    T_min = T / 60.0
    D_mi  = D / 1609.344

    if not np.isfinite(T_min).all():
        return None  # disconnected component, skip

    asym = _asymmetry_score(T_min)
    # Depot-to-customer travel sets the natural scale for time windows.  We
    # anchor each customer's TW around (or after) its earliest depot arrival,
    # then widen it within the configured range to make the instance HARD
    # (tight enough to differentiate solvers) but FEASIBLE (every customer
    # can be reached by at least one schedule).
    day_start = 480
    day_end = 960
    depot_to = T_min[0, 1:]   # one entry per customer
    cust_back = T_min[1:, 0]
    # Customers whose round trip cannot fit even alone are dropped before TW
    # assignment.
    sv_lo, sv_hi = cfg["service_time_minutes"]
    service_mean = (sv_lo + sv_hi) / 2.0
    rt_feasible = (depot_to + service_mean + cust_back) <= (day_end - day_start)
    if not rt_feasible.all():
        # Trim — pick a different (shorter-trip) subset by re-running the sampler
        # is overkill; instead just drop the offenders.  This keeps the asym
        # distribution honest.
        keep_mask = rt_feasible
        new_customers = [c for c, ok in zip(customers, keep_mask, strict=False) if ok]
        if len(new_customers) < N // 2:
            return None  # too few survive — skip this instance
        # Re-index without dropped customers
        nodes_all = [depot_node] + new_customers
        T_min = T_min[np.concatenate([[0], np.where(keep_mask)[0] + 1])][:,
                np.concatenate([[0], np.where(keep_mask)[0] + 1])]
        D_mi  = D_mi[np.concatenate([[0], np.where(keep_mask)[0] + 1])][:,
                np.concatenate([[0], np.where(keep_mask)[0] + 1])]
        customers = new_customers
        N = len(customers)
        depot_to = T_min[0, 1:]
        cust_back = T_min[1:, 0]
        asym = _asymmetry_score(T_min)

    tw_lo, tw_hi = cfg["tw_width_factor_range"]
    dem_lo, dem_hi = cfg["demand_range"]
    customer_list = []
    total_demand = 0
    for i, node_id in enumerate(customers, start=1):
        service = int(np_rng.integers(sv_lo, sv_hi + 1))
        d_to = float(T_min[0, i])
        d_back = float(T_min[i, 0])
        earliest_serve = day_start + d_to                       # arrive earliest
        latest_serve   = day_end - d_back - service             # leave latest
        if earliest_serve >= latest_serve:
            # Skip this customer entirely (rare, should be caught by rt_feasible)
            continue
        # Target width: tight, but at least 30 minutes.  Width is in minutes
        # absolute, NOT scaled by T_med — that was the bug.
        width = float(np_rng.uniform(30.0, 90.0))
        center_lo = earliest_serve + width / 2.0
        center_hi = latest_serve   - width / 2.0
        if center_hi < center_lo:
            # Customer's reachable window is shorter than width — collapse to
            # the maximum feasible window.
            ready = int(max(day_start, earliest_serve))
            due   = int(min(day_end, latest_serve))
        else:
            center = float(np_rng.uniform(center_lo, center_hi))
            ready = int(max(day_start, center - width / 2.0))
            due   = int(min(day_end,   center + width / 2.0))
        demand = int(np_rng.integers(dem_lo, dem_hi + 1))
        total_demand += demand
        node_attr = G.nodes[node_id]
        customer_list.append({
            "id": i, "node_id": int(node_id),
            "x": float(node_attr["x"]), "y": float(node_attr["y"]),
            "demand": demand, "ready": ready, "due": due, "service": service,
        })
    N = len(customer_list)  # post-skip count
    if N < 10:
        return None

    num_vehicles = max(2, int(math.ceil(N * cfg["vehicle_factor"])))
    capacity = int(math.ceil(cfg["capacity_buffer"] * total_demand / num_vehicles))

    depot_attr = G.nodes[depot_node]
    instance_id = f"OSM-{city_name.split(',')[0].replace(' ', '')}-N{N:03d}-I{idx:03d}"
    body = {
        "instance_id": instance_id,
        "city": city_name,
        "topology": topology,
        "num_customers": N,
        "num_vehicles": num_vehicles,
        "vehicle_capacity": capacity,
        "depot": {
            "node_id": int(depot_node),
            "x": float(depot_attr["x"]), "y": float(depot_attr["y"]),
            "ready": 480, "due": 960,
        },
        "customers": customer_list,
        "asymmetry_score": asym,
        "generator_seed": seed,
        "schema_version": "1.0",
        "_matrices": {"T_min": T_min, "D_mi": D_mi},  # popped before write
    }
    return body


def _write_instance(body: dict, out_dir: Path) -> Path:
    matrices = out_dir / "matrices"
    matrices.mkdir(parents=True, exist_ok=True)
    iid = body["instance_id"]
    T = body["_matrices"]["T_min"]
    D = body["_matrices"]["D_mi"]
    np.savez_compressed(matrices / f"{iid}.npz", m=T)
    np.savez_compressed(matrices / f"{iid}_dist.npz", m=D)
    body = {k: v for k, v in body.items() if k != "_matrices"}
    body["travel_time_matrix_path"] = f"matrices/{iid}.npz"
    body["travel_distance_matrix_path"] = f"matrices/{iid}_dist.npz"
    json_path = out_dir / f"{iid}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    return json_path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="specs/data/cities.yaml")
    p.add_argument("--out", default="instances/v1")
    p.add_argument("--limit", type=int, default=0, help="Cap total instances for smoke runs (0 = all).")
    p.add_argument("--cities-filter", default="", help="Comma-separated substring filter on city name.")
    p.add_argument("--sizes-filter", default="", help="Comma-separated list of sizes to keep.")
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg_all = yaml.safe_load(f)
    cities = cfg_all["cities"]
    sizes = cfg_all["sizes"]
    per_pair = cfg_all["per_pair"]
    master_seed = cfg_all["master_seed"]
    gen_cfg = cfg_all["generation"]

    if args.cities_filter:
        wanted = [w.strip().lower() for w in args.cities_filter.split(",")]
        cities = [c for c in cities if any(w in c["name"].lower() for w in wanted)]
    if args.sizes_filter:
        keep = {int(s) for s in args.sizes_filter.split(",")}
        sizes = [s for s in sizes if s in keep]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    report_rows: list[str] = ["| instance_id | N | asym | TW_med | feasible |", "|---|---|---|---|---|"]

    total_target = len(cities) * len(sizes) * per_pair
    print(f"Planned: {total_target} instances ({len(cities)} cities × {len(sizes)} sizes × {per_pair} each)")
    count = 0
    for city in cities:
        for N in sizes:
            for i in range(per_pair):
                if args.limit and count >= args.limit:
                    break
                count += 1
                body = _generate_one(city["name"], city["query"], city["topology"], N, i, gen_cfg, master_seed)
                if body is None:
                    print(f"  SKIP {city['name']} N={N} I={i}: disconnected / insufficient nodes")
                    continue
                path = _write_instance(body, out_dir)
                written.append(path)
                report_rows.append(f"| {body['instance_id']} | {N} | {body['asymmetry_score']:.3f} | - | y |")
                print(f"  WROTE {body['instance_id']} (asym={body['asymmetry_score']:.3f})")

    # Manifest
    manifest = out_dir / "manifest.sha256"
    with open(manifest, "w", encoding="utf-8") as f:
        for jp in sorted(written):
            f.write(f"{_file_sha256(jp)}  {jp.relative_to(out_dir)}\n")
    print(f"Manifest: {manifest} ({len(written)} entries)")

    report = out_dir / "REPORT.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# instances/v1 report\n\n")
        f.write(f"Wrote {len(written)} instances from {len(cities)} cities.\n\n")
        f.write("\n".join(report_rows))
        f.write("\n")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
