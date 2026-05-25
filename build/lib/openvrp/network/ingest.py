"""Network ingestion (SPEC-OPENVRP-06 §2).

Loads an OSM-derived or user-supplied graph; snaps stops/depots per
``SnapConfig``; produces a sparse CSR adjacency for OD construction.

Snapped node id per stop is recorded into ``SolveDiagnostics.snap_report``.

Implementation notes:
- osmnx is imported lazily; absent ⇒ ``MissingExtra("network")``.
- ``cache_dir`` defaults to ``~/.cache/openvrp``; second run is a cache hit.
- Graph fetch/load happens before the solve timer (caller of this module
  is responsible for that contract).
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openvrp.errors import MissingExtra, ProblemValidationError, ValidationReport
from openvrp.schema.input import Coordinate, Network, SnapConfig


@dataclass
class LoadedNetwork:
    """A loaded networkx DiGraph plus metadata."""

    graph: Any                       # networkx.DiGraph
    crs: str                         # final CRS (default EPSG:4326 for lon/lat)
    node_xy: dict[int, tuple[float, float]]   # node -> (lon, lat)
    cache_key: str


def _cache_dir(network: Network) -> Path:
    d = network.cache_dir or str(Path.home() / ".cache" / "openvrp")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(network: Network) -> str:
    """Stable hash of the ingest spec."""
    payload = json.dumps({
        "source": network.source,
        "osm_place": network.osm_place,
        "osm_bbox": list(network.osm_bbox) if network.osm_bbox else None,
        "graph_path": network.graph_path,
        "graph_crs": network.graph_crs,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def load_network(network: Network) -> LoadedNetwork:
    """Load (or cache-hit) the network graph. Returns a ``LoadedNetwork``."""
    try:
        import networkx as nx
    except ImportError as e:
        raise MissingExtra("network", reason=str(e)) from e

    cache_dir = _cache_dir(network)
    key = _cache_key(network)
    cache_file = cache_dir / f"graph_{key}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            ln = pickle.load(f)
        if isinstance(ln, LoadedNetwork):
            return ln

    if network.source in ("osm_place", "osm_bbox"):
        try:
            import osmnx as ox
        except ImportError as e:
            raise MissingExtra("network", reason=f"osmnx required: {e}") from e
        if network.source == "osm_place":
            if not network.osm_place:
                raise ValueError("osm_place source requires Network.osm_place.")
            g = ox.graph_from_place(network.osm_place, network_type="drive")
        else:
            if network.osm_bbox is None:
                raise ValueError("osm_bbox source requires Network.osm_bbox.")
            west, south, east, north = network.osm_bbox
            g = ox.graph_from_bbox(north, south, east, west, network_type="drive")
        # Convert MultiDiGraph -> DiGraph (keep shortest edge per pair)
        h = nx.DiGraph()
        for u, v, data in g.edges(data=True):
            length = float(data.get("length", 0.0))
            speed = float(data.get("speed_kph", 50.0))   # km/h default
            travel_time_s = (length / 1000.0) / max(0.1, speed) * 3600.0
            if h.has_edge(u, v):
                if length < h[u][v]["length"]:
                    h[u][v]["length"] = length
                    h[u][v]["travel_time"] = travel_time_s
            else:
                h.add_edge(u, v, length=length, travel_time=travel_time_s)
        # Node coords
        node_xy = {n: (float(d["x"]), float(d["y"])) for n, d in g.nodes(data=True)}
        crs = "EPSG:4326"
    elif network.source == "graph_file":
        if not network.graph_path:
            raise ValueError("graph_file source requires Network.graph_path.")
        path = Path(network.graph_path)
        if not path.exists():
            raise FileNotFoundError(f"Network.graph_path not found: {path}")
        if path.suffix.lower() == ".graphml":
            g = nx.read_graphml(path)
            h = nx.DiGraph()
            for u, v, data in g.edges(data=True):
                length = float(data.get("length", data.get("weight", 1.0)))
                travel_time = float(data.get("travel_time", length))
                h.add_edge(int(u), int(v), length=length, travel_time=travel_time)
            node_xy = {}
            for n, d in g.nodes(data=True):
                if "x" in d and "y" in d:
                    node_xy[int(n)] = (float(d["x"]), float(d["y"]))
        else:
            raise ValueError(f"Unsupported graph file format: {path.suffix}")
        crs = network.graph_crs or "EPSG:4326"
    elif network.source == "none":
        raise ValueError("Cannot load_network from source='none'.")
    else:
        raise ValueError(f"Unknown Network.source={network.source!r}")

    ln = LoadedNetwork(graph=h, crs=crs, node_xy=node_xy, cache_key=key)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(ln, f)
    except Exception:
        pass   # cache write is best-effort
    return ln


def _haversine_meters(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance in meters between two (lon, lat) points."""
    import math
    lon1, lat1 = a
    lon2, lat2 = b
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    aa = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(aa))


def snap_to_nodes(loaded: LoadedNetwork, points: dict[str, Coordinate],
                  config: SnapConfig) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Snap each point id -> nearest graph node. Returns ``(stop_id -> node_id, snap_report)``.

    ``snap_report`` carries one record per stop with ``stop_id``,
    ``node_id``, ``snap_meters``.
    """
    node_ids = list(loaded.node_xy.keys())
    node_coords = list(loaded.node_xy.values())

    rep_issues: list[dict[str, Any]] = []
    rep_validation = ValidationReport()
    out: dict[str, int] = {}
    for stop_id, coord in points.items():
        target = coord.as_tuple()
        # Brute-force nearest-node (for big graphs we'd kd-tree)
        best_d = float("inf")
        best_n = node_ids[0]
        for n, xy in zip(node_ids, node_coords, strict=True):
            d = _haversine_meters(target, xy)
            if d < best_d:
                best_d = d
                best_n = n
        out[stop_id] = best_n
        rep_issues.append({
            "stop_id": stop_id,
            "node_id": best_n,
            "snap_meters": round(best_d, 2),
        })
        if best_d > config.max_snap_meters:
            if config.strict:
                rep_validation.add_error(
                    "snap.too_far",
                    f"Stop {stop_id!r} snapped to {best_d:.0f}m > max_snap_meters {config.max_snap_meters:.0f}.",
                    f"stops[id={stop_id!r}]")
            else:
                rep_validation.add_warning(
                    "snap.too_far",
                    f"Stop {stop_id!r} snapped to {best_d:.0f}m > max_snap_meters {config.max_snap_meters:.0f}.",
                    f"stops[id={stop_id!r}]")
    if rep_validation.errors:
        raise ProblemValidationError(rep_validation.issues)
    return out, rep_issues


__all__ = ["LoadedNetwork", "load_network", "snap_to_nodes"]
