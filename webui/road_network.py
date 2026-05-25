"""Road-network routing for solution rendering.

Lazy-fetches the OSM driving network for an instance's bounding box,
caches it to disk, and returns shortest-path polylines per route leg.

Used by `svrptw.viz.renderer.render_llm_compare` when
`route_mode="network"`. If the network fetch fails (offline, bad bbox)
or shortest-path fails for ANY leg of a route, the renderer falls back
to `stops_only` mode for that ENTIRE solution image.

Cached graphs live under `webui/cache/networks/<bbox-hash>.graphml`.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import networkx as _nx  # noqa

_LOG = logging.getLogger("webui.road_network")
_CACHE = Path(__file__).resolve().parent / "cache" / "networks"
_CACHE.mkdir(parents=True, exist_ok=True)


def _bbox_hash(bbox: tuple[float, float, float, float]) -> str:
    return hashlib.sha1(
        f"{bbox[0]:.5f},{bbox[1]:.5f},{bbox[2]:.5f},{bbox[3]:.5f}".encode()
    ).hexdigest()[:12]


def get_network(bbox: tuple[float, float, float, float],
                 network_type: str = "drive"):
    """Fetch (or load from cache) the OSM driving graph covering bbox.

    bbox = (min_lon, min_lat, max_lon, max_lat) in WGS84.
    Returns a networkx MultiDiGraph in lon/lat node coords (osmnx default)
    or None on failure.
    """
    cache_path = _CACHE / f"{network_type}__{_bbox_hash(bbox)}.graphml"
    try:
        import osmnx as ox
    except ImportError:
        _LOG.warning("osmnx not installed; cannot fetch network")
        return None

    if cache_path.exists():
        try:
            return ox.load_graphml(str(cache_path))
        except Exception as e:
            _LOG.warning("cached graph load failed (%s); refetching", e)

    try:
        # osmnx 1.x signature: (north, south, east, west) for bbox=
        # osmnx 2.x signature: bbox=(left, bottom, right, top)
        # Use the modern 2.x signature; fall back to 1.x on TypeError.
        try:
            G = ox.graph_from_bbox(
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                network_type=network_type, simplify=True, retain_all=False,
            )
        except TypeError:
            G = ox.graph_from_bbox(
                bbox[3], bbox[1], bbox[2], bbox[0],
                network_type=network_type, simplify=True, retain_all=False,
            )
        ox.save_graphml(G, str(cache_path))
        _LOG.info("fetched + cached graph %s (%d nodes, %d edges)",
                   cache_path.name, len(G.nodes), len(G.edges))
        return G
    except Exception as e:
        _LOG.warning("graph_from_bbox failed: %s", e)
        return None


def route_polyline(G, src_lonlat: tuple[float, float],
                    dst_lonlat: tuple[float, float],
                    weight: str = "length"
                    ) -> Optional[list[tuple[float, float]]]:
    """Shortest-path polyline (lon, lat) sequence from src to dst.

    Returns None on failure (graph None, nodes off-network, no path).
    """
    if G is None:
        return None
    try:
        import osmnx as ox
        import networkx as nx
        # nearest_nodes accepts (X=lon, Y=lat) per osmnx convention
        n_src = ox.nearest_nodes(G, X=src_lonlat[0], Y=src_lonlat[1])
        n_dst = ox.nearest_nodes(G, X=dst_lonlat[0], Y=dst_lonlat[1])
        path = nx.shortest_path(G, n_src, n_dst, weight=weight)
        return [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]
    except Exception as e:
        _LOG.debug("route_polyline failed (%s -> %s): %s", src_lonlat, dst_lonlat, e)
        return None


def route_solution(G, depot_lonlat: tuple[float, float],
                    visit_lonlat: list[tuple[float, float]]
                    ) -> Optional[list[tuple[float, float]]]:
    """Concatenate polylines for depot -> v[0] -> v[1] -> ... -> depot.

    Returns the full polyline OR None if any leg fails. The renderer
    treats None as a signal to fall back to stops_only mode.
    """
    if G is None or not visit_lonlat:
        return None
    full: list[tuple[float, float]] = []
    pts = [depot_lonlat] + list(visit_lonlat) + [depot_lonlat]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = route_polyline(G, a, b)
        if seg is None:
            return None
        if full and seg:
            seg = seg[1:]   # avoid duplicating the join point
        full.extend(seg)
    return full
