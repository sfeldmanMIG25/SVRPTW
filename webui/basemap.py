"""Minimal OSM tile fetcher + stitcher for v1/v2 instance maps.

Stdlib + PIL only (PIL is already a transitive dep via matplotlib).
No contextily / no geopandas / no heavy GIS stack.

WHY: render_llm_compare in `svrptw/viz/renderer.py` needs a clean
basemap when the instance carries WGS84 coordinates (v1/v2 OSM
instances do; Solomon/Homberger don't). The basemap makes it MUCH
easier for VLM judges to read the routes against real geography
("this route loops back into the center vs that one doesn't").

Cache: fetched tiles under `webui/cache/tiles/{style}/{z}/{x}/{y}.png`.
Tile sources are CartoDB-Positron (clean light style, LLM-friendly)
or osm.standard. We respect the OSM usage policy by setting a
descriptive User-Agent and caching aggressively.
"""
from __future__ import annotations

import logging
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from svrptw.io.instance import Instance
    from PIL import Image as _PILImage

_LOG = logging.getLogger("webui.basemap")
_CACHE = Path(__file__).resolve().parent / "cache" / "tiles"
_TILE_SIZE = 256
_USER_AGENT = "WorldsFinestVRP/0.1 (research; contact via repo)"
_TIMEOUT_S = 8.0


_TILE_TEMPLATES = {
    "positron": "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    "voyager":  "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    "dark":     "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    "osm":      "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}

_DEFAULT_STYLE = "positron"


@dataclass
class BasemapResult:
    """Stitched basemap image plus the lon/lat extent it covers."""
    image_bytes: bytes        # PNG
    extent_lonlat: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    zoom: int
    n_tiles: int


def is_geographic(inst: "Instance") -> bool:
    """True iff the instance's customer x/y look like WGS84 coords.

    Heuristic: lon in [-180, 180], lat in [-90, 90], and (max-min) span
    is plausible city-scale (< 5 deg in either dim).
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


def instance_bbox(inst: "Instance",
                   padding_frac: float = 0.05) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) covering depot + all customers."""
    xs = [c.x for c in inst.customers] + [inst.depot.x]
    ys = [c.y for c in inst.customers] + [inst.depot.y]
    pad_x = (max(xs) - min(xs)) * padding_frac
    pad_y = (max(ys) - min(ys)) * padding_frac
    return (min(xs) - pad_x, min(ys) - pad_y,
            max(xs) + pad_x, max(ys) + pad_y)


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
                   / math.pi) / 2.0 * n)
    return x_tile, y_tile


def _tile_to_lonlat_corner(x: int, y: int, zoom: int) -> tuple[float, float]:
    """North-west corner of tile (x, y) at zoom z."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def _pick_zoom(bbox: tuple[float, float, float, float],
                target_max_tiles: int = 9) -> int:
    """Pick the largest zoom such that the bbox covers <= target_max_tiles."""
    min_lon, min_lat, max_lon, max_lat = bbox
    for z in range(18, 0, -1):
        x0, y0 = _lonlat_to_tile(min_lon, max_lat, z)
        x1, y1 = _lonlat_to_tile(max_lon, min_lat, z)
        n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
        if n_tiles <= target_max_tiles:
            return z
    return 1


def _fetch_tile(z: int, x: int, y: int, style: str) -> Optional[bytes]:
    """Return the PNG bytes for tile (z, x, y), using on-disk cache."""
    cache_path = _CACHE / style / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except OSError:
            pass
    template = _TILE_TEMPLATES.get(style, _TILE_TEMPLATES[_DEFAULT_STYLE])
    url = template.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        _LOG.warning("tile fetch failed %s: %s", url, e)
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_path.write_bytes(data)
    except OSError as e:
        _LOG.warning("tile cache write failed %s: %s", cache_path, e)
    return data


def fetch_basemap(bbox: tuple[float, float, float, float],
                   *,
                   style: str = _DEFAULT_STYLE,
                   target_max_tiles: int = 9) -> Optional[BasemapResult]:
    """Fetch + stitch tiles covering bbox; return PNG bytes + actual extent.

    Returns None if PIL is missing OR if no tile fetched successfully
    (e.g. offline). Caller decides whether to fall back to a clean grid.
    """
    try:
        from PIL import Image
    except ImportError:
        _LOG.warning("Pillow not installed; cannot stitch basemap")
        return None
    z = _pick_zoom(bbox, target_max_tiles=target_max_tiles)
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = _lonlat_to_tile(min_lon, max_lat, z)   # NW
    x1, y1 = _lonlat_to_tile(max_lon, min_lat, z)   # SE
    cols, rows = (x1 - x0 + 1), (y1 - y0 + 1)
    canvas = Image.new("RGB", (cols * _TILE_SIZE, rows * _TILE_SIZE), (240, 240, 240))
    fetched_any = False
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            data = _fetch_tile(z, ix, iy, style)
            if data is None:
                continue
            try:
                tile = Image.open(BytesIO(data)).convert("RGB")
            except Exception as e:
                _LOG.warning("tile decode failed (%d,%d): %s", ix, iy, e)
                continue
            canvas.paste(tile, ((ix - x0) * _TILE_SIZE, (iy - y0) * _TILE_SIZE))
            fetched_any = True
    if not fetched_any:
        return None
    nw_lon, nw_lat = _tile_to_lonlat_corner(x0, y0, z)
    se_lon, se_lat = _tile_to_lonlat_corner(x1 + 1, y1 + 1, z)
    out = BytesIO()
    canvas.save(out, format="PNG")
    return BasemapResult(
        image_bytes=out.getvalue(),
        extent_lonlat=(nw_lon, se_lat, se_lon, nw_lat),
        zoom=z,
        n_tiles=cols * rows,
    )
