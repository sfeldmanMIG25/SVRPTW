"""Route plotter — produces the image ViVRP consumes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from svrptw.io import Instance
from svrptw.solvers.common import Solution

_PALETTE = plt.cm.tab20.colors


def render_solution(inst: Instance, sol: Solution, out_path: str | Path,
                    dpi: int = 180, show_unrouted: bool = True,
                    fair_mode: bool = False) -> Path:
    """Render a solution to PNG for visualization or VLM judging.

    `fair_mode=True` (recommended for VLM judging): all routes drawn in a
    single muted color so the VLM cannot use "many bright colors" as a
    proxy for "many routes" — that's a rendering confound, not a real
    routing-quality signal. Route count is also removed from the title.
    `fair_mode=False` (default, for human debugging): colorful per-route
    palette + route count in title; legacy behaviour.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cust_xy = {c.id: (c.x, c.y) for c in inst.customers}
    depot = (inst.depot.x, inst.depot.y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    served = sol.visited_customer_ids()
    if show_unrouted:
        unrouted = [c.id for c in inst.customers if c.id not in served]
        if unrouted:
            xs = [cust_xy[i][0] for i in unrouted]
            ys = [cust_xy[i][1] for i in unrouted]
            ax.scatter(xs, ys, c="#cccccc", s=14, marker="x",
                       label=f"unrouted ({len(unrouted)})", zorder=2)

    # Info-bearing fair_mode: route color = load utilisation (viridis 0→1),
    # directional arrows show traversal direction (matters for asymmetric
    # OSM with one-ways), no numeric metrics in the title (those go in the
    # text channel of the VLM prompt).
    if fair_mode:
        import matplotlib.cm as cm
        cap = max(1.0, float(inst.vehicle_capacity))
        cust_demand = {c.id: c.demand for c in inst.customers}
        viridis = cm.get_cmap("viridis")
        for route in sol.routes:
            if not route.customers:
                continue
            load = float(sum(cust_demand.get(c, 0) for c in route.customers))
            util = min(1.0, load / cap)
            color = viridis(util)
            path_pts = [depot] + [cust_xy[c] for c in route.customers] + [depot]
            xs = [p[0] for p in path_pts]
            ys = [p[1] for p in path_pts]
            ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.85, zorder=3)
            # Directional arrows on ~3 evenly-spaced legs along the route.
            n_legs = len(path_pts) - 1
            if n_legs >= 1:
                stride = max(1, n_legs // 3)
                for i in range(0, n_legs, stride):
                    x0, y0 = path_pts[i]
                    x1, y1 = path_pts[i + 1]
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    dx, dy = (x1 - x0) * 0.18, (y1 - y0) * 0.18
                    ax.annotate(
                        "", xy=(mx + dx, my + dy), xytext=(mx - dx, my - dy),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        alpha=0.95, lw=1.2),
                        zorder=4,
                    )
            cust_xs = [cust_xy[c][0] for c in route.customers]
            cust_ys = [cust_xy[c][1] for c in route.customers]
            ax.scatter(cust_xs, cust_ys, c=[color], s=24, zorder=4,
                       edgecolor="black", linewidths=0.3)
        # Colorbar carrying the util→color legend.
        sm = cm.ScalarMappable(
            norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=viridis,
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("route load util (0=empty, 1=full)", fontsize=8)
    else:
        for vi, route in enumerate(sol.routes):
            if not route.customers:
                continue
            color = _PALETTE[vi % len(_PALETTE)]
            path = [depot] + [cust_xy[c] for c in route.customers] + [depot]
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.7, zorder=3)
            cust_xs = [cust_xy[c][0] for c in route.customers]
            cust_ys = [cust_xy[c][1] for c in route.customers]
            ax.scatter(cust_xs, cust_ys, c=[color], s=22, zorder=4)

    ax.scatter([depot[0]], [depot[1]], c="black", s=120, marker="*",
               edgecolor="white", linewidths=1, zorder=5, label="depot")

    if fair_mode:
        # Strip route-count from the title so the VLM doesn't read "K=11"
        # vs "K=7" and infer quality from that alone.
        ax.set_title(f"{inst.instance_id} | N={inst.num_customers} | "
                     f"miss={int(sol.metrics.get('missed_deliveries', 0))}")
    else:
        ax.set_title(f"{inst.instance_id} | {sol.solver} | "
                     f"N={inst.num_customers}, K={sol.num_vehicles_used}, "
                     f"miss={int(sol.metrics.get('missed_deliveries', 0))}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse
    import json

    from svrptw.config import Settings
    from svrptw.io import load_instance
    from svrptw.solvers.classical import greedy as greedy_mod
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--out", default="solution.png")
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = greedy_mod.solve(inst, Settings())
    path = render_solution(inst, sol, args.out)
    print(json.dumps({"image": str(path), "metrics": sol.metrics}, indent=2))



# ---- SPEC-WEBUI-01: snapshot helper + llm_compare mode -----------------

import time as _time
from typing import Optional as _Optional

_PALETTE_DISTINCT = plt.cm.tab10.colors


def render_llm_compare(inst: Instance, sol: Solution, out_path: str | Path,
                       dpi: int = 180,
                       title: _Optional[str] = None,
                       show_basemap: bool = True,
                       palette_offset: int = 0,
                       basemap_style: str = "positron_nolabels",
                       route_mode: str = "network",
                       padding_frac: float = 0.02,
                       # SPEC-WEBUI-10 -- overlay variants for VLM ablation.
                       # "minimal":      no title, legend = R# + cust count only
                       # "utilization":  + util% in legend
                       # "cost":         + per-route route-distance in legend
                       # "tw":           + TW-tightness ring on each customer
                       # "full":         all of the above (legacy default)
                       overlay_mode: str = "full",
                       # Stable color assignment ACROSS pair renders:
                       # caller passes an empty dict on the first render and
                       # the same dict on subsequent renders. Routes with the
                       # same customer-set get the same color in both images;
                       # new routes (different customer set) get the next slot.
                       route_color_map: dict | None = None) -> Path:
    """Render in LLM-comparison-safe mode.

    Distinct per-route colors (tab10), numbered routes, white background,
    optional clean basemap. The palette is consistent across paired
    comparisons when callers pass the same palette_offset, so route i
    in solution A and route i in solution B share a color and the VLM
    is not confounded by color permutations.

    show_basemap=True attempts contextily overlay if the instance carries
    lat/lon metadata; falls back to a clean grid otherwise. Route geometry
    is plotted in the instance's coordinate frame regardless.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # SPEC-WEBUI-08 -- proper Web Mercator projection + contextily basemap
    # for geographic (v1/v2 OSM) instances. Routes plot in mercator meters,
    # contextily fetches OSM/CartoDB tiles auto-aligned. Solomon/Homberger
    # (non-geographic) fall back to the original clean-grid mode.
    geographic = False
    proj = None
    if show_basemap:
        try:
            from webui.basemap import is_geographic
            geographic = is_geographic(inst)
        except Exception:
            geographic = False
    if geographic:
        try:
            from pyproj import Transformer
            proj = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        except Exception:
            geographic = False
            proj = None

    def _xy(lon: float, lat: float) -> tuple[float, float]:
        if proj is not None:
            return proj.transform(lon, lat)
        return (lon, lat)

    cust_xy = {c.id: _xy(c.x, c.y) for c in inst.customers}
    depot = _xy(inst.depot.x, inst.depot.y)
    cust_demand = {c.id: c.demand for c in inst.customers}
    cap = max(1.0, float(inst.vehicle_capacity))

    # extra width for the side legend
    fig_w = 11 if geographic else 9
    fig, ax = plt.subplots(figsize=(fig_w, 8), facecolor="white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    # SPEC-WEBUI-09 -- road-following routes via osmnx, with stops_only fallback.
    # When route_mode="network" we fetch the OSM driving graph and shortest-path
    # each leg. If ANY leg fails we drop to stops_only for the WHOLE image
    # (per user direction: "if we cannot solve for best route along network,
    # then drop routes from being shown and instead just show stops + number sequence").
    network_polylines: dict[int, list[tuple[float, float]]] = {}
    network_failed = False
    network_fail_reason = ""
    if route_mode == "network" and geographic:
        try:
            from webui.basemap import instance_bbox
            from webui.road_network import get_network, route_solution
            G = get_network(instance_bbox(inst, padding_frac=0.02))
            if G is None:
                network_failed = True
                network_fail_reason = "graph fetch returned None"
            else:
                cust_by_id = {c.id: c for c in inst.customers}
                depot_lonlat = (inst.depot.x, inst.depot.y)
                for ridx, route in enumerate(sol.routes):
                    if not route.customers:
                        continue
                    visits = []
                    for cid in route.customers:
                        cu = cust_by_id.get(cid)
                        if cu is None:
                            network_failed = True
                            network_fail_reason = f"customer id={cid} not in instance"
                            break
                        visits.append((cu.x, cu.y))
                    if network_failed:
                        break
                    poly_lonlat = route_solution(G, depot_lonlat, visits)
                    if poly_lonlat is None:
                        network_failed = True
                        network_fail_reason = f"shortest-path failed on route {ridx+1}"
                        break
                    network_polylines[ridx] = [_xy(lon, lat) for (lon, lat) in poly_lonlat]
        except Exception as _e:
            network_failed = True
            network_fail_reason = f"{type(_e).__name__}: {_e}"
    if network_failed and route_mode == "network":
        import sys as _sys
        print(f"[render_llm_compare] network routing failed -> stops_only fallback: {network_fail_reason}",
               file=_sys.stderr)

    effective_mode = ("stops_only" if network_failed
                       else ("network" if route_mode == "network" and geographic and network_polylines
                             else ("stops_only" if route_mode == "stops_only" else "straight")))

    # unrouted customers (light grey x)
    served = sol.visited_customer_ids()
    unrouted = [c.id for c in inst.customers if c.id not in served]
    if unrouted:
        xs = [cust_xy[i][0] for i in unrouted]
        ys = [cust_xy[i][1] for i in unrouted]
        ax.scatter(xs, ys, c="#999999", s=22, marker="x", linewidths=1.4, zorder=4,
                   label=f"unrouted ({len(unrouted)})")

    # Build legend entries for the side legend
    from matplotlib.lines import Line2D as _Line2D
    legend_handles: list = []

    def _draw_arrows(xs: list[float], ys: list[float], color,
                      n_arrows: int = 4) -> None:
        """Place big directional arrowheads with white halo along the polyline.

        Visible arrow direction = visit sequence. White outline makes the
        arrowheads readable even when many colored routes overlap in the
        same area.
        """
        from matplotlib.patheffects import Stroke as _Stroke, Normal as _Normal
        n = len(xs)
        if n < 2:
            return
        for i in range(1, n_arrows + 1):
            idx = int(n * i / (n_arrows + 1))
            if idx < 1 or idx >= n:
                continue
            x0, y0 = xs[idx - 1], ys[idx - 1]
            x1, y1 = xs[idx], ys[idx]
            if x0 == x1 and y0 == y1:
                continue
            ann = ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                               arrowprops=dict(arrowstyle="-|>", color=color,
                                                lw=0.0, alpha=1.0,
                                                mutation_scale=20,
                                                shrinkA=0, shrinkB=0),
                               zorder=6)
            # White halo behind the arrow head so it reads against any basemap
            try:
                ann.arrow_patch.set_path_effects([
                    _Stroke(linewidth=2.6, foreground="white"),
                    _Normal(),
                ])
            except Exception:
                pass

    # SPEC-WEBUI-14 -- COLOR BY VEHICLE ID (route index), assigned at birth.
    # Per user direction: route_id / vehicle must maintain a fixed color
    # regardless of which customers that vehicle serves in A vs B.
    # vehicle 0 -> palette slot 0 in BOTH images, even if A's vehicle 0
    # carries customers {3,7,12} and B's vehicle 0 carries {5,9,14}.
    #
    # This makes "is vehicle 5 doing the same kind of work in A and B?"
    # the natural visual question. Sector / centroid keying was wrong:
    # it tied color to geography, breaking vehicle identity.
    _PAL_BIG = (list(plt.cm.tab20.colors)
                + list(plt.cm.tab20b.colors)
                + list(plt.cm.tab10.colors))

    def _color_for_route(ridx: int) -> tuple:
        # route_color_map is now optional / unused for vehicle-id keying;
        # we still accept it for API compat but don't write to it.
        return _PAL_BIG[(ridx + palette_offset) % len(_PAL_BIG)]

    _ridx_counter = [0]
    for ridx, route in enumerate(sol.routes):
        if not route.customers:
            continue
        color = _color_for_route(ridx)
        cust_xs = [cust_xy[c][0] for c in route.customers]
        cust_ys = [cust_xy[c][1] for c in route.customers]
        if effective_mode == "network" and ridx in network_polylines:
            pts = network_polylines[ridx]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linewidth=2.4, alpha=0.92, zorder=3,
                    solid_capstyle="round", solid_joinstyle="round",
                    path_effects=[__import__("matplotlib.patheffects",
                                              fromlist=["Stroke","Normal"]).Stroke(
                                      linewidth=4.0, foreground="white", alpha=0.6),
                                  __import__("matplotlib.patheffects",
                                              fromlist=["Normal"]).Normal()])
            _draw_arrows(xs, ys, color, n_arrows=max(3, min(6, len(route.customers) // 2)))
            ax.scatter(cust_xs, cust_ys, c=[color], s=28, zorder=4,
                       edgecolor="white", linewidths=0.9)
        elif effective_mode == "straight":
            path_pts = [depot] + [cust_xy[c] for c in route.customers] + [depot]
            xs = [p[0] for p in path_pts]; ys = [p[1] for p in path_pts]
            ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.88, zorder=3,
                    solid_capstyle="round")
            _draw_arrows(xs, ys, color, n_arrows=max(2, min(4, len(route.customers) // 3)))
            ax.scatter(cust_xs, cust_ys, c=[color], s=24, zorder=4,
                       edgecolor="white", linewidths=0.7)
        else:  # stops_only
            ax.scatter(cust_xs, cust_ys, c=[color], s=70, zorder=4,
                       edgecolor="white", linewidths=1.3)
            for seq, c in enumerate(route.customers, start=1):
                cx, cy = cust_xy[c]
                ax.text(cx, cy, str(seq),
                        fontsize=6.5, fontweight="bold", color="white",
                        ha="center", va="center", zorder=6)
        load = float(sum(cust_demand.get(c, 0) for c in route.customers))
        util = min(1.0, load / cap)
        # Per-route distance (in projection units; rough for legend purposes)
        if effective_mode == "network" and ridx in network_polylines:
            pts = network_polylines[ridx]
            dist = sum(((pts[i+1][0] - pts[i][0])**2 + (pts[i+1][1] - pts[i][1])**2)**0.5
                       for i in range(len(pts) - 1))
        else:
            seq = [depot] + [cust_xy[c] for c in route.customers] + [depot]
            dist = sum(((seq[i+1][0] - seq[i][0])**2 + (seq[i+1][1] - seq[i][1])**2)**0.5
                       for i in range(len(seq) - 1))
        # Build label per overlay_mode.  Cost/util are deliberately HIDDEN
        # in 'minimal' so the VLM has to judge from visual structure alone.
        if overlay_mode == "minimal":
            lbl = f"R{ridx+1:>2}  {len(route.customers):>3} cust"
        elif overlay_mode == "utilization":
            lbl = f"R{ridx+1:>2}  {len(route.customers):>3} cust  util {util*100:>3.0f}%"
        elif overlay_mode == "cost":
            lbl = f"R{ridx+1:>2}  {len(route.customers):>3} cust  d~{dist:.0f}"
        elif overlay_mode == "tw":
            lbl = f"R{ridx+1:>2}  {len(route.customers):>3} cust"
        else:  # "full"
            lbl = f"R{ridx+1:>2}  {len(route.customers):>3} cust  util {util*100:>3.0f}%  d~{dist:.0f}"
        legend_handles.append(_Line2D(
            [0], [0], color=color, lw=2.5, marker="o", markersize=5,
            label=lbl,
        ))

    # depot -- a black square
    ax.scatter([depot[0]], [depot[1]], c="black", s=130, marker="s",
               edgecolor="white", linewidths=1.4, zorder=5)
    legend_handles.insert(0, _Line2D([0], [0], color="black", lw=0,
                                       marker="s", markersize=8, label="depot"))

    # SPEC-WEBUI-10b -- TW-tightness ring overlay.
    # In 'tw' mode, draw a ring around each customer whose color encodes
    # how tight that customer's time window is. red = tight, yellow = mid,
    # green = wide. Lets the VLM see whether routes serve hard customers.
    if overlay_mode == "tw":
        try:
            day = max(1.0, float(inst.depot.due - inst.depot.ready))
            tw_widths = {c.id: float(c.due - c.ready) / day for c in inst.customers}
            served_set = sol.visited_customer_ids()
            from matplotlib.cm import RdYlGn as _cm
            for c in inst.customers:
                if c.id not in served_set:
                    continue
                w = max(0.0, min(1.0, tw_widths[c.id]))
                ring_color = _cm(w)  # 0=tight=red, 1=wide=green
                cx, cy = cust_xy[c.id]
                ax.scatter([cx], [cy], facecolors="none", edgecolors=[ring_color],
                           s=110, linewidths=1.6, zorder=4.5)
        except Exception:
            pass

    # Tight axis limits to crop basemap to the data extent (no whitespace pad).
    all_xs = [v[0] for v in cust_xy.values()] + [depot[0]]
    all_ys = [v[1] for v in cust_xy.values()] + [depot[1]]
    span_x = max(all_xs) - min(all_xs); span_y = max(all_ys) - min(all_ys)
    pad_x = span_x * padding_frac; pad_y = span_y * padding_frac
    ax.set_xlim(min(all_xs) - pad_x, max(all_xs) + pad_x)
    ax.set_ylim(min(all_ys) - pad_y, max(all_ys) + pad_y)

    # Real OSM basemap via contextily (geographic instances only).
    if geographic:
        try:
            import contextily as _ctx
            # Provider table: Positron (very washed; high contrast for routes),
            # PositronNoLabels (cleanest -- no street labels stealing attention),
            # Voyager (more street detail), OSM Mapnik (busy).
            provider_map = {
                "positron":          _ctx.providers.CartoDB.Positron,
                "positron_nolabels": _ctx.providers.CartoDB.PositronNoLabels,
                "voyager":           _ctx.providers.CartoDB.Voyager,
                "osm":               _ctx.providers.OpenStreetMap.Mapnik,
            }
            src = provider_map.get(basemap_style, provider_map["positron_nolabels"])
            # SPEC-WEBUI-12 -- in minimal/tw modes, suppress the
            # contextily attribution overlay (text bias). Other modes
            # keep small attribution per CartoDB ToS.
            attr = False if overlay_mode in ("minimal", "tw") else None
            _ctx.add_basemap(ax, source=src, crs="EPSG:3857",
                              attribution=attr,
                              attribution_size=6, zoom="auto", alpha=0.55)
        except Exception:
            ax.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)
    else:
        ax.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)

    # Side legend: color-coded route key off to the right.
    # SPEC-WEBUI-12 -- in minimal mode, suppress the legend entirely
    # (the route count K and per-route customer count would leak abstraction).
    if overlay_mode != "minimal":
        ax.legend(handles=legend_handles, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95,
                  borderaxespad=0.0, handlelength=1.5,
                  title=f"Routes (K={sum(1 for r in sol.routes if r.customers)})",
                  title_fontsize=9)

    # SPEC-WEBUI-12 -- in 'minimal' and 'tw' modes, strip ALL text from
    # the image including the A/B identifier. The prompt tells the VLM
    # which image is left vs right; the image itself carries no text bias.
    if title and overlay_mode not in ("minimal", "tw"):
        mode_tag = {"network": " [road-routed]",
                    "straight": " [straight-line]",
                    "stops_only": " [STOPS-ONLY: routing unavailable]"}.get(effective_mode, "")
        ax.set_title(title + mode_tag, fontsize=11, color="#222222")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def push_snapshot_to_webui(inst: Instance, sol: Solution, label: str,
                           *, mode: str = "llm_compare",
                           operator: _Optional[str] = None,
                           iteration: _Optional[int] = None,
                           snapshots_dir: _Optional[str | Path] = None) -> _Optional[Path]:
    """Render a snapshot and push it to webui.event_bus.BUS.

    Safe inside a solver hot path: matplotlib Agg render is ~50-150ms
    for N<=500; the bus push is a single dict append. Returns the
    snapshot path on success, None if the web UI module is not on
    sys.path (lets headless benches keep running).
    """
    try:
        from webui import client as _ui
    except Exception:
        return None
    if snapshots_dir is None:
        snapshots_dir = Path("webui/static/snapshots")
    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    ts = _time.time()
    safe_label = label.replace("/", "_").replace(" ", "_")[:40]
    fname = f"{int(ts*1000)}_{safe_label}.png"
    full = snapshots_dir / fname
    if mode == "llm_compare":
        render_llm_compare(inst, sol, full, title=label)
    elif mode == "fair":
        render_solution(inst, sol, full, fair_mode=True)
    else:
        render_solution(inst, sol, full, fair_mode=False)
    cost = sol.metrics.get("operational_cost") if hasattr(sol, "metrics") else None
    n_routes = sol.metrics.get("num_vehicles_used") if hasattr(sol, "metrics") else None
    stream = (operator or "default")
    meta: dict = {}
    if cost is not None:
        meta["cost"] = float(cost)
    if n_routes is not None:
        meta["n_routes"] = int(n_routes)
    if iteration is not None:
        meta["iteration"] = int(iteration)
    if operator is not None:
        meta["operator"] = operator
    _ui.push_snapshot(label=label, rel_path=fname, stream=stream, **meta)
    return full
