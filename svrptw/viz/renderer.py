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
