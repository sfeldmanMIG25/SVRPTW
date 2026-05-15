"""Render multiple solvers' solutions for one instance into a single side-by-side
figure.  Useful for diagnosing why one solver dominates another on a specific
instance.

Usage:
    python bench/scripts/compare_solvers_visual.py \
        --instance instances/v1_smoke/OSM-Manhattan-N050-I000.json \
        --solvers greedy,lkh3@1,auction_gart \
        --out bench/figures/I000_compare.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import auction_gart, greedy, lkh3, ortools_gart, ortools_solver

_FACTORIES = {
    "greedy":          lambda inst, s: greedy.solve(inst, s),
    "ortools@1":       lambda inst, s: ortools_solver.solve(inst, s, 1.0),
    "ortools@10":      lambda inst, s: ortools_solver.solve(inst, s, 10.0),
    "ortools_gart@10": lambda inst, s: ortools_gart.solve(inst, s, 10.0),
    "lkh3@1":          lambda inst, s: lkh3.solve(inst, s, 1.0),
    "lkh3@10":         lambda inst, s: lkh3.solve(inst, s, 10.0),
    "auction_gart":    lambda inst, s: auction_gart.solve(inst, s),
}


def _draw(ax, inst, sol):
    cust_xy = {c.id: (c.x, c.y) for c in inst.customers}
    depot = (inst.depot.x, inst.depot.y)
    palette = plt.cm.tab20.colors
    for vi, route in enumerate(sol.routes):
        if not route.customers:
            continue
        color = palette[vi % len(palette)]
        path = [depot] + [cust_xy[c] for c in route.customers] + [depot]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.7, zorder=3)
        cust_xs = [cust_xy[c][0] for c in route.customers]
        cust_ys = [cust_xy[c][1] for c in route.customers]
        ax.scatter(cust_xs, cust_ys, c=[color], s=22, zorder=4)
    ax.scatter([depot[0]], [depot[1]], c="black", s=120, marker="*",
               edgecolor="white", linewidths=1, zorder=5)
    ax.set_aspect("equal")
    ax.set_title(
        f"{sol.solver}\n"
        f"cost={sol.metrics['operational_cost']:.1f}  "
        f"K={int(sol.metrics['num_vehicles_used'])}  "
        f"miss={int(sol.metrics['missed_deliveries'])}",
        fontsize=10,
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--solvers", default="greedy,lkh3@1,auction_gart")
    p.add_argument("--out", default="bench/figures/compare.png")
    args = p.parse_args(argv)

    inst = load_instance(args.instance)
    solvers = args.solvers.split(",")
    settings = Settings()
    fig, axes = plt.subplots(1, len(solvers), figsize=(6 * len(solvers), 6),
                             squeeze=False)
    for i, sname in enumerate(solvers):
        sol = _FACTORIES[sname](inst, settings)
        _draw(axes[0][i], inst, sol)
    fig.suptitle(f"{inst.instance_id}", fontsize=12)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
