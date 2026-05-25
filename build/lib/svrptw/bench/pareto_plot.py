"""Generate a speed-quality Pareto figure from a bench JSON.

Usage:
    python -m svrptw.bench.pareto_plot bench/runs/v1_partial.json \
        --out bench/figures/v1_partial_pareto.pdf
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pareto_mask(points):
    n = len(points)
    keep = [True] * n
    for i in range(n):
        ci, ti = points[i]
        for j in range(n):
            if i == j or not keep[j]:
                continue
            cj, tj = points[j]
            if cj <= ci and tj <= ti and (cj < ci or tj < ti):
                keep[i] = False
                break
    return keep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--out", default="bench/figures/pareto.pdf")
    p.add_argument("--metric", default="operational_cost")
    p.add_argument("--per-n", action="store_true",
                   help="One subplot per N-bin instead of one mean point per solver.")
    args = p.parse_args(argv)

    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    rows = data["rows"]
    agg: dict[tuple[str, int], dict] = defaultdict(lambda: {"cost": [], "wall": []})
    for r in rows:
        if r.get(args.metric) is None:
            continue
        key = (r["solver"], r["n"])
        agg[key]["cost"].append(r[args.metric])
        agg[key]["wall"].append(r.get("wall_clock_seconds", 0))

    bins = sorted({n for _, n in agg.keys()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.per_n:
        fig, axes = plt.subplots(1, len(bins), figsize=(6 * len(bins), 5),
                                 sharex=False, sharey=False, squeeze=False)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6), squeeze=False)

    for col, n in enumerate(bins):
        ax = axes[0][col] if args.per_n else axes[0][0]
        solvers = sorted({s for (s, nn) in agg.keys() if nn == n})
        xs = []
        ys = []
        labels = []
        for s in solvers:
            a = agg[(s, n)]
            xs.append(sum(a["wall"]) / len(a["wall"]) if a["wall"] else 0)
            ys.append(sum(a["cost"]) / len(a["cost"]))
            labels.append(s)
        mask = _pareto_mask(list(zip(xs, ys, strict=False)))
        for x, y, lab, p in zip(xs, ys, labels, mask, strict=False):
            color = "tab:red" if p else "tab:gray"
            marker = "*" if p else "o"
            size = 200 if p else 80
            ax.scatter(x + 1e-3, y, c=color, marker=marker, s=size,
                       edgecolor="black", linewidths=0.7, zorder=3)
            ax.annotate(lab, (x + 1e-3, y), xytext=(6, 4),
                        textcoords="offset points", fontsize=8)
        # Pareto frontier line
        pareto = sorted([(x, y) for x, y, p in zip(xs, ys, mask, strict=False) if p])
        if len(pareto) > 1:
            ax.plot([p[0] + 1e-3 for p in pareto], [p[1] for p in pareto],
                    "r--", alpha=0.4, linewidth=1, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("wall clock (s, log)")
        ax.set_ylabel(args.metric)
        ax.set_title(f"N = {n}" if args.per_n else "Speed-quality Pareto")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(Path(args.path).stem, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
