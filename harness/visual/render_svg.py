"""SVG renderer for a single solution — per-route colored, no GUI deps.

Output is a standalone .svg file readable by any browser, suitable for
human review or downstream VLM judging. Routes use the tab10 palette
so colors are distinguishable but NOT load-encoded — a VLM can't be
fooled into reading load from hue.

Use as a library:

    from harness.visual.render_svg import render_solution_svg
    svg = render_solution_svg(coords, routes, depot_xy, title="N=100 openvrp_native")
    open('out.svg', 'w').write(svg)

Or from the CLI:

    python harness/visual/render_svg.py --n 100 --solver openvrp_native \\
        --out out.svg
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# tab10-style palette — visible on dark + light bg, 10 distinct hues
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def render_solution_svg(coords: list[tuple[float, float]],
                        routes: list[list[int]],
                        *,
                        title: str = "",
                        subtitle: str = "",
                        width: int = 600,
                        height: int = 600,
                        bg: str = "#0f1115",
                        fg: str = "#e6edf3",
                        dim: str = "#8b949e") -> str:
    """Render a single solution as a self-contained SVG string.

    ``coords[0]`` is the depot; ``coords[1..N]`` are customers.
    ``routes`` is a list of customer-id sequences (each is 1..N).
    """
    if not coords:
        return "<svg/>"
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = max(1e-6, x_max - x_min)
    span_y = max(1e-6, y_max - y_min)
    pad = 30
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad - 40    # leave 40px at top for titles

    def to_px(c: tuple[float, float]) -> tuple[float, float]:
        x = pad + (c[0] - x_min) / span_x * plot_w
        # Flip y so positive y is up
        y = pad + 40 + plot_h - (c[1] - y_min) / span_y * plot_h
        return (x, y)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet" width="{width}" height="{height}" '
        f'style="background:{bg};font-family:system-ui,sans-serif;">'
    )
    parts.append(
        f'<text x="{pad}" y="22" fill="{fg}" font-size="14" font-weight="600">'
        f'{_escape(title)}</text>'
    )
    if subtitle:
        parts.append(
            f'<text x="{pad}" y="38" fill="{dim}" font-size="11">'
            f'{_escape(subtitle)}</text>'
        )

    # Routes as polylines
    depot_px = to_px(coords[0])
    for r_idx, seq in enumerate(routes):
        if not seq:
            continue
        color = PALETTE[r_idx % len(PALETTE)]
        path_pts = [depot_px] + [to_px(coords[c]) for c in seq] + [depot_px]
        d = " ".join(f"{x:.1f},{y:.1f}" for (x, y) in path_pts)
        parts.append(
            f'<polyline points="{d}" stroke="{color}" stroke-width="1.6" '
            f'fill="none" opacity="0.85"/>'
        )
        # Customer points along this route
        for c in seq:
            cx, cy = to_px(coords[c])
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
                f'fill="{color}" stroke="{bg}" stroke-width="0.5"/>'
            )

    # Depot marker (always last so it sits on top)
    dx, dy = depot_px
    parts.append(
        f'<rect x="{dx - 5:.1f}" y="{dy - 5:.1f}" width="10" height="10" '
        f'fill="{fg}" stroke="{bg}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{dx + 8:.1f}" y="{dy + 3:.1f}" fill="{dim}" font-size="9">D</text>'
    )

    parts.append('</svg>')
    return "".join(parts)


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def render_for_solver(n: int, solver: str, *,
                      seed: int = 0, budget: float = 30.0,
                      out_path: Path | None = None) -> str:
    """Build an instance, run one solver, render result to SVG.

    Imports the bench wrapper so the same generator + solver adapters
    are reused.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    # Reuse the canonical bench instance + solver adapters
    from examples_openvrp.bench_all_solvers import (
        SOLVERS, build_instance, shared_evaluate,
    )
    b = build_instance(n, seed)
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; options: {sorted(SOLVERS)}")
    routes = SOLVERS[solver](b, budget)
    q = shared_evaluate(routes, b)
    # Reconstruct deterministic xy coords (must match bench_all_solvers.build_instance)
    import numpy as np
    rng = np.random.default_rng(seed)
    coords = [(0.0, 0.0)]
    for _ in range(n):
        coords.append((float(rng.uniform(-50, 50)),
                       float(rng.uniform(-50, 50))))
    subtitle = (f"K={int(q.get('n_routes', 0))}  "
                f"obj=${q.get('objective', 0):.0f}  "
                f"crossings={int(q.get('route_crossings', 0))}  "
                f"feas={'Y' if q.get('feasible') else 'n'}")
    svg = render_solution_svg(coords, routes,
                              title=f"{solver}  N={n}", subtitle=subtitle)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(svg, encoding="utf-8")
    return svg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--solver", default="openvrp_native")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--out", default="harness/reports/solution.svg")
    args = ap.parse_args()
    render_for_solver(args.n, args.solver, seed=args.seed,
                       budget=args.budget, out_path=Path(args.out))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
