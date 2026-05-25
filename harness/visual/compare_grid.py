"""Side-by-side comparison grid — one SVG per solver, in a single HTML page.

This is the surface a human (or downstream VLM judge) inspects to decide
which solver's solution is geometrically/operationally better.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def build_grid(n: int, solvers: list[str], *,
               seed: int = 0, budget: float = 30.0,
               out_path: Path) -> dict:
    """Render every solver's solution at N, write an HTML grid + return summary.

    Returns ``{solver: {wall, K, objective, crossings, feasible, svg_inline}}``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from harness.visual.render_svg import render_for_solver

    # Reuse bench solver adapters + shared evaluator
    from examples_openvrp.bench_all_solvers import (
        SOLVERS, build_instance, shared_evaluate,
    )

    rows = []
    b = build_instance(n, seed)
    for s in solvers:
        if s not in SOLVERS:
            print(f"  WARN: unknown solver {s!r}; skipping")
            continue
        t0 = time.perf_counter()
        try:
            routes = SOLVERS[s](b, budget)
            wall = time.perf_counter() - t0
            q = shared_evaluate(routes, b)
            svg = render_for_solver(n, s, seed=seed, budget=budget)
            rows.append({
                "solver": s, "wall": wall,
                "K": int(q.get("n_routes", 0)),
                "obj": q.get("objective", 0.0),
                "crossings": int(q.get("route_crossings", 0)),
                "feasible": bool(q.get("feasible")),
                "missed": int(q.get("missed", 0)),
                "svg": svg,
            })
            print(f"  {s:20s}  wall={wall:6.2f}s  K={int(q.get('n_routes',0)):>3}  "
                  f"obj=${q.get('objective', 0):>8.2f}  "
                  f"xings={int(q.get('route_crossings',0)):>3}  "
                  f"feas={'Y' if q.get('feasible') else 'n'}")
        except Exception as e:
            print(f"  {s:20s}  FAIL: {type(e).__name__}: {e}")
            rows.append({"solver": s, "error": str(e), "svg": ""})

    # Write HTML grid
    cells = []
    for r in rows:
        if "error" in r:
            cells.append(
                f'<div class="cell err"><h3>{r["solver"]}</h3>'
                f'<p>FAIL: {_e(r["error"])}</p></div>'
            )
            continue
        meta = (f'K={r["K"]} · obj=${r["obj"]:.0f} · xings={r["crossings"]} · '
                f'wall={r["wall"]:.2f}s · {"feas" if r["feasible"] else "INFEAS"}')
        cells.append(
            f'<div class="cell">'
            f'<h3>{r["solver"]} <span class="meta">{meta}</span></h3>'
            f'<div class="svg">{r["svg"]}</div>'
            f'</div>'
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OpenVRP — solver grid N={n}</title>
<style>
:root {{ --bg:#0f1115; --fg:#e6edf3; --dim:#8b949e; --border:#30363d; --card:#161b22; }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 system-ui,sans-serif;
  margin:0; padding:18px 22px; }}
h1 {{ font-size: 18px; margin: 0 0 6px; }}
p.lede {{ color:var(--dim); margin: 0 0 18px; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 14px; }}
.cell {{ background:var(--card); border:1px solid var(--border); border-radius:6px;
  padding: 10px 12px; }}
.cell.err {{ border-color: #f85149; }}
.cell h3 {{ font-size: 13px; margin: 0 0 4px; }}
.cell .meta {{ color:var(--dim); font-weight: 400; font-size: 11px;
  font-family: "JetBrains Mono", monospace; }}
.svg {{ width: 100%; }}
.svg svg {{ width: 100%; height: auto; display: block; border-radius:4px; }}
footer {{ color:var(--dim); font-size: 11px; margin-top: 24px; text-align: center; }}
</style></head><body>
<h1>OpenVRP — solver grid · N={n} · seed={seed} · budget={budget}s</h1>
<p class="lede">Same deterministic instance solved by every solver.
Routes colored by tab10 palette (NOT load-encoded — color carries no
quality signal). Depot marked with the white square.</p>
<div class="grid">
{''.join(cells)}
</div>
<footer>Generated {time.strftime("%Y-%m-%d %H:%M")} ·
harness/visual/compare_grid.py</footer>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    summary = {r["solver"]: {k: v for k, v in r.items() if k != "svg"} for r in rows}
    return summary


def _e(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--solvers", nargs="+",
                    default=["openvrp_native", "openvrp_pyvrp",
                             "pyvrp_standalone", "ortools", "svrptw_greedy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(
        f"harness/reports/compare_n{args.n}.html")
    print(f"=== compare_grid: N={args.n}  seed={args.seed}  budget={args.budget}s "
          f"solvers={args.solvers} ===")
    summary = build_grid(args.n, args.solvers, seed=args.seed,
                         budget=args.budget, out_path=out)
    print(f"\n[wrote] {out}")
    Path("harness/reports").mkdir(parents=True, exist_ok=True)
    json_out = Path(f"harness/reports/compare_n{args.n}.json")
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[wrote] {json_out}")


if __name__ == "__main__":
    main()
