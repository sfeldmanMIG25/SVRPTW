"""Aggregate parallel sub-agent slice JSONs + regenerate the dashboard.

After the orchestrator sub-agents have each written
``harness/reports/slice_n{N}.json``, this script:
  1. Aggregates them into ``orchestrator_run.json``.
  2. Refreshes the comparison table embedded in
     ``WorldsFinestVRP/openvrp_dashboard.html`` with the latest numbers.
  3. Adds a sub-agent timings section.

Run after sub-agents complete:
  python harness/bench/render_dashboard.py --n 50 100 250
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def aggregate(report_dir: Path, n_values: list[int]) -> dict:
    rows = []
    timings = {}
    for n in n_values:
        p = report_dir / f"slice_n{n}.json"
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        timings[n] = blob.get("generated_at")
        for r in blob["results"]:
            rows.append(r)
    agg = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_values": n_values,
        "slice_timings": timings,
        "results": rows,
    }
    (report_dir / "orchestrator_run.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8")
    return agg


def render_comparison_rows(agg: dict, n_target: int) -> str:
    """Render <tr>…</tr> rows for one N column in the dashboard table."""
    by_solver = {}
    for r in agg["results"]:
        if r["n"] == n_target and r["ok"]:
            by_solver[r["solver"]] = r
    out = []
    for solver in sorted(by_solver):
        r = by_solver[solver]
        q = r["quality"]
        out.append(
            f"<tr>"
            f"<td>{solver}</td>"
            f"<td class='num mono'>{r['wall_s']:.2f}s</td>"
            f"<td class='num mono'>{r['peak_rss_mb']:.0f}MB</td>"
            f"<td class='num'>{int(q.get('n_routes', 0))}</td>"
            f"<td class='num mono'>${q.get('objective', 0):.0f}</td>"
            f"<td class='num'>{int(q.get('route_crossings', 0))}</td>"
            f"<td>{'✓' if q.get('feasible') else '✗'}</td>"
            f"</tr>"
        )
    return "\n".join(out)


def build_subagent_timings_table(agg: dict) -> str:
    """A small table showing per-slice sub-agent completion times."""
    rows = []
    for n in agg["n_values"]:
        ts = agg["slice_timings"].get(n)
        slice_results = [r for r in agg["results"] if r["n"] == n]
        ok = sum(1 for r in slice_results if r["ok"])
        total = len(slice_results)
        total_wall = sum(r["wall_s"] for r in slice_results)
        rows.append(
            f"<tr><td>sub-agent N={n}</td>"
            f"<td class='num'>{ok}/{total}</td>"
            f"<td class='num mono'>{total_wall:.1f}s</td>"
            f"<td class='mono'>{ts or '—'}</td></tr>"
        )
    return "\n".join(rows)


def update_dashboard(dashboard_path: Path, agg: dict) -> None:
    """In-place patch the existing dashboard's comparison table + add
    sub-agent timings section + harness link."""
    html = dashboard_path.read_text(encoding="utf-8")
    sa_rows = build_subagent_timings_table(agg)
    # Build per-N freshness tables
    n_summary = []
    for n in agg["n_values"]:
        rows = render_comparison_rows(agg, n)
        n_summary.append(
            f'<h3>N={n}</h3>\n<table>'
            f'<tr><th>Solver</th><th class="num">Wall</th>'
            f'<th class="num">Peak RSS</th><th class="num">K</th>'
            f'<th class="num">Objective</th><th class="num">Xings</th>'
            f'<th>Feas</th></tr>{rows}</table>'
        )
    # Link to whichever per-N visual grids actually exist on disk
    visual_links: list[str] = []
    for n in agg["n_values"]:
        p = Path(f"harness/reports/compare_n{n}.html")
        if p.exists():
            visual_links.append(
                f'<a href="../harness/reports/compare_n{n}.html">'
                f'compare_n{n}.html</a>'
            )
    visual_html = (" · ".join(visual_links)
                   if visual_links else "(not yet generated)")
    fresh_section = (
        '<!-- HARNESS-LATEST-START -->\n'
        '<h2>Harness — latest parallel sub-agent run</h2>\n'
        '<p class="muted">Each (N, solver) cell measured under sub-agent '
        'orchestration via <code>harness/bench/orchestrator.py --mode slice</code>. '
        'Sub-agents fanned out per N value; aggregator consolidates into one '
        f'run. Generated {agg["generated_at"]}. Raw data: '
        '<a href="../harness/reports/orchestrator_run.json">orchestrator_run.json</a>. '
        f'Visual side-by-side: {visual_html}.</p>\n'
        '<table><tr><th>Sub-agent</th><th class="num">OK/total</th>'
        '<th class="num">Sum wall</th><th>Completed at</th></tr>'
        f'{sa_rows}</table>\n'
        + "\n".join(n_summary) + "\n"
        '<!-- HARNESS-LATEST-END -->\n'
    )
    # Replace or append the harness-latest block
    start_marker = '<!-- HARNESS-LATEST-START -->'
    end_marker = '<!-- HARNESS-LATEST-END -->'
    if start_marker in html and end_marker in html:
        before, _, rest = html.partition(start_marker)
        _, _, after = rest.partition(end_marker)
        # Strip up to and including the end marker
        after = after.lstrip("\n")
        html = before + fresh_section + after
    else:
        # Insert before the closing footer
        marker = '<!-- Footer -->'
        if marker in html:
            html = html.replace(marker, fresh_section + "\n" + marker)
        else:
            html = html.replace('</div></body>',
                                fresh_section + "\n</div></body>")
    dashboard_path.write_text(html, encoding="utf-8")
    print(f"[updated] {dashboard_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[50, 100, 250])
    ap.add_argument("--report-dir", default="harness/reports")
    ap.add_argument("--dashboard",
                    default="WorldsFinestVRP/openvrp_dashboard.html")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    agg = aggregate(report_dir, args.n)
    print(f"[aggregated] {report_dir / 'orchestrator_run.json'}")
    print(f"  {len(agg['results'])} total results "
          f"across {len(args.n)} N values")
    dash = Path(args.dashboard)
    update_dashboard(dash, agg)


if __name__ == "__main__":
    main()
