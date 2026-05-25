"""Generate a markdown leaderboard with Pareto annotation from a bench JSON.

Usage:
    python -m svrptw.bench.leaderboard bench/runs/v1_partial.json
    python -m svrptw.bench.leaderboard --per-instance bench/runs/v1_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _pareto_mask(points: list[tuple[float, float]]) -> list[bool]:
    """Return a boolean list indicating which (cost, wall_clock) points are
    Pareto-optimal (lower is better on both axes)."""
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


def render(rows: list[dict], per_instance: bool = False,
           filter_capacity_overload: bool = True) -> str:
    """Render the leaderboard.

    `filter_capacity_overload` (default True, SPEC-0-EVAL-01) drops any
    row whose `capacity_overload > 0` from the cost ranking and notes
    the count of dropped rows in a header line. Set False only when
    auditing the impact of the bug fix.
    """
    out: list[str] = []
    dropped_overload = 0
    if filter_capacity_overload:
        kept: list[dict] = []
        for r in rows:
            if (r.get("capacity_overload") or 0.0) > 0.0:
                dropped_overload += 1
                continue
            kept.append(r)
        rows = kept

    # Aggregate by (solver, n)
    agg: dict[tuple[str, int], dict] = defaultdict(lambda: {
        "costs": [], "miss": [], "wall": [], "vehicles": [], "vivrp": []
    })
    for r in rows:
        if r.get("operational_cost") is None:
            continue
        key = (r["solver"], r["n"])
        a = agg[key]
        a["costs"].append(r["operational_cost"])
        a["miss"].append(r.get("missed_deliveries", 0))
        a["wall"].append(r.get("wall_clock_seconds", 0))
        a["vehicles"].append(r.get("num_vehicles_used", 0))
        if r.get("vivrp_overall") is not None:
            a["vivrp"].append(r["vivrp_overall"])

    # By N-bin: solver ranks + Pareto annotation
    bins = sorted({n for _, n in agg.keys()})
    out.append("# Leaderboard\n")
    out.append(f"_Generated from {len(rows)} rows across {len(bins)} N-bins._\n")
    if dropped_overload > 0:
        out.append(f"_SPEC-0-EVAL-01 filter dropped **{dropped_overload}** "
                   f"row(s) with capacity_overload > 0 from the ranking._\n")

    for n in bins:
        out.append(f"\n## N = {n}\n")
        solvers = sorted([s for (s, nn) in agg.keys() if nn == n])
        # Pareto on (mean cost, mean wall)
        pts = [(sum(agg[(s, n)]["costs"]) / len(agg[(s, n)]["costs"]),
                sum(agg[(s, n)]["wall"])  / len(agg[(s, n)]["wall"])) for s in solvers]
        mask = _pareto_mask(pts)

        cols = ["solver", "mean cost", "mean miss", "mean veh", "mean s", "n inst"]
        any_vivrp = any(agg[(s, n)]["vivrp"] for s in solvers)
        if any_vivrp:
            cols.insert(-1, "vivrp")
        cols.append("Pareto")
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")

        ordered = sorted(zip(solvers, pts, mask, strict=False), key=lambda x: x[1][0])
        for solver, (cost, wall), is_pareto in ordered:
            a = agg[(solver, n)]
            mean_miss = sum(a["miss"]) / len(a["miss"])
            mean_veh = sum(a["vehicles"]) / len(a["vehicles"])
            n_inst = len(a["costs"])
            vivrp_str = "-"
            if a["vivrp"]:
                vivrp_str = f"{sum(a['vivrp']) / len(a['vivrp']):.1f}"
            star = "*" if is_pareto else ""
            row = [f"`{solver}`{star}",
                   f"{cost:.2f}",
                   f"{mean_miss:.2f}",
                   f"{mean_veh:.1f}",
                   f"{wall:.2f}",
                   str(n_inst)]
            if any_vivrp:
                row.insert(-1, vivrp_str)
            row.append("**Y**" if is_pareto else "")
            out.append("| " + " | ".join(row) + " |")

    if per_instance:
        out.append("\n## Per-instance cost\n")
        all_solvers = sorted({r["solver"] for r in rows if r.get("operational_cost") is not None})
        by_inst: dict[str, dict[str, float]] = defaultdict(dict)
        for r in rows:
            if r.get("operational_cost") is None:
                continue
            by_inst[r["instance_id"]][r["solver"]] = r["operational_cost"]
        out.append("| instance | " + " | ".join(f"`{s}`" for s in all_solvers) + " |")
        out.append("|" + "|".join(["---"] * (len(all_solvers) + 1)) + "|")
        for iid, row in sorted(by_inst.items()):
            vals = [iid] + [
                f"{row.get(s, float('nan')):.2f}" if s in row else "-"
                for s in all_solvers
            ]
            out.append("| " + " | ".join(vals) + " |")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--per-instance", action="store_true")
    p.add_argument("--out", default=None, help="markdown output path (default stdout)")
    args = p.parse_args(argv)

    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    md = render(data["rows"], per_instance=args.per_instance)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({len(md)} chars)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
