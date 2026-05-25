"""SPEC-6-PARETO-3AXIS-01 — three-axis (cost, time, logic) Pareto dominance.

Headline metric: fraction of v1 instances where our challenger solver
strictly Pareto-dominates a baseline (e.g. PyVRP) under the three
axes. The logic axis drops out per-instance when the ensemble is
non-authoritative — uncertainty-aware reporting.

Pairs with the existing two-axis `svrptw.bench.pareto` module; that
one stays for analysis frontiers, this one writes the headline number.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

_LOG = logging.getLogger("svrptw.bench.pareto3")

# Per-axis tolerance for "strictly better" (1 % relative).
_TOL = 0.01


@dataclass
class DominanceReport:
    """Result of comparing one challenger against one baseline."""

    challenger: str
    baseline: str
    axes: tuple[str, ...]
    minimise: tuple[bool, ...]
    headline: float            # fraction of instances dominated
    n_total: int
    n_dominated: int
    n_ties: int
    logic_dropped: int          # instances where logic axis was dropped
    per_n: dict[int, dict] = field(default_factory=dict)
    detail: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "challenger": self.challenger,
            "baseline": self.baseline,
            "axes": list(self.axes),
            "minimise": list(self.minimise),
            "headline": self.headline,
            "n_total": self.n_total,
            "n_dominated": self.n_dominated,
            "n_ties": self.n_ties,
            "logic_dropped": self.logic_dropped,
            "per_n": self.per_n,
            "detail": self.detail,
        }


def _better(a: float, b: float, minimise: bool, tol: float = _TOL) -> bool:
    """Strictly better with 1 % relative tolerance."""
    if minimise:
        return a < b * (1.0 - tol)
    return a > b * (1.0 + tol)


def _at_least(a: float, b: float, minimise: bool, tol: float = _TOL) -> bool:
    """Better-or-equal within 1 % relative tolerance."""
    if minimise:
        return a <= b * (1.0 + tol)
    return a >= b * (1.0 - tol)


def _dominates(
    challenger: tuple[Optional[float], ...],
    baseline: tuple[Optional[float], ...],
    minimise: tuple[bool, ...],
) -> tuple[bool, int]:
    """Returns (challenger strictly dominates baseline, num_axes_used).

    Any None on either side drops that axis from the comparison (this is
    how the logic axis falls out when the ensemble is non-authoritative).
    """
    axes_used = 0
    at_least_all = True
    strictly_one = False
    for c, b, mn in zip(challenger, baseline, minimise):
        if c is None or b is None:
            continue
        axes_used += 1
        if not _at_least(c, b, mn):
            at_least_all = False
            break
        if _better(c, b, mn):
            strictly_one = True
    if axes_used == 0:
        return (False, 0)
    return (at_least_all and strictly_one, axes_used)


def _row_logic_authoritative(row: dict) -> bool:
    """Default rule: a row's logic axis is authoritative iff `logic_score`
    is present and `logic_authoritative` is missing-or-True."""
    if "logic_score" not in row or row["logic_score"] is None:
        return False
    auth = row.get("logic_authoritative", True)
    return bool(auth)


def _extract(row: dict, axes: tuple[str, ...], logic_auth: bool
             ) -> tuple[Optional[float], ...]:
    out: list[Optional[float]] = []
    for ax in axes:
        if ax == "logic_score" and not logic_auth:
            out.append(None)
        else:
            v = row.get(ax)
            out.append(float(v) if v is not None else None)
    return tuple(out)


def dominance_report(
    rows: Iterable[dict],
    *,
    challenger: str,
    baseline: str,
    axes: tuple[str, ...] = ("operational_cost", "wall_clock_seconds", "logic_score"),
    minimise: tuple[bool, ...] = (True, True, False),
    capacity_overload_filter: bool = True,
) -> DominanceReport:
    """Compute the fraction of instances where `challenger` strictly
    Pareto-dominates `baseline` under the given axes.

    `capacity_overload_filter` drops rows with capacity_overload > 0 from
    consideration (post SPEC-0-EVAL-01). Defaults to on — required when
    comparing solvers where some produce overloaded routes.
    """
    rows = list(rows)
    if capacity_overload_filter:
        rows = [r for r in rows if (r.get("capacity_overload") or 0.0) == 0.0]

    by_inst: dict[str, dict[str, dict]] = {}
    for r in rows:
        iid = r["instance_id"]
        by_inst.setdefault(iid, {})[r["solver"]] = r

    n_total = 0
    n_dominated = 0
    n_ties = 0
    logic_dropped = 0
    per_n: dict[int, dict] = {}
    detail: list[dict] = []

    for iid, by_solver in sorted(by_inst.items()):
        c_row = by_solver.get(challenger)
        b_row = by_solver.get(baseline)
        if c_row is None or b_row is None:
            continue
        # Logic-axis drop is per-instance: drop if EITHER side is not authoritative.
        c_auth = _row_logic_authoritative(c_row)
        b_auth = _row_logic_authoritative(b_row)
        logic_axis_dropped = ("logic_score" in axes) and not (c_auth and b_auth)
        if logic_axis_dropped:
            logic_dropped += 1
        c_vals = _extract(c_row, axes, c_auth and b_auth)
        b_vals = _extract(b_row, axes, c_auth and b_auth)
        dominates, axes_used = _dominates(c_vals, b_vals, minimise)
        baseline_dominates, _ = _dominates(b_vals, c_vals, minimise)
        if axes_used == 0:
            continue
        n_total += 1
        if dominates:
            n_dominated += 1
            outcome = "dominates"
        elif baseline_dominates:
            outcome = "dominated_by_baseline"
        else:
            outcome = "tie"
            n_ties += 1

        n_bin = int(c_row.get("n", 0))
        per_n.setdefault(n_bin, {"n": 0, "dom": 0, "tie": 0, "logic_drop": 0})
        per_n[n_bin]["n"] += 1
        if dominates:
            per_n[n_bin]["dom"] += 1
        elif outcome == "tie":
            per_n[n_bin]["tie"] += 1
        if logic_axis_dropped:
            per_n[n_bin]["logic_drop"] += 1

        detail.append({
            "instance_id": iid,
            "n": n_bin,
            "axes_used": axes_used,
            "logic_axis_dropped": logic_axis_dropped,
            "outcome": outcome,
            "challenger": {ax: c_row.get(ax) for ax in axes},
            "baseline":   {ax: b_row.get(ax) for ax in axes},
        })

    headline = (n_dominated / n_total) if n_total > 0 else float("nan")
    # per_n: fold to ratios
    for v in per_n.values():
        n = max(v["n"], 1)
        v["headline"] = v["dom"] / n

    return DominanceReport(
        challenger=challenger,
        baseline=baseline,
        axes=axes,
        minimise=minimise,
        headline=headline,
        n_total=n_total,
        n_dominated=n_dominated,
        n_ties=n_ties,
        logic_dropped=logic_dropped,
        per_n=per_n,
        detail=detail,
    )


def write_headline(report: DominanceReport, out_path: Path) -> None:
    """One-line dispatcher-acceptance headline. Bit-stable for repro."""
    pct = report.headline * 100.0 if not math.isnan(report.headline) else float("nan")
    axes_str = ", ".join(report.axes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"_Our solver Pareto-dominates **{report.baseline}** under "
        f"({axes_str}) on **{pct:.1f}%** of v1 instances "
        f"({report.n_dominated}/{report.n_total}; "
        f"logic axis dropped on {report.logic_dropped})._\n"
    )
    out_path.write_text(line, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True, help="Path to bench rows JSON")
    p.add_argument("--challenger", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--axes", default="operational_cost,wall_clock_seconds,logic_score")
    p.add_argument("--minimise", default="true,true,false",
                   help="Comma-separated bool flags matching axes")
    p.add_argument("--out", default="bench/figures/v1_pareto_3axis_report.json")
    p.add_argument("--headline-out", default="bench/figures/HEADLINE.md")
    args = p.parse_args(argv)

    data = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data

    axes = tuple(s.strip() for s in args.axes.split(","))
    minimise = tuple(s.strip().lower() == "true" for s in args.minimise.split(","))
    if len(axes) != len(minimise):
        raise SystemExit("--axes and --minimise must have the same length")

    report = dominance_report(
        rows,
        challenger=args.challenger,
        baseline=args.baseline,
        axes=axes,
        minimise=minimise,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))
    write_headline(report, Path(args.headline_out))
    print(f"wrote {args.out}")
    print(f"wrote {args.headline_out}")
    print(f"headline: {report.headline*100:.1f}% "
          f"({report.n_dominated}/{report.n_total}, ties={report.n_ties}, "
          f"logic-dropped={report.logic_dropped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
