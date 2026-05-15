"""Multi-objective Pareto stack: non-dominated front, hypervolume,
IGD+, R2, lex-then-HV ranking.  SPEC-5-PARETO-01.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Each objective is (key, minimize?)
@dataclass(frozen=True)
class Objective:
    key: str
    minimize: bool = True


def _to_min_matrix(rows: list[dict], objs: list[Objective]) -> tuple[np.ndarray, list[int]]:
    """Return (M, valid_idx) where M is the (n_valid, d) array of
    minimization-form objective values (max-objectives are negated).
    Skips rows missing any objective."""
    valid: list[int] = []
    out: list[list[float]] = []
    for i, r in enumerate(rows):
        if any(r.get(o.key) is None for o in objs):
            continue
        vec = [float(r[o.key]) if o.minimize else -float(r[o.key]) for o in objs]
        out.append(vec)
        valid.append(i)
    return np.asarray(out, dtype=np.float64), valid


def nondominated(rows: list[dict], objs: list[Objective]) -> list[int]:
    """Return indices (into `rows`) that are Pareto non-dominated."""
    M, valid = _to_min_matrix(rows, objs)
    if M.shape[0] == 0:
        return []
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    fronts = NonDominatedSorting().do(M, only_non_dominated_front=True)
    return [valid[i] for i in fronts.tolist()]


def hypervolume(rows: list[dict], objs: list[Objective],
                ref: list[float] | None = None) -> float:
    """Hypervolume of `rows` w.r.t. a reference point (max + 10% per axis)."""
    M, _ = _to_min_matrix(rows, objs)
    if M.shape[0] == 0:
        return 0.0
    if ref is None:
        ref = (M.max(axis=0) * 1.1).tolist()
    ref_arr = np.asarray(ref, dtype=np.float64)
    from pymoo.indicators.hv import HV
    return float(HV(ref_point=ref_arr).do(M))


def igd_plus(rows: list[dict], reference_rows: list[dict],
             objs: list[Objective]) -> float:
    """IGD+ vs. a known reference front (lower is better)."""
    M, _ = _to_min_matrix(rows, objs)
    R, _ = _to_min_matrix(reference_rows, objs)
    if M.shape[0] == 0 or R.shape[0] == 0:
        return float("inf")
    from pymoo.indicators.igd_plus import IGDPlus
    return float(IGDPlus(R).do(M))


def lex_then_hv_rank(rows: list[dict], hard_zero: list[str],
                     objs: list[Objective]) -> list[int]:
    """Single-decision ranking: lex-order by hard constraints (must be 0),
    then by HV-contribution within survivors.  Returns indices ranked
    best-first."""
    # 1) Lex filter: rows where every hard_zero key is 0 (or absent).
    def is_hard_clean(r: dict) -> bool:
        return all((r.get(k, 0) or 0) == 0 for k in hard_zero)

    clean_idx = [i for i, r in enumerate(rows) if is_hard_clean(r)]
    dirty_idx = [i for i, r in enumerate(rows) if not is_hard_clean(r)]

    # 2) Among clean rows, rank by HV contribution (drop-one effect on HV).
    if len(clean_idx) <= 1:
        return clean_idx + dirty_idx

    clean_rows = [rows[i] for i in clean_idx]
    M, _ = _to_min_matrix(clean_rows, objs)
    if M.shape[0] == 0:
        return clean_idx + dirty_idx
    ref = (M.max(axis=0) * 1.1).tolist()
    from pymoo.indicators.hv import HV
    hv = HV(ref_point=np.asarray(ref))
    full_hv = float(hv.do(M))
    contribs = []
    for k in range(len(clean_idx)):
        mask = np.ones(M.shape[0], dtype=bool)
        mask[k] = False
        sub = M[mask]
        contribs.append((full_hv - float(hv.do(sub)) if sub.size else full_hv, clean_idx[k]))
    contribs.sort(reverse=True)
    ranked_clean = [i for _, i in contribs]
    return ranked_clean + dirty_idx


def _aggregate_by_solver(rows: list[dict], objs: list[Objective]) -> list[dict]:
    """Collapse multi-instance rows into one mean row per solver."""
    by_solver: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_solver[r["solver"]].append(r)
    agg: list[dict] = []
    for solver, rs in by_solver.items():
        a: dict = {"solver": solver, "n_inst": len(rs)}
        for o in objs:
            vals = [float(r[o.key]) for r in rs if r.get(o.key) is not None]
            a[o.key] = sum(vals) / max(1, len(vals))
        # Carry through hard-constraint keys as means too (so lex filter sees them).
        for hk in ("missed_deliveries", "tw_late_minutes"):
            vals = [float(r[hk]) for r in rs if r.get(hk) is not None]
            if vals:
                a[hk] = sum(vals) / len(vals)
        agg.append(a)
    return agg


def render_report(rows: list[dict],
                  objs: list[Objective] | None = None,
                  hard_zero: list[str] | None = None) -> str:
    objs = objs or [
        Objective("operational_cost", minimize=True),
        Objective("wall_clock_seconds", minimize=True),
        Objective("missed_deliveries", minimize=True),
        Objective("vivrp_overall", minimize=False),
    ]
    hard_zero = hard_zero or ["missed_deliveries", "tw_late_minutes"]

    # Filter to rows that have at least the cost objective.
    rows = [r for r in rows if r.get("operational_cost") is not None]

    # Per N-bin section.
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)
    lines: list[str] = ["# Pareto report\n",
                        f"_Generated from {len(rows)} rows across {len(by_n)} N-bins._\n"]

    for n in sorted(by_n.keys()):
        agg = _aggregate_by_solver(by_n[n], objs)
        # Keep only objectives that (a) appear in every solver row and
        # (b) have non-trivial range across solvers.
        present = []
        for o in objs:
            vals = [a[o.key] for a in agg if o.key in a]
            if len(vals) < len(agg):
                continue
            if max(vals) - min(vals) < 1e-9:
                continue  # zero range — adds no Pareto information
            present.append(o)
        if not present:
            lines.append(f"\n## N = {n}\n_no varying objectives present._\n")
            continue
        front_idx = nondominated(agg, present)
        ref_vals = [
            max(float(a[o.key]) if o.minimize else -float(a[o.key]) for a in agg)
            for o in present
        ]
        # Ensure strictly positive reference for HV math even when some axis is near 0.
        ref_pt = [max(rv * 1.1, rv + 1.0) for rv in ref_vals]
        hv = hypervolume(agg, present, ref=ref_pt)
        ranked = lex_then_hv_rank(agg, hard_zero, present)

        lines.append(f"\n## N = {n}\n")
        cols = ["solver", "n_inst"] + [o.key for o in present] + ["Pareto"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for j in ranked:
            a = agg[j]
            row_vals = [f"`{a['solver']}`", str(a["n_inst"])]
            for o in present:
                v = a.get(o.key)
                row_vals.append(f"{v:.2f}" if v is not None else "-")
            row_vals.append("**Y**" if j in front_idx else "")
            lines.append("| " + " | ".join(row_vals) + " |")
        lines.append(f"\n**Hypervolume:** {hv:.4g}   (lower-cost & higher-quality is better)\n")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    md = render_report(data["rows"])
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({len(md)} chars)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
