"""Reproduce svrptw 0.1.0 headline numbers in one command.

Single-script validation of every claim in the README. Designed for
"contributor cold-clone validation" — run once after a fresh install,
and you get the same wholesale leaderboard + per-constraint bench wins
+ full-stack scaling proof that the docs reference.

Usage:
    PYTHONPATH=. python bench/scripts/headline_results.py [--quick]

    --quick  skip the slow 6-instance benches (&gt;=30 min wall total),
             only run the microbenchmarks and the single-instance
             stack tests. Useful for CI smoke.

Outputs:
    bench/runs/headline_results.json    machine-readable
    bench/runs/headline_results.md      human-readable summary table
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# Suite of bench scripts to run, in dependency order. Each entry:
# (label, script_path, args, expected_artifact_path, quick_safe)
SUITE = [
    # Quick microbench: evaluator overhead with all 17 cost terms ON
    ("eval_microbench", "bench/scripts/iter5u_metric_K_audit.py", [],
        "bench/runs/iter5u_metric_K_audit.json", True),
    # Single-instance full-stack solve at N=500 (3-4 min)
    ("stack16_N500_b75", "bench/scripts/iter6a_full_stack16_solve.py",
        ["instances/v1_large/OSM-Manhattan-N0500-I000.json", "75"],
        "bench/runs/iter6a_full_stack16_OSM-Manhattan-N0500-I000_b75.json", True),
    # Single-instance full-stack solve at N=1000 (5-6 min)
    ("stack16_N1000_b150", "bench/scripts/iter6a_full_stack16_solve.py",
        ["instances/v1_large/OSM-Manhattan-N1000-I000.json", "150"],
        "bench/runs/iter6a_full_stack16_OSM-Manhattan-N1000-I000_b150.json", True),
    # Wholesale 8-solver comparison (~30 min) -- SLOW
    ("wholesale_v2", "bench/scripts/wholesale_comparison.py",
        ["--solvers", "solve_auto", "pyvrp", "ortools", "lkh3",
         "fast_construct", "fast_construct_v2", "greedy", "regret_3",
         "--instance-set", "v1_large", "--workers", "4", "--budget-base", "30.0",
         "--out", "bench/runs/wholesale_v1large_headline.json"],
        "bench/runs/wholesale_v1large_headline.json", False),
    # Per-term benches (each ~7 min on v1_large)
    ("shift_overrun", "bench/scripts/iter5w_shift_overrun_v1large.py", [],
        "bench/runs/iter5w_shift_overrun_v1large.json", False),
    ("driver_breaks_tight", "bench/scripts/iter6a1_driver_breaks_v1large.py", [],
        "bench/runs/iter6a1_driver_breaks_v1large_bis.json", False),
    ("embargo", "bench/scripts/iter6a2_embargo_v1large.py", [],
        "bench/runs/iter6a2_embargo_v1large.json", False),
    ("mixed_fleets", "bench/scripts/iter6a3_mixed_fleets_v1large.py", [],
        "bench/runs/iter6a3_mixed_fleets_v1large.json", False),
    ("ev_range", "bench/scripts/iter6a4_ev_range_v1large.py", [],
        "bench/runs/iter6a4_ev_range_v1large.json", False),
    ("pd_pairs", "bench/scripts/iter6a6_pd_pairs_v1large.py", [],
        "bench/runs/iter6a6_pd_pairs_v1large.json", False),
    ("skills", "bench/scripts/iter6a7_bis_skills_v1large.py", [],
        "bench/runs/iter6a7_bis_skills_v1large.json", False),
    ("min_routes", "bench/scripts/iter6a8_min_routes_v1large.py", [],
        "bench/runs/iter6a8_min_routes_v1large.json", False),
]


def run_one(label: str, script: str, args: list[str], artifact: str) -> dict:
    t0 = time.perf_counter()
    cmd = [sys.executable, script, *args]
    try:
        res = subprocess.run(cmd, env={**__import__("os").environ,
                                       "PYTHONPATH": "."},
                              timeout=3600, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        artifact_exists = Path(artifact).exists()
        return {
            "label": label,
            "ok": res.returncode == 0 and artifact_exists,
            "exit_code": res.returncode,
            "wall_s": wall,
            "artifact": artifact,
            "artifact_exists": artifact_exists,
            "stderr_tail": (res.stderr or "")[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"label": label, "ok": False, "exit_code": -1,
                "wall_s": time.perf_counter() - t0, "artifact": artifact,
                "error": "timeout"}


def summarize(results: list[dict]) -> str:
    """Markdown table summary."""
    lines = ["# svrptw 0.1.0 headline results\n"]
    lines.append("| label | wall (s) | ok | artifact |")
    lines.append("|-------|----------|----|----------|")
    for r in results:
        ok = "OK" if r["ok"] else "X"
        lines.append(f"| {r['label']} | {r['wall_s']:.1f} | {ok} | "
                     f"`{r['artifact']}` |")
    # Pull headline metrics out of artifacts we know the shape of
    lines.append("\n## Key numbers (parsed from artifacts)\n")
    for r in results:
        if not r["ok"]: continue
        try:
            d = json.loads(Path(r["artifact"]).read_text())
            if r["label"].startswith("stack16"):
                lines.append(f"- **{r['label']}**: wall_x={d['wall_overhead_x']:.2f} "
                             f"K {d['K_baseline']}-&gt;{d['K_full']} "
                             f"net=${d['net_under_full_obj']:+.1f}/inst")
            elif r["label"] == "eval_microbench":
                lb = d.get("fixed_cost_releaderboard", {})
                if lb:
                    lines.append(f"- **eval_microbench**: solver re-leaderboard at "
                                 f"per_route_fixed_cost=$50 -- {list(lb.get('$50', {}).keys())}")
            elif r["label"] == "wholesale_v2":
                rows = d if isinstance(d, list) else []
                if rows:
                    from collections import defaultdict
                    bs = defaultdict(list)
                    for row in rows:
                        if not row.get("_failed"):
                            bs[row["solver"]].append(row["operational_cost"])
                    means = sorted((sum(c)/len(c), s) for s, c in bs.items())
                    lines.append(f"- **wholesale_v2**: top-3 by mean cost: "
                                 + ", ".join(f"{s}=${c:.0f}" for c, s in means[:3]))
            elif isinstance(d, dict) and "n_winners" in d:
                lines.append(f"- **{r['label']}**: {d['n_winners']}/{d['n_instances']} wins, "
                             f"mean ${d['total_savings']/max(1,d['n_instances']):+.1f}/inst")
        except Exception as e:
            lines.append(f"- **{r['label']}**: (parse error: {e})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Skip slow benches (&gt;=30 min). Only microbench + single-instance stack tests.")
    args = ap.parse_args()

    targets = [(label, script, a, art) for label, script, a, art, qsafe in SUITE
               if args.quick is False or qsafe]
    print(f"=== headline_results: {len(targets)} benches "
          f"({'quick mode' if args.quick else 'full mode'}) ===\n")

    results = []
    t0_total = time.perf_counter()
    for i, (label, script, a, art) in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {label} ...", flush=True)
        r = run_one(label, script, a, art)
        results.append(r)
        status = "OK" if r["ok"] else f"FAIL (exit {r['exit_code']})"
        print(f"   -> {status} in {r['wall_s']:.1f}s\n", flush=True)

    Path("bench/runs/headline_results.json").write_text(
        json.dumps({"results": results,
                    "wall_total_s": time.perf_counter() - t0_total,
                    "quick": args.quick}, indent=2))
    Path("bench/runs/headline_results.md").write_text(summarize(results))
    print(f"\nWrote bench/runs/headline_results.{{json,md}} "
          f"in {time.perf_counter() - t0_total:.0f}s total.")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
