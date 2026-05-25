"""Phase F at scale -- bandit reward shaping on v1_large N=500/1000.

iter5q closed the warmstart-source-switch composition. The remaining
composition lever for the wholesale split-win (solve_auto wins cost,
fast_construct_v2 wins quality) is bandit reward shaping: change what
"improvement" means to the bandit so the same construction lands in a
different basin.

Phase F arm E (crossings_penalty + util_imbalance + tw_buffer_bonus all
combined) showed cost+quality wins at N=50 in iter5l smoke. The open
question is whether the unification holds at v1_large N=500/1000.

This script wraps `cost_term_ablation.py` with v1_large defaults and
a longer budget (the smoke at N=50 used 10s; iter5q sweeps used 75s
at N=500 and 150s at N=1000).

Default arms: A (baseline), E (all three coefs combined). The full
A..E sweep adds 24 solves at high budget; trim to A vs E for the
unification check first.

Usage:
  PYTHONPATH=. python bench/scripts/phase_f_at_scale.py
  PYTHONPATH=. python bench/scripts/phase_f_at_scale.py --N 500  # N=500 only
  PYTHONPATH=. python bench/scripts/phase_f_at_scale.py --arms A B C D E --N both
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


_INSTANCES_BY_N: dict[str, list[str]] = {
    "500": [
        "instances/v1_large/OSM-Manhattan-N0500-I000.json",
        "instances/v1_large/OSM-Paris-N0500-I000.json",
        "instances/v1_large/OSM-SanFrancisco-N0500-I000.json",
    ],
    "1000": [
        "instances/v1_large/OSM-Manhattan-N1000-I000.json",
        "instances/v1_large/OSM-Paris-N1000-I000.json",
        "instances/v1_large/OSM-SanFrancisco-N1000-I000.json",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", choices=["500", "1000", "both"], default="500",
                    help="instance size (default: 500 only). 'both' = 6 instances.")
    ap.add_argument("--arms", nargs="+", default=["A", "E"],
                    help="arms to run (default: A E -- baseline vs all-three-coefs)")
    ap.add_argument("--budget-N500", type=float, default=75.0,
                    help="wall budget per N=500 solve (default 75s)")
    ap.add_argument("--budget-N1000", type=float, default=150.0,
                    help="wall budget per N=1000 solve (default 150s)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-pyvrp", action="store_true",
                    help="skip the PyVRP baseline reference cells")
    ap.add_argument("--out", default="bench/runs/phase_f_at_scale.json")
    ap.add_argument("--webui", action="store_true",
                    help="emit progress to webui via SVRPTW_WEBUI_URL")
    args = ap.parse_args()

    # Pick the instances + budget
    if args.N == "500":
        instances = _INSTANCES_BY_N["500"]
        budget = args.budget_N500
    elif args.N == "1000":
        instances = _INSTANCES_BY_N["1000"]
        budget = args.budget_N1000
    else:
        # 'both': N=500 and N=1000 share the same script invocation, but
        # cost_term_ablation.py uses ONE budget for all instances. Run them
        # as two separate sub-invocations and merge the outputs by hand.
        # Here we just do the N=500 set first and the N=1000 set second.
        rc1 = _run_one(args, _INSTANCES_BY_N["500"], args.budget_N500,
                       Path(args.out).with_suffix(".N500.json"))
        rc2 = _run_one(args, _INSTANCES_BY_N["1000"], args.budget_N1000,
                       Path(args.out).with_suffix(".N1000.json"))
        return rc1 or rc2

    return _run_one(args, instances, budget, Path(args.out))


def _run_one(args: argparse.Namespace, instances: list[str],
             budget: float, out: Path) -> int:
    cmd = [
        sys.executable, "bench/scripts/cost_term_ablation.py",
        "--budget", str(budget),
        "--workers", str(args.workers),
        "--arms", *args.arms,
        "--instances", *instances,
        "--out", str(out),
    ]
    if args.skip_pyvrp:
        cmd.append("--skip-pyvrp")
    if args.webui:
        cmd.append("--webui")
    print("[phase_f_at_scale]", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
