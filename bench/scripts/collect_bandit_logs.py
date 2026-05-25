"""Phase D3 — collect bandit transitions for offline policy training.

Runs `solve_auto` over a configurable instance set with the LoggingBandit
plumbed in, then dumps every (state, action, reward) transition to a
single JSONL file. The dataset feeds `train_policy_offline.py`.

Defaults: v1 OSM N=100 + N=200 instances I003+I004 (held-out from any
tuning sweep). Override via --instances or --pattern.

Usage:
    python bench/scripts/collect_bandit_logs.py
    python bench/scripts/collect_bandit_logs.py --pattern "instances/v1/OSM-Manhattan-N100-I003.json"
    python bench/scripts/collect_bandit_logs.py --budget 20 --output runs/my_logs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

# Ensure the repo root is on sys.path when invoked as
# `python bench/scripts/collect_bandit_logs.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio as pm
from svrptw.solvers.classical import portfolio_pyvrp_warm as pwm


DEFAULT_PATTERN = "instances/v1/OSM-*-N{N}-I00{I}.json"


def _collect_default_instances(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    for N in (100, 200):
        for I in (3, 4):
            pat = str(repo_root / DEFAULT_PATTERN.format(N=N, I=I))
            out.extend(Path(p) for p in sorted(glob(pat)))
    return out


def _collect_pattern(repo_root: Path, pattern: str) -> list[Path]:
    pat = pattern if Path(pattern).is_absolute() else str(repo_root / pattern)
    return [Path(p) for p in sorted(glob(pat))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", type=str, default=None,
                   help="Glob (relative to repo root) of instance JSON files. "
                   "Defaults to v1 OSM N=100/200 I=3/4.")
    p.add_argument("--instances", nargs="*", default=None,
                   help="Explicit list of instance paths (overrides --pattern).")
    p.add_argument("--budget", type=float, default=15.0,
                   help="Per-instance solve budget, seconds. Default 15.")
    p.add_argument("--output", type=str,
                   default="bench/runs/bandit_transitions.jsonl",
                   help="JSONL output path. Parent dir is created.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape-reward", action="store_true",
                   help="Apply Phase D4 reward shaping while logging.")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.instances:
        paths = [Path(p) for p in args.instances]
    elif args.pattern:
        paths = _collect_pattern(repo_root, args.pattern)
    else:
        paths = _collect_default_instances(repo_root)

    if not paths:
        print("No instances matched. Check --pattern.", file=sys.stderr)
        return 2


    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    n_total = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for pth in paths:
            print(f"[collect] {pth.name} budget={args.budget}s ...",
                  file=sys.stderr, flush=True)
            inst = load_instance(pth)
            try:
                _ = pwm.solve_auto(
                    inst, settings, budget_seconds=args.budget,
                    bandit_kind="logging",
                    shape_reward=bool(args.shape_reward),
                    seed=args.seed,
                )
            except Exception as e:
                print(f"  ! solve failed: {e}", file=sys.stderr)
                continue
            bandit = getattr(pm, "_LAST_BANDIT", None)
            if bandit is None or not hasattr(bandit, "transitions"):
                print("  ! no LoggingBandit registered; skipping",
                      file=sys.stderr)
                continue
            for t in bandit.transitions():
                row = dict(t)
                row["instance"] = pth.name
                fh.write(json.dumps(row) + "\n")
                n_total += 1
            print(f"  -> {len(bandit.transitions())} transitions",
                  file=sys.stderr, flush=True)

    print(f"[collect] wrote {n_total} transitions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
