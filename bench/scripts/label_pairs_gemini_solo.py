"""Label preference pairs with a Gemini-solo-friendly committee.

OpenRouter free-tier is rate-limited too aggressively for the full
committee. This script constructs a committee with min_responders=1
and a generous max_std so Gemini-direct's solo vote counts as
authoritative. The 4 OpenRouter voters still participate when
available — their votes just aren't required.

Usage:
    python bench/scripts/label_pairs_gemini_solo.py \\
        --bench bench/runs/v1_n50_merged_for_labels.json \\
        --out data/logic/pairs_n50_gemini.json \\
        --n-pairs 30
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from svrptw.logic.committee import OpenRouterCommittee
from svrptw.logic.dataset import build_pairs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--instances", default="instances/v1")
    p.add_argument("--n-pairs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Permissive committee: Gemini-solo counts as authoritative.
    # OpenRouter voters contribute when they aren't rate-limited.
    committee = OpenRouterCommittee(
        per_model_timeout_s=25.0,
        authoritative_min_responders=1,
        authoritative_max_std=1.0,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(
        bench_path=Path(args.bench),
        out_path=Path(args.out),
        instances_dir=Path(args.instances),
        n_pairs=args.n_pairs,
        seed=args.seed,
        committee=committee,
    )
    print(f"wrote {args.out} ({len(pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
