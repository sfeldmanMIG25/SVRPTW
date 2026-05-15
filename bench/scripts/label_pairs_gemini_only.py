"""Label preference pairs with Gemini DIRECT only — bypass OpenRouter.

After observing 20+ minute hangs on the OpenRouter free tier (429
loops far exceeding per-model timeouts), this script constructs a
committee with ALL OpenRouter tiers empty and only Gemini-direct
active. Each call is ~5-10 s, so 15 pairs × 2 cells = ~3-5 min total.

Single-judge teacher per SPEC-6-LOGIC-01's fallback path.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from svrptw.logic.committee import OpenRouterCommittee, _TierConfig
from svrptw.logic.dataset import build_pairs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--instances", default="instances/v1")
    p.add_argument("--n-pairs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    empty = _TierConfig("X", (), 0.0, False)
    committee = OpenRouterCommittee(
        tiers_primary=(empty,),
        tier_stealth=None,
        tier_tiebreaker=empty,
        per_model_timeout_s=15.0,
        authoritative_min_responders=1,
        authoritative_max_std=1.0,
        include_gemini=True,
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
