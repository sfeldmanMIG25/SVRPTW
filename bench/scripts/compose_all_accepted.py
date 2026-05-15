"""Composability bench on every LLM-accepted proposal in the corpus."""
from __future__ import annotations

import sqlite3
from svrptw.council.memory import _DB_PATH
from svrptw.council.proposal import _parse_operator
from svrptw.council.shadow_bench import composability_run


def main() -> int:
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.execute(
        "SELECT proposal_id, seed_inspired_by, code FROM proposals "
        "WHERE reject_reason IS NULL OR reject_reason LIKE 'advisory%' "
        "ORDER BY created_at DESC"
    )
    accepted = cur.fetchall()
    print(f"running composability on {len(accepted)} accepted proposals")
    results = []
    for pid, seed, code in accepted:
        op = _parse_operator(code)
        if op is None:
            print(f"  skip {pid}: failed to parse")
            continue
        print(f"\n=== {pid}  (inspired-by: {seed[:60]}) ===")
        r = composability_run(pid, op, budget_seconds=10.0, n_repeats=3)
        results.append((pid, seed, r))
        print(f"  mean_delta:    {r['mean_abs_delta']:+.2f}")
        print(f"  hit_rate:      {r['hit_rate']:.2f}")
        print(f"  worst_regress: {r['worst_regress_pct']*100:.2f}%")
        print(f"  accepted:      {r['accepted']}")

    print("\n=== summary (sorted by mean_delta) ===")
    results.sort(key=lambda x: -x[2]["mean_abs_delta"])
    for pid, seed, r in results:
        print(f"  {pid}  mean={r['mean_abs_delta']:+7.2f}  hit={r['hit_rate']:.2f}  "
              f"accept={r['accepted']}  ({seed[:60]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
