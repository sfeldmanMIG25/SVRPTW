"""Run composability bench on the LLM-accepted proposal stored in the corpus."""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

from svrptw.council.memory import _DB_PATH
from svrptw.council.proposal import _parse_operator
from svrptw.council.shadow_bench import composability_run


def main() -> int:
    pid = sys.argv[1] if len(sys.argv) > 1 else "p_1153d0865298"
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.execute("SELECT code FROM proposals WHERE proposal_id=?", (pid,))
    row = cur.fetchone()
    if row is None:
        print(f"no such proposal: {pid}")
        return 1
    code = row[0]
    op = _parse_operator(code)
    if op is None:
        print(f"could not parse proposal {pid}")
        return 1
    print(f"composability bench on {pid} ...")
    r = composability_run(pid, op, budget_seconds=10.0, n_repeats=3)
    print(f"\nmean_delta:    {r['mean_abs_delta']:+.2f}")
    print(f"hit_rate:      {r['hit_rate']:.2f}")
    print(f"worst_regress: {r['worst_regress_pct']*100:.2f}%")
    print(f"accepted:      {r['accepted']}")
    print(f"reasoning:     {r['reasoning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
