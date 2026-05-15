"""Construction-budget tuning at N=100 — does cb=8 generalize down from N=200?

If cb=8 wins at N=200 but cb=3 or cb=5 wins at N=100, that's a scale-
dependent parameter and we should make construction_budget adaptive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from svrptw.config import Settings
from svrptw.io import load_instance
from svrptw.solvers.classical import portfolio_pyvrp_warm as ppw


def main() -> int:
    # 4 held-out cities × I=003 (not in any tuning).
    paths = [Path(f"instances/v1/OSM-{c}-N100-I003.json")
             for c in ("Charleston", "Austin", "Pittsburgh", "Cambridge")]
    splits = (3.0, 5.0, 8.0)  # skip 10 — too much construction for 15s budget
    rows = []
    for ip in paths:
        if not ip.exists(): continue
        inst = load_instance(str(ip))
        s = Settings()
        sys.stdout.write(f"\n{inst.instance_id}:\n")
        for cb in splits:
            sol = ppw.solve(inst, s, budget_seconds=15.0,
                             pyvrp_construction_budget=cb)
            rows.append({
                "instance_id": inst.instance_id,
                "construction_budget": cb,
                "cost": sol.metrics["operational_cost"],
                "wall": sol.wall_clock_seconds,
            })
            sys.stdout.write(f"  cb={cb:>4.1f}s  cost={rows[-1]['cost']:.1f}  wall={rows[-1]['wall']:.1f}s\n")
            sys.stdout.flush()
    Path("bench/runs/cb_tune_n100.json").write_text(json.dumps(rows, indent=2))

    print("\n[N=100 cb tuning] mean cost by construction budget:")
    by = {}
    for r in rows: by.setdefault(r["construction_budget"], []).append(r)
    for cb in splits:
        rs = by.get(cb, [])
        if not rs: continue
        mc = sum(r["cost"] for r in rs) / len(rs)
        mw = sum(r["wall"] for r in rs) / len(rs)
        print(f"  cb={cb:>4.1f}s  mean_cost={mc:.1f}  mean_wall={mw:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
