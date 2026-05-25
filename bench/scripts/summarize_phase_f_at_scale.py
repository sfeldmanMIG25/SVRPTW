"""Quick post-hoc analysis of phase_f_at_scale_*.json.

Compares arm A (baseline) vs arm E (all 3 cost-shape coefs combined) on
v1_large. Prints per-instance Δcost and Δquality, and decides whether
arm E unifies the cost-vs-quality split.

Usage: python bench/scripts/summarize_phase_f_at_scale.py [JSON_PATH]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_PATH = "bench/runs/phase_f_at_scale_N500_AE.json"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not Path(path).exists():
        print(f"NOT_READY {path} does not exist yet")
        return 1
    blob = json.loads(Path(path).read_text())
    rows = blob.get("arm_rows") if isinstance(blob, dict) else blob
    if not rows:
        print(f"EMPTY no arm_rows")
        return 1

    by_inst_arm: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r.get("_failed"):
            continue
        by_inst_arm[(r["instance_id"], r["arm"])] = r

    instances = sorted({k[0] for k in by_inst_arm})
    arms = sorted({k[1] for k in by_inst_arm})
    print(f"=== arms={arms} instances={len(instances)} ===")
    print()
    print(f"{'instance':32s}", end="")
    for a in arms:
        print(f"  {f'{a}_cost':>9s} {f'{a}_q':>6s}", end="")
    if "A" in arms and "E" in arms:
        print(f"  {'d_cost(E-A)':>11s} {'d_q(E-A)':>9s}")
    else:
        print()
    sums = defaultdict(lambda: {"cost": 0.0, "q": 0.0, "n": 0})
    diffs = {"d_cost": 0.0, "d_q": 0.0, "n": 0}
    e_wins_cost = 0
    e_wins_q = 0
    for iid in instances:
        print(f"{iid:32s}", end="")
        ca = qa = ce = qe = None
        for a in arms:
            r = by_inst_arm.get((iid, a))
            if r is None:
                print(f"  {'-':>9s} {'-':>6s}", end="")
                continue
            cost = r.get("op_cost", r.get("operational_cost", float("nan")))
            qi = r.get("qi", r.get("quality_index", float("nan")))
            print(f"  {cost:>9.1f} {qi:>6.3f}", end="")
            sums[a]["cost"] += cost
            sums[a]["q"] += qi
            sums[a]["n"] += 1
            if a == "A":
                ca, qa = cost, qi
            elif a == "E":
                ce, qe = cost, qi
        if ca is not None and ce is not None:
            dcost = ce - ca
            dq = qe - qa
            print(f"  {dcost:>+11.1f} {dq:>+9.3f}")
            diffs["d_cost"] += dcost
            diffs["d_q"] += dq
            diffs["n"] += 1
            if dcost < 0: e_wins_cost += 1
            if dq > 0: e_wins_q += 1
        else:
            print()

    print()
    print("=== mean per-arm ===")
    for a in arms:
        s = sums[a]
        if s["n"]:
            print(f"  arm {a}: mean_cost={s['cost']/s['n']:.1f} mean_q={s['q']/s['n']:.3f} (n={s['n']})")

    if diffs["n"]:
        n = diffs["n"]
        mc = diffs["d_cost"] / n
        mq = diffs["d_q"] / n
        print()
        print(f"=== arm E vs arm A on {n} paired instances ===")
        print(f"  E wins cost: {e_wins_cost}/{n}  (d_E-A: {mc:+.1f} mean)")
        print(f"  E wins qi:   {e_wins_q}/{n}    (d_E-A: {mq:+.3f} mean)")
        if e_wins_cost == n and e_wins_q == n:
            print(f"  VERDICT: arm E STRICTLY DOMINATES arm A both axes -- unification found")
        elif e_wins_cost == 0 and e_wins_q == 0:
            print(f"  VERDICT: arm E STRICTLY DOMINATED by arm A both axes -- shaping HURTS at scale")
        elif e_wins_q >= n - 1 and mc <= 100:
            print(f"  VERDICT: arm E wins quality with acceptable cost give-back -- promising")
        else:
            print(f"  VERDICT: MIXED -- per-instance breakdown above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
