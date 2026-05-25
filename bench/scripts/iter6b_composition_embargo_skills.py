"""iter-6b composition micro-bench: embargo (time-axis, shift_start operator)
+ skills (class-axis, class_shift operator).

Question: does (embargo+skills)_aware - baseline ~= (embargo - baseline) +
(skills - baseline) under bandit-allocated budget? Linear = clean composition;
sub-linear = bandit time competition; super-linear = complementary axes.

4 modes per instance: baseline, embargo, skills, both. Each mode evaluated
under the joint objective (both terms on). Bandit gets the matching settings.
"""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

OUT = "bench/runs/iter6b_composition_embargo_skills.json"
INSTANCES = [
    ("instances/v1_large/OSM-Manhattan-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Paris-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-SanFrancisco-N0500-I000.json", 75.0),
    ("instances/v1_large/OSM-Manhattan-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-Paris-N1000-I000.json", 150.0),
    ("instances/v1_large/OSM-SanFrancisco-N1000-I000.json", 150.0),
]
EMBARGO_STARTS = (480, 720)
EMBARGO_ENDS = (510, 750)
EMBARGO_PEN = 50.0


def _make_settings(inst, mode: str):
    """Return a Settings configured per mode. 'joint' enables both terms."""
    from svrptw.config import Settings
    s = Settings()
    cap = float(inst.vehicle_capacity)
    # skills requires mixed_fleets to be meaningful; always set the class
    # capacities so the K assignment is consistent across modes.
    s.economics.vehicle_class_capacities = (cap * 0.5, cap)
    s.economics.vehicle_class_fixed_premiums = (0.0, 0.0)
    s.economics.vehicle_class_per_mile_premiums = (0.0, 0.0)
    if mode in ("embargo", "both"):
        s.economics.embargo_window_starts = EMBARGO_STARTS
        s.economics.embargo_window_ends = EMBARGO_ENDS
        s.economics.embargo_violation_penalty_per_visit = EMBARGO_PEN
    if mode in ("skills", "both"):
        pairs = []
        for cid in range(1, 31):
            pairs.append(cid); pairs.append(2)
        s.economics.customer_skill_levels_flat = tuple(pairs)
        s.economics.vehicle_class_skill_levels = (0, 2)
        s.economics.skill_mismatch_penalty_per_visit = 100.0
    return s


def _run_one(task):
    p, mode, b = task
    from svrptw.io import load_instance
    from svrptw.solvers.classical import portfolio_pyvrp_warm as pw
    from svrptw.solvers.common.solution import evaluate
    inst = load_instance(p)
    # Solver gets per-mode settings (so embargo-only solver sees no skills cost)
    s_solve = _make_settings(inst, mode)
    # All evaluated under JOINT objective for fair cost comparison
    s_joint = _make_settings(inst, "both")
    s_base_eval = _make_settings(inst, "baseline")
    t0 = time.perf_counter()
    sol = pw.solve_auto(inst, s_solve, budget_seconds=b, seed=0)
    wall = time.perf_counter() - t0
    cost_base = evaluate(inst, sol, s_base_eval)["operational_cost"]
    cost_joint = evaluate(inst, sol, s_joint)["operational_cost"]
    return {
        "instance_id": inst.instance_id, "mode": mode,
        "K": int(sol.metrics["num_vehicles_used"]),
        "cost_no_terms": cost_base,
        "cost_joint_eval": cost_joint,
        "wall_s": wall, "budget_s": b,
    }


def main():
    tasks = [(p, m, b) for p, b in INSTANCES for m in ("baseline", "embargo", "skills", "both")]
    print(f"=== iter-6b composition: embargo+skills, {len(tasks)} solves ===")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            print(f"  [{len(rows):>2}/{len(tasks)}] {r['mode']:8s} on {r['instance_id']:32s} K={r['K']:>3d} joint=${r['cost_joint_eval']:>9.1f} wall={r['wall_s']:>5.1f}s", flush=True)
    by = {}
    for r in rows: by.setdefault(r["instance_id"], {})[r["mode"]] = r
    n_lin = 0; n_super = 0; n_sub = 0
    sum_delta_emb = 0.0; sum_delta_skl = 0.0; sum_delta_both = 0.0
    n = 0
    print("\n=== composition analysis (delta = baseline - mode, all eval'd under joint) ===")
    print(f"  {'instance':32s} {'d_emb':>10s} {'d_skl':>10s} {'sum':>10s} {'d_both':>10s} {'verdict':>10s}")
    for iid in sorted(by):
        m = by[iid]
        if not all(k in m for k in ("baseline", "embargo", "skills", "both")):
            continue
        base = m["baseline"]["cost_joint_eval"]
        d_emb = base - m["embargo"]["cost_joint_eval"]
        d_skl = base - m["skills"]["cost_joint_eval"]
        d_both = base - m["both"]["cost_joint_eval"]
        d_sum = d_emb + d_skl
        # Verdict relative to 10% tolerance band
        tol = 0.10 * max(abs(d_sum), 1.0)
        if d_both > d_sum + tol: verdict = "SUPER"; n_super += 1
        elif d_both < d_sum - tol: verdict = "SUB"; n_sub += 1
        else: verdict = "LINEAR"; n_lin += 1
        sum_delta_emb += d_emb; sum_delta_skl += d_skl; sum_delta_both += d_both; n += 1
        print(f"  {iid:32s} {d_emb:>+10.1f} {d_skl:>+10.1f} {d_sum:>+10.1f} {d_both:>+10.1f} {verdict:>10s}")
    if n > 0:
        avg_emb = sum_delta_emb / n; avg_skl = sum_delta_skl / n
        avg_sum = avg_emb + avg_skl; avg_both = sum_delta_both / n
        ratio = avg_both / avg_sum if avg_sum != 0 else float('nan')
        print(f"\n  {'AGGREGATE':32s} {avg_emb:>+10.1f} {avg_skl:>+10.1f} {avg_sum:>+10.1f} {avg_both:>+10.1f}")
        print(f"\n  composition ratio (d_both / (d_emb+d_skl)) = {ratio:.3f}")
        print(f"  verdict counts: LINEAR={n_lin} SUPER={n_super} SUB={n_sub}")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps({
        "embargo_starts": list(EMBARGO_STARTS), "embargo_ends": list(EMBARGO_ENDS),
        "embargo_pen": EMBARGO_PEN, "rows": rows,
        "n_instances": n, "n_linear": n_lin, "n_super": n_super, "n_sub": n_sub,
        "avg_d_emb": sum_delta_emb / max(1, n),
        "avg_d_skl": sum_delta_skl / max(1, n),
        "avg_d_both": sum_delta_both / max(1, n),
        "wall_total_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
