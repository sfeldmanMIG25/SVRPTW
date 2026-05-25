"""svrptw.constraints — documentation catalog of opt-in cost terms.

This module is documentation, not behavior. Every constraint here is an
opt-in field on `Settings.economics` (default 0 / empty / 1.0 — bit-identical
to the un-extended baseline). To activate a constraint, set the relevant
field(s) on `Settings.economics`. See per-entry docstrings for the recipe.

Bench validation (all 6 v1_large instances, paired-seed comparison):

| iter | term                    | verdict        | mean savings/inst |
|------|-------------------------|----------------|--------------------|
| 5w   | shift_overrun           | 6/6 wins       | +$155.2           |
| 5x   | peak_hour wide windows  | 2/6 partial    | -$12.2 (geometry-bound)|
| 5y   | driver_time_variance    | 5/6 wins       | +$36.1            |
| 6a-1bis | driver_breaks tight cap| 6/6 wins    | +$179.7           |
| 6a-2 | embargo narrow windows  | 6/6 wins       | +$1,796.7         |
| 6a-3 | mixed_fleets            | 6/6 wins       | +$64.4            |
| 6a-4 | EV range                | 6/6 wins       | +$58.4            |
| 6a-6 | PD pairs                | 6/6 wins       | +$51.2            |
| 6a-7bis | skills + class_shift | 5/6 wins      | +$2,471.0         |
| 6a-8 | min_routes              | 6/6 wins       | +$300.3           |

Aggregate savings (10 stackable terms): +$5,113/inst at v1_large N=500/1000.
Combined 16-term stack at N=500 b=150s, 3-seed mean (post conditional-
registration fix, 2026-05-16): **+$816.5 +/- $399/inst**, wall_x **0.90x**
(faster than baseline -- cleaner arm pool than pre-fix +$482 @ 1.24x).
Evaluator overhead: 4.89x baseline.

Three "Step 0 remediation" operators are bundled and CONDITIONALLY registered
in the bandit's arm set only when their target constraint axis is active
(see `svrptw.solvers.classical.portfolio._filter_ops_pool`):
  - `shift_start` operator -- per-segment-of-time terms (peak_hour, embargo)
  - `class_shift` operator -- class-axis terms (mixed_fleets, skills)
  - `depot_shift` operator -- multi-depot (when `inst.depots` is set)

The conditional registration is the fix for the iter-6a-2 embargo drift
(see docs/META_RECIPE.md failure mode #3). Pruning unused arms restores per-
arm exploration time for operators that actually have work to do.

Four construction options ship in svrptw.solvers.classical:
  - `pyvrp_warm` (default) -- PyVRP HGS construction. Best end-to-end with
    bandit refinement. Requires PyVRP.
  - `fast_construct_v4` -- Solomon I1 sequential insertion. Pyvrp-independent.
    Standalone beats pyvrp by -3% cost at 3.8x faster wall on v1_large 6-inst.
    Recommended pyvrp-independent option.
  - `fast_construct_v2` -- Louvain communities + NN + merge polish. Uses OSM
    dual graph clustering. 13.7s wall at v1_large, +37% cost vs pyvrp.
  - `fast_construct_v1` -- multi-start NN + regret + polish. Quality close to
    pyvrp but suffers from 4x deadline overruns at large N (deferred fix).
  - `fast_construct_v3` -- Clarke-Wright savings (RULED OUT; 99.7% TW-reject).
Select via `construction="fast_construct_v4"` parameter on solve_auto().
"""
from __future__ import annotations


CATALOG = {
    # ─── Baseline (always-on) ───────────────────────────────────────
    "operational_cost": {
        "fields": "wage_per_hour, cost_per_mile, hard_late_penalty",
        "default": "$14.50/hr, $0.50/mi, $1000/missed",
        "category": "baseline",
        "description": "Per-route wage + distance cost + hard time-window late penalty.",
    },
    "capacity_overload": {
        "fields": "(none — inferred from vehicle_capacity)",
        "default": "always active",
        "category": "baseline",
        "description": "Penalty per unit of demand exceeding vehicle_capacity.",
    },
    # ─── Phase F (iter-5l) — lite-refactor; util/tw inline, crossings O(N^2) ──
    "crossings_penalty": {
        "fields": "crossings_penalty_per_pair",
        "default": "0.0",
        "category": "phase-F",
        "description": "Penalty per pair of inter-route segment crossings. O(N^2) per cold call BUT per-route-pair structural-hash cache (iter-6a-perf-2) gives 7.6x speedup on realistic bandit mutate-eval cycles -- shippable for production stacks.",
    },
    "util_imbalance": {
        "fields": "util_imbalance_penalty_coef",
        "default": "0.0",
        "category": "phase-F",
        "description": "Penalty proportional to coefficient-of-variation of route load utilizations. Lite path: computed inline in evaluate.",
    },
    "tw_buffer_bonus": {
        "fields": "tw_buffer_bonus_coef",
        "default": "0.0",
        "category": "phase-F",
        "description": "Reward (negative cost) for arriving close to ready time without excessive wait. Lite path: computed inline.",
    },
    # ─── iter-5v shift_overrun ───────────────────────────────────────
    "shift_overrun": {
        "fields": "shift_max_minutes, shift_overrun_penalty_per_min",
        "default": "0.0, 0.0",
        "category": "iter-5v",
        "description": "Per-route duration cap with linear penalty over. Example: 480 min cap, $1/min -> 8-hour shifts with overtime equivalent. 6/6 +$155.2.",
    },
    # ─── iter-5x peak_hour (needs shift_start operator, auto-registered) ───
    "peak_hour": {
        "fields": "peak_window_starts, peak_window_ends, peak_hour_wage_multiplier",
        "default": "(), (), 1.0",
        "category": "iter-5x",
        "description": "Wage multiplier during specified minute-of-day windows. Example: (480,1020), (600,1140), 1.5 -> 8-10am + 5-7pm at 1.5x. Needs shift_start operator (auto-registered) AND narrow windows for clean bandit response.",
    },
    # ─── iter-5y fairness ────────────────────────────────────────────
    "driver_time_variance": {
        "fields": "driver_time_variance_penalty_coef",
        "default": "0.0",
        "category": "iter-5y",
        "description": "Penalty * variance(wage_per_min * route_duration) across routes. Equalizes driver workload. Calibrate coef so baseline penalty is 3-10% of ops cost. 5/6 +$36.1.",
    },
    # ─── iter-6a-1 driver breaks ─────────────────────────────────────
    "driver_breaks": {
        "fields": "driving_max_minutes, break_violation_penalty_per_min",
        "default": "0.0, 0.0",
        "category": "iter-6a-1",
        "description": "EU 561 / US HOS lite. Per-route DRIVING-only minutes (excl service+wait) over cap. Example: 270min cap (4.5h EU), $2/min. 6/6 +$179.7 at tight cap.",
    },
    # ─── iter-6a-2 hard zones ────────────────────────────────────────
    "embargo": {
        "fields": "embargo_window_starts, embargo_window_ends, embargo_violation_penalty_per_visit",
        "default": "(), (), 0.0",
        "category": "iter-6a-2",
        "description": "Per-visit penalty for arriving inside any embargo window. Example: (480, 720), (510, 750), 50.0 -> 8-8:30am + 12-12:30pm closed, $50/visit. At-landing: 6/6 +$1,796.7. Post-conditional-registration-fix (2026-05-16): 6/6 +$1,648.4 (92% recovery of original; 2 instances actually beat the original because the cleaner arm pool + Phase F lite + crossings cache combine well).",
    },
    # ─── iter-6a-3 mixed fleets ──────────────────────────────────────
    "mixed_fleets": {
        "fields": "vehicle_class_capacities, vehicle_class_fixed_premiums, vehicle_class_per_mile_premiums",
        "default": "(), (), ()",
        "category": "iter-6a-3",
        "description": "Per-class fixed + per-mile premiums. Each route auto-assigned smallest viable class (or explicit Route.vehicle_class_idx). class_shift operator auto-registered. 6/6 +$64.4.",
    },
    # ─── iter-6a-4 EV range ──────────────────────────────────────────
    "ev_range": {
        "fields": "vehicle_range_miles, range_violation_penalty_per_mile",
        "default": "0.0, 0.0",
        "category": "iter-6a-4",
        "description": "Per-route distance cap with linear penalty per mile over. Example: 200mi range, $1/mi. 6/6 +$58.4.",
    },
    # ─── iter-6a-6 pickup-delivery ──────────────────────────────────
    "pd_pairs": {
        "fields": "pd_pairs_flat, pd_violation_penalty_per_pair",
        "default": "(), 0.0",
        "category": "iter-6a-6",
        "description": "Flat tuple (pickup_cid, delivery_cid, ...). Penalty if pair split across routes OR delivery comes before pickup. 6/6 +$51.2.",
    },
    # ─── iter-6a-7 skills (needs class_shift operator, auto-registered) ──
    "skills": {
        "fields": "customer_skill_levels_flat, vehicle_class_skill_levels, skill_mismatch_penalty_per_visit",
        "default": "(), (), 0.0",
        "category": "iter-6a-7",
        "description": "Per-customer required skill level + per-class provided level. Mismatch (cust > class) = penalty. Needs class_shift operator (auto-registered). 5/6 +$2,471.0.",
    },
    # ─── iter-6a-8 min routes ────────────────────────────────────────
    "min_routes": {
        "fields": "min_routes_required, under_min_routes_penalty_per_route",
        "default": "0, 0.0",
        "category": "iter-6a-8",
        "description": "Labor / union contract: floor on K. Penalty per route below min. 6/6 +$300.3.",
    },
    # ─── iter-6a-5 multi_depot ───────────────────────────────────────
    "multi_depot": {
        "fields": "Instance.depots (list[Depot]) + Route.depot_idx (int)",
        "default": "Instance.depots=None -> single-depot legacy",
        "category": "iter-6a-5",
        "description": "Multi-depot routing. Routes pick which depot to start/end at via Route.depot_idx. depot_shift operator auto-registered to sweep candidates. Note: travel_time matrix still single-depot indexed (per-depot OD = future work).",
    },
    # ─── Legacy ──────────────────────────────────────────────────────
    "per_route_fixed_cost": {
        "fields": "per_route_fixed_cost",
        "default": "0.0",
        "category": "legacy",
        "description": "Vehicle-day rental, driver-shift overhead. Set to realistic value (~$50-100) for K-fair cross-solver comparisons.",
    },
}


def print_catalog():
    """Print the constraint catalog in a readable table."""
    print(f"\n{'category':12s} {'term':30s} {'default':20s} {'description':50s}")
    print("-" * 130)
    for name, meta in CATALOG.items():
        cat = meta["category"]
        desc = meta["description"][:80]
        print(f"{cat:12s} {name:30s} {meta['default']:20s} {desc}")


def categories():
    """Group terms by category."""
    out = {}
    for name, meta in CATALOG.items():
        out.setdefault(meta["category"], []).append(name)
    return out
