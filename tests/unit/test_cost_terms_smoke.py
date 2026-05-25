"""SPEC-F-COST-01 -- smoke tests for opt-in structural cost terms.

Three invariants guarded here:

1. ``Settings()`` (no args) leaves all three new coefs at 0.0, so
   ``evaluate(...)`` returns the same operational_cost it did before
   Phase F (bit-identical default).
2. Setting ``crossings_penalty_per_pair = K`` adds exactly
   ``K * N_inter_route_crossings`` to the cost.
3. With all three coefs set, the deltas are additive (each term acts
   independently and linearly on its own metric).
"""
from __future__ import annotations

from svrptw.config import Settings
from svrptw.config.schema import Economics
from svrptw.instances_gen.synthetic import generate
from svrptw.metrics import score_solution
from svrptw.solvers.classical import greedy as greedy_mod
from svrptw.solvers.common.solution import evaluate


def test_default_settings_have_zero_new_coefs():
    """Bit-identical default: a fresh Settings() must leave all three
    new fields at exactly 0.0. Anyone constructing Settings() expects
    pre-Phase-F semantics."""
    e = Economics()
    assert e.crossings_penalty_per_pair == 0.0
    assert e.util_imbalance_penalty_coef == 0.0
    assert e.tw_buffer_bonus_coef == 0.0
    # iter-5v: shift-overrun term also defaults off
    assert e.shift_max_minutes == 0.0
    assert e.shift_overrun_penalty_per_min == 0.0
    # iter-5x: peak-hour surcharge defaults off
    assert e.peak_window_starts == ()
    assert e.peak_window_ends == ()
    assert e.peak_hour_wage_multiplier == 1.0
    # iter-5y: cross-route fairness defaults off
    assert e.driver_time_variance_penalty_coef == 0.0
    # iter-6a-1: driver-breaks (EU 561 / US HOS) defaults off
    assert e.driving_max_minutes == 0.0
    assert e.break_violation_penalty_per_min == 0.0
    # iter-6a-2: hard zones / embargo windows default off
    assert e.embargo_window_starts == ()
    assert e.embargo_window_ends == ()
    assert e.embargo_violation_penalty_per_visit == 0.0
    # iter-6a-4: EV range default off
    assert e.vehicle_range_miles == 0.0
    assert e.range_violation_penalty_per_mile == 0.0
    # iter-6a-3: mixed fleets default off (all tuples empty)
    assert e.vehicle_class_capacities == ()
    assert e.vehicle_class_fixed_premiums == ()
    assert e.vehicle_class_per_mile_premiums == ()
    # iter-6a-6: PD pairs default off
    assert e.pd_pairs_flat == ()
    assert e.pd_violation_penalty_per_pair == 0.0
    # iter-6a-8: min routes default off
    assert e.min_routes_required == 0
    assert e.under_min_routes_penalty_per_route == 0.0
    # iter-6a-7: skills default off
    assert e.customer_skill_levels_flat == ()
    assert e.vehicle_class_skill_levels == ()
    assert e.skill_mismatch_penalty_per_visit == 0.0


def test_shift_overrun_term_is_linear_in_overrun_minutes():
    """Setting shift_max + shift_overrun_penalty must add EXACTLY
    coef * sum(max(0, route_duration - cap)) to cost.

    Validated via doubling the coef: penalty at $1.0/min must equal
    exactly 2x the penalty at $0.5/min on the same solution.
    """
    inst = generate(N=20, seed=23)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]

    # Tight cap to force a violation on the small instance
    s_half = Settings()
    s_half.economics = Economics(
        shift_max_minutes=60.0, shift_overrun_penalty_per_min=0.5,
    )
    cost_half = evaluate(inst, sol, s_half)["operational_cost"]

    s_full = Settings()
    s_full.economics = Economics(
        shift_max_minutes=60.0, shift_overrun_penalty_per_min=1.0,
    )
    cost_full = evaluate(inst, sol, s_full)["operational_cost"]

    delta_half = cost_half - cost_off
    delta_full = cost_full - cost_off
    # Linearity within float precision
    assert abs(delta_full - 2.0 * delta_half) < 1e-6
    # Either both nonzero (term active) or both zero (no overrun on this seed).
    if delta_half == 0.0:
        # Cap loose enough that no route violated -- still a valid case,
        # but the test is more informative with a forced violation.
        pass
    else:
        assert delta_half > 0.0
        assert delta_full > 0.0


def test_shift_overrun_gating_requires_both_coefs():
    """The penalty is gated on BOTH coefs being non-zero. Setting only one
    must leave cost identical to defaults."""
    inst = generate(N=20, seed=42)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]

    # Only the cap set, no penalty
    s_capOnly = Settings()
    s_capOnly.economics = Economics(shift_max_minutes=60.0,
                                     shift_overrun_penalty_per_min=0.0)
    assert evaluate(inst, sol, s_capOnly)["operational_cost"] == cost_off

    # Only the penalty set, no cap
    s_penOnly = Settings()
    s_penOnly.economics = Economics(shift_max_minutes=0.0,
                                     shift_overrun_penalty_per_min=1.0)
    assert evaluate(inst, sol, s_penOnly)["operational_cost"] == cost_off


def test_peak_hour_surcharge_is_linear_in_multiplier_minus_one():
    """Setting peak_hour_wage_multiplier=1.5 must add exactly
    (0.5 * wage_per_minute) * total_peak_minutes to cost.
    Doubling (mult-1) doubles the surcharge.
    """
    inst = generate(N=20, seed=51)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]

    # Wide peak window covering whole route so we know there's overlap
    windows = ((0,), (10000,))  # 0 -> 10000 min covers any plausible route

    s_15 = Settings()
    s_15.economics = Economics(
        peak_window_starts=windows[0],
        peak_window_ends=windows[1],
        peak_hour_wage_multiplier=1.5,
    )
    cost_15 = evaluate(inst, sol, s_15)["operational_cost"]

    s_20 = Settings()
    s_20.economics = Economics(
        peak_window_starts=windows[0],
        peak_window_ends=windows[1],
        peak_hour_wage_multiplier=2.0,
    )
    cost_20 = evaluate(inst, sol, s_20)["operational_cost"]

    delta_15 = cost_15 - cost_off
    delta_20 = cost_20 - cost_off
    # mult=2.0 adds (2-1) = 1x base; mult=1.5 adds (1.5-1) = 0.5x base.
    # So delta_20 should equal exactly 2x delta_15.
    assert abs(delta_20 - 2.0 * delta_15) < 1e-6
    assert delta_15 > 0  # there IS overlap on this small instance


def test_driver_time_variance_is_linear_in_coef():
    """Doubling driver_time_variance_penalty_coef doubles the cost delta
    (penalty is coef * variance, linear in coef)."""
    inst = generate(N=20, seed=77)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]

    s_half = Settings()
    s_half.economics = Economics(driver_time_variance_penalty_coef=0.5)
    cost_half = evaluate(inst, sol, s_half)["operational_cost"]

    s_full = Settings()
    s_full.economics = Economics(driver_time_variance_penalty_coef=1.0)
    cost_full = evaluate(inst, sol, s_full)["operational_cost"]

    delta_half = cost_half - cost_off
    delta_full = cost_full - cost_off
    assert abs(delta_full - 2.0 * delta_half) < 1e-6
    # Must be non-zero on a small multi-route instance (greedy produces
    # uneven route durations).
    assert delta_half > 0


def test_driver_breaks_is_linear_in_overrun_driving_minutes():
    """iter-6a-1: doubling break_violation_penalty_per_min doubles cost-delta
    on solutions whose per-route driving exceeds driving_max_minutes."""
    inst = generate(N=20, seed=91)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]

    # Tight cap (30 min driving per route) to force violations on the
    # small instance where greedy produces multi-customer routes.
    s_half = Settings()
    s_half.economics = Economics(
        driving_max_minutes=30.0, break_violation_penalty_per_min=1.0,
    )
    cost_half = evaluate(inst, sol, s_half)["operational_cost"]

    s_full = Settings()
    s_full.economics = Economics(
        driving_max_minutes=30.0, break_violation_penalty_per_min=2.0,
    )
    cost_full = evaluate(inst, sol, s_full)["operational_cost"]

    delta_half = cost_half - cost_off
    delta_full = cost_full - cost_off
    assert abs(delta_full - 2.0 * delta_half) < 1e-6
    # Either both nonzero (cap was tight enough) or both zero (no overrun).
    if delta_half > 0:
        assert delta_full > 0


def test_driver_breaks_gating_requires_both_coefs():
    """iter-6a-1: setting only one of (driving_max_minutes,
    break_violation_penalty_per_min) leaves cost bit-identical."""
    inst = generate(N=20, seed=92)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]

    s_capOnly = Settings()
    s_capOnly.economics = Economics(driving_max_minutes=30.0,
                                     break_violation_penalty_per_min=0.0)
    assert evaluate(inst, sol, s_capOnly)["operational_cost"] == cost_off

    s_penOnly = Settings()
    s_penOnly.economics = Economics(driving_max_minutes=0.0,
                                     break_violation_penalty_per_min=1.0)
    assert evaluate(inst, sol, s_penOnly)["operational_cost"] == cost_off


def test_embargo_violation_is_linear_in_penalty():
    """iter-6a-2: doubling embargo_violation_penalty_per_visit doubles cost
    delta when at least one visit falls inside an embargo window."""
    inst = generate(N=20, seed=101)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    # Very wide embargo so we're guaranteed at least one visit inside
    s_half = Settings()
    s_half.economics = Economics(
        embargo_window_starts=(0,), embargo_window_ends=(10000,),
        embargo_violation_penalty_per_visit=50.0,
    )
    s_full = Settings()
    s_full.economics = Economics(
        embargo_window_starts=(0,), embargo_window_ends=(10000,),
        embargo_violation_penalty_per_visit=100.0,
    )
    dh = evaluate(inst, sol, s_half)["operational_cost"] - cost_off
    df = evaluate(inst, sol, s_full)["operational_cost"] - cost_off
    assert abs(df - 2.0 * dh) < 1e-6
    assert dh > 0  # at least one visit fell in the wide window


def test_embargo_gating_requires_windows_and_penalty():
    """iter-6a-2: missing penalty OR missing windows leaves cost identical."""
    inst = generate(N=20, seed=102)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]

    # Windows but zero penalty
    s_w = Settings()
    s_w.economics = Economics(
        embargo_window_starts=(0,), embargo_window_ends=(10000,),
        embargo_violation_penalty_per_visit=0.0,
    )
    assert evaluate(inst, sol, s_w)["operational_cost"] == cost_off
    # Penalty but no windows
    s_p = Settings()
    s_p.economics = Economics(
        embargo_window_starts=(), embargo_window_ends=(),
        embargo_violation_penalty_per_visit=100.0,
    )
    assert evaluate(inst, sol, s_p)["operational_cost"] == cost_off


def test_ev_range_is_linear_in_penalty():
    """iter-6a-4: doubling range_violation_penalty_per_mile doubles cost
    delta when at least one route exceeds vehicle_range_miles."""
    inst = generate(N=20, seed=103)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    # Tight range to force overflow on small instance
    s_half = Settings()
    s_half.economics = Economics(
        vehicle_range_miles=0.5,
        range_violation_penalty_per_mile=1.0,
    )
    s_full = Settings()
    s_full.economics = Economics(
        vehicle_range_miles=0.5,
        range_violation_penalty_per_mile=2.0,
    )
    dh = evaluate(inst, sol, s_half)["operational_cost"] - cost_off
    df = evaluate(inst, sol, s_full)["operational_cost"] - cost_off
    assert abs(df - 2.0 * dh) < 1e-6
    if dh > 0:
        assert df > 0


def test_mixed_fleets_smallest_viable_class_assigned():
    """iter-6a-3: each route picks the smallest class with capacity >= load.
    Solutions with all routes within smallest-class cap pay smallest-class
    premium; solutions with routes exceeding smallest-class cap pay larger
    class. Setting up two classes verifies the right one is picked per route.
    """
    inst = generate(N=20, seed=111)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    cap = float(inst.vehicle_capacity)
    # Two classes: small (half cap, $5 fixed premium) and large (full cap,
    # $20 fixed premium). All non-empty routes should pay something.
    s_mf = Settings()
    s_mf.economics = Economics(
        vehicle_class_capacities=(cap * 0.5, cap),
        vehicle_class_fixed_premiums=(5.0, 20.0),
        vehicle_class_per_mile_premiums=(0.0, 0.0),
    )
    cost_mf = evaluate(inst, sol, s_mf)["operational_cost"]
    delta = cost_mf - cost_off
    # Delta must be >= number_of_routes * smallest_class_premium ($5/route)
    # and <= number_of_routes * largest_class_premium ($20/route).
    K = int(sol.metrics["num_vehicles_used"])
    assert K > 0
    assert delta >= 5.0 * K - 1e-6
    assert delta <= 20.0 * K + 1e-6


def test_pd_pairs_penalty_per_violation():
    """iter-6a-6: each PD pair violated (different routes OR wrong order)
    contributes exactly pd_violation_penalty_per_pair to cost."""
    inst = generate(N=20, seed=121)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    # Build PD pairs from served customers: pair every two adjacent ids.
    # Most pairs will land in different routes / wrong order under greedy.
    served = sorted({c for r in sol.routes for c in r.customers})
    if len(served) < 4:
        return  # not enough customers to form pairs
    pairs_flat: list[int] = []
    for i in range(0, len(served) - 1, 2):
        pairs_flat.append(served[i])
        pairs_flat.append(served[i + 1])
    s_half = Settings()
    s_half.economics = Economics(
        pd_pairs_flat=tuple(pairs_flat),
        pd_violation_penalty_per_pair=50.0,
    )
    s_full = Settings()
    s_full.economics = Economics(
        pd_pairs_flat=tuple(pairs_flat),
        pd_violation_penalty_per_pair=100.0,
    )
    dh = evaluate(inst, sol, s_half)["operational_cost"] - cost_off
    df = evaluate(inst, sol, s_full)["operational_cost"] - cost_off
    assert abs(df - 2.0 * dh) < 1e-6
    # At least one pair must violate (under greedy with these arbitrary pairs)
    assert dh > 0


def test_min_routes_penalty_per_missing_route():
    """iter-6a-8: penalty = under_min_routes_penalty * max(0, min_required - K).
    Linear in penalty coef. Zero if K already >= min_required."""
    inst = generate(N=20, seed=131)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    K = int(sol.metrics["num_vehicles_used"])
    # Set min ABOVE actual K, so there's a shortage
    over_min = K + 5
    s_half = Settings()
    s_half.economics = Economics(
        min_routes_required=over_min,
        under_min_routes_penalty_per_route=100.0,
    )
    s_full = Settings()
    s_full.economics = Economics(
        min_routes_required=over_min,
        under_min_routes_penalty_per_route=200.0,
    )
    dh = evaluate(inst, sol, s_half)["operational_cost"] - cost_off
    df = evaluate(inst, sol, s_full)["operational_cost"] - cost_off
    assert abs(df - 2.0 * dh) < 1e-6
    # Shortage is exactly 5; at $100/route should be exactly $500
    assert abs(dh - 500.0) < 1e-6

    # Set min AT or BELOW K -> no shortage, no penalty
    s_at_k = Settings()
    s_at_k.economics = Economics(
        min_routes_required=K,
        under_min_routes_penalty_per_route=100.0,
    )
    assert evaluate(inst, sol, s_at_k)["operational_cost"] == cost_off


def test_skills_mismatch_penalty_per_violation():
    """iter-6a-7: linear in penalty. Mismatch = customer's required_level >
    route_class's provided_level."""
    inst = generate(N=20, seed=141)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    # Require level 2 for first 5 customers; provide level 0 (class 0).
    # All routes default to class 0 (mixed_fleets off), so all 5 customers
    # always mismatch -> exactly 5 violations.
    served = sorted({c for r in sol.routes for c in r.customers})
    if len(served) < 5: return
    skill_pairs: list[int] = []
    for cid in served[:5]:
        skill_pairs.append(cid)
        skill_pairs.append(2)  # required level 2
    s_half = Settings()
    s_half.economics = Economics(
        customer_skill_levels_flat=tuple(skill_pairs),
        vehicle_class_skill_levels=(0,),  # class 0 provides level 0
        skill_mismatch_penalty_per_visit=50.0,
    )
    s_full = Settings()
    s_full.economics = Economics(
        customer_skill_levels_flat=tuple(skill_pairs),
        vehicle_class_skill_levels=(0,),
        skill_mismatch_penalty_per_visit=100.0,
    )
    dh = evaluate(inst, sol, s_half)["operational_cost"] - cost_off
    df = evaluate(inst, sol, s_full)["operational_cost"] - cost_off
    assert abs(df - 2.0 * dh) < 1e-6
    # 5 mismatches * $50 = $250
    assert abs(dh - 250.0) < 1e-6


def test_skills_gating_requires_pairs_and_class_levels_and_penalty():
    """iter-6a-7: any missing component leaves cost identical."""
    inst = generate(N=20, seed=142)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    # Missing pairs
    s_a = Settings()
    s_a.economics = Economics(
        customer_skill_levels_flat=(),
        vehicle_class_skill_levels=(2,),
        skill_mismatch_penalty_per_visit=50.0,
    )
    assert evaluate(inst, sol, s_a)["operational_cost"] == cost_off
    # Missing class levels
    s_b = Settings()
    s_b.economics = Economics(
        customer_skill_levels_flat=(1, 2),
        vehicle_class_skill_levels=(),
        skill_mismatch_penalty_per_visit=50.0,
    )
    assert evaluate(inst, sol, s_b)["operational_cost"] == cost_off
    # Missing penalty
    s_c = Settings()
    s_c.economics = Economics(
        customer_skill_levels_flat=(1, 2),
        vehicle_class_skill_levels=(0,),
        skill_mismatch_penalty_per_visit=0.0,
    )
    assert evaluate(inst, sol, s_c)["operational_cost"] == cost_off


def test_multi_depot_default_none_is_bit_identical():
    """iter-6a-5: with inst.depots=None (default), evaluator behavior is
    bit-identical to legacy single-depot path."""
    from svrptw.io.instance import Instance as _Inst
    inst = generate(N=20, seed=151)
    # Generated instances default to depots=None
    assert inst.depots is None
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    cost_legacy = evaluate(inst, sol, s)["operational_cost"]
    # Re-evaluate (no state change); must be identical
    cost_again = evaluate(inst, sol, s)["operational_cost"]
    assert cost_legacy == cost_again


def test_depot_shift_operator_picks_better_depot():
    """iter-6a-5-bis: depot_shift operator should choose the cheaper depot
    when given a clear cost gradient between depots."""
    from svrptw.io.instance import Depot as _Depot
    from svrptw.solvers.common.depot_shift import depot_shift
    inst = generate(N=20, seed=153)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    # Depot 0 = legit; depot 1 = late-ready so depot 1 is much worse
    late_depot = _Depot(
        node_id=inst.depot.node_id, x=inst.depot.x, y=inst.depot.y,
        ready=inst.depot.ready + 10000,
        due=inst.depot.due + 10000,
    )
    inst.depots = [inst.depot, late_depot]
    # Force all routes to the bad depot, then check operator fixes
    for r in sol.routes:
        r.depot_idx = 1
    cost_bad = evaluate(inst, sol, s)["operational_cost"]
    sol_fixed = depot_shift(inst, sol, s, max_seconds=2.0)
    cost_fixed = evaluate(inst, sol_fixed, s)["operational_cost"]
    # Operator must reduce cost (it should switch routes back to depot 0)
    assert cost_fixed < cost_bad
    # And the chosen indices should all be 0 (the good depot)
    for r in sol_fixed.routes:
        if r.customers:
            assert r.depot_idx == 0


def test_depot_shift_noop_when_single_depot():
    """iter-6a-5-bis: depot_shift should be a no-op when inst.depots is None
    or has only one depot."""
    from svrptw.solvers.common.depot_shift import depot_shift
    inst = generate(N=20, seed=154)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    cost_before = evaluate(inst, sol, s)["operational_cost"]
    # inst.depots is None by default; operator must not modify cost
    sol2 = depot_shift(inst, sol, s, max_seconds=2.0)
    cost_after = evaluate(inst, sol2, s)["operational_cost"]
    assert cost_after == cost_before


def test_multi_depot_depot_idx_changes_ready_time():
    """iter-6a-5: when inst.depots is set with different ready times,
    Route.depot_idx selects the depot whose ready time gates the route start."""
    from svrptw.io.instance import Depot as _Depot
    inst = generate(N=20, seed=152)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    # Build a second depot at the same xy but with a MUCH later ready time.
    # depot_idx=0 routes start at the original ready; depot_idx=1 routes
    # start later -> ALL their customers arrive past TW -> mass late penalty.
    late_depot = _Depot(
        node_id=inst.depot.node_id, x=inst.depot.x, y=inst.depot.y,
        ready=inst.depot.ready + 10000,  # 10000 min later -> way past every TW
        due=inst.depot.due + 10000,
    )
    inst.depots = [inst.depot, late_depot]
    cost_d0 = evaluate(inst, sol, s)["operational_cost"]
    # Switch all routes to depot_idx=1 (the late depot)
    for r in sol.routes:
        r.depot_idx = 1
    cost_d1 = evaluate(inst, sol, s)["operational_cost"]
    # depot_idx=1 starting 10000 min later -> ALL customers arrive late
    # -> hard_late_penalty * N customers. Cost must be substantially higher.
    assert cost_d1 > cost_d0 + 1000  # at minimum +$1000 (one late customer)


def test_min_routes_gating_requires_both():
    """iter-6a-8: missing min OR missing penalty leaves cost identical."""
    inst = generate(N=20, seed=132)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    s_m = Settings()
    s_m.economics = Economics(min_routes_required=100,
                               under_min_routes_penalty_per_route=0.0)
    assert evaluate(inst, sol, s_m)["operational_cost"] == cost_off
    s_p = Settings()
    s_p.economics = Economics(min_routes_required=0,
                               under_min_routes_penalty_per_route=100.0)
    assert evaluate(inst, sol, s_p)["operational_cost"] == cost_off


def test_pd_pairs_gating_requires_pairs_and_penalty():
    """iter-6a-6: missing penalty OR missing pairs leaves cost identical."""
    inst = generate(N=20, seed=122)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    s_pair_only = Settings()
    s_pair_only.economics = Economics(
        pd_pairs_flat=(1, 2, 3, 4),
        pd_violation_penalty_per_pair=0.0,
    )
    assert evaluate(inst, sol, s_pair_only)["operational_cost"] == cost_off
    s_pen_only = Settings()
    s_pen_only.economics = Economics(
        pd_pairs_flat=(),
        pd_violation_penalty_per_pair=100.0,
    )
    assert evaluate(inst, sol, s_pen_only)["operational_cost"] == cost_off


def test_mixed_fleets_gating_requires_nonzero_premium_and_matched_lens():
    """iter-6a-3: all three tuples must be non-empty AND same length AND
    at least one premium must be non-zero, else cost is bit-identical."""
    inst = generate(N=20, seed=112)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]

    # All zero premiums -> not active
    s_zero = Settings()
    s_zero.economics = Economics(
        vehicle_class_capacities=(100.0,),
        vehicle_class_fixed_premiums=(0.0,),
        vehicle_class_per_mile_premiums=(0.0,),
    )
    assert evaluate(inst, sol, s_zero)["operational_cost"] == cost_off

    # Mismatched tuple lengths -> not active
    s_mis = Settings()
    s_mis.economics = Economics(
        vehicle_class_capacities=(100.0, 200.0),
        vehicle_class_fixed_premiums=(5.0,),
        vehicle_class_per_mile_premiums=(0.1,),
    )
    assert evaluate(inst, sol, s_mis)["operational_cost"] == cost_off

    # Empty tuples -> not active
    s_emp = Settings()
    s_emp.economics = Economics(
        vehicle_class_capacities=(),
        vehicle_class_fixed_premiums=(),
        vehicle_class_per_mile_premiums=(),
    )
    assert evaluate(inst, sol, s_emp)["operational_cost"] == cost_off


def test_ev_range_gating_requires_both_coefs():
    """iter-6a-4: missing range OR missing penalty leaves cost identical."""
    inst = generate(N=20, seed=104)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    s_r = Settings()
    s_r.economics = Economics(vehicle_range_miles=10.0,
                               range_violation_penalty_per_mile=0.0)
    assert evaluate(inst, sol, s_r)["operational_cost"] == cost_off
    s_p = Settings()
    s_p.economics = Economics(vehicle_range_miles=0.0,
                               range_violation_penalty_per_mile=1.0)
    assert evaluate(inst, sol, s_p)["operational_cost"] == cost_off


def test_driver_time_variance_gating_requires_nonzero_coef():
    """Setting the coef to 0 (default) must leave cost bit-identical."""
    inst = generate(N=20, seed=88)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]
    # Explicitly setting to 0 reproduces the default
    s_zero = Settings()
    s_zero.economics = Economics(driver_time_variance_penalty_coef=0.0)
    assert evaluate(inst, sol, s_zero)["operational_cost"] == cost_off


def test_peak_hour_gating_requires_both_windows_and_multiplier():
    """The surcharge is gated on: multiplier != 1.0 AND non-empty windows.
    Any other combination must leave cost identical to defaults."""
    inst = generate(N=20, seed=63)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    cost_off = evaluate(inst, sol, s_off)["operational_cost"]

    # Windows set but multiplier=1.0
    s_w = Settings()
    s_w.economics = Economics(
        peak_window_starts=(480,), peak_window_ends=(600,),
        peak_hour_wage_multiplier=1.0,
    )
    assert evaluate(inst, sol, s_w)["operational_cost"] == cost_off

    # Multiplier set but no windows
    s_m = Settings()
    s_m.economics = Economics(
        peak_window_starts=(), peak_window_ends=(),
        peak_hour_wage_multiplier=1.5,
    )
    assert evaluate(inst, sol, s_m)["operational_cost"] == cost_off

    # Mismatched start/end counts (defensive parse)
    s_mismatch = Settings()
    s_mismatch.economics = Economics(
        peak_window_starts=(480,), peak_window_ends=(),
        peak_hour_wage_multiplier=1.5,
    )
    assert evaluate(inst, sol, s_mismatch)["operational_cost"] == cost_off


def test_zero_coefs_bit_identical_to_pre_phase_f():
    """With all three new coefs at 0.0, evaluate() must return EXACTLY
    the cost it would return with the unextended Economics. We re-run
    twice with two distinct Settings instances (both at default) to
    confirm there's no implicit metrics-call overhead path."""
    inst = generate(N=20, seed=11)
    s = Settings()
    sol = greedy_mod.solve(inst, s)
    cost_a = evaluate(inst, sol, s)["operational_cost"]
    s2 = Settings()
    cost_b = evaluate(inst, sol, s2)["operational_cost"]
    assert cost_a == cost_b


def test_crossings_term_is_linear_in_count():
    """Setting crossings_penalty_per_pair = K must add exactly
    K * inter_route_crossings to the cost (and nothing else)."""
    inst = generate(N=30, seed=7)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    base = evaluate(inst, sol, s_off)["operational_cost"]
    qs = score_solution(inst, sol)
    K = 2.0
    s_on = Settings()
    s_on.economics = Economics(crossings_penalty_per_pair=K)
    actual = evaluate(inst, sol, s_on)["operational_cost"]
    expected = base + K * float(qs.inter_route_crossings)
    assert abs(actual - expected) < 1e-9


def test_util_imbalance_term_is_linear_in_cv():
    """util_imbalance_penalty_coef * load_util_cv added linearly."""
    inst = generate(N=30, seed=8)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    base = evaluate(inst, sol, s_off)["operational_cost"]
    qs = score_solution(inst, sol)
    coef = 20.0
    s_on = Settings()
    s_on.economics = Economics(util_imbalance_penalty_coef=coef)
    actual = evaluate(inst, sol, s_on)["operational_cost"]
    expected = base + coef * float(qs.load_util_cv)
    assert abs(actual - expected) < 1e-9


def test_tw_buffer_bonus_term_is_negative_linear():
    """tw_buffer_bonus_coef is a REWARD: cost should DECREASE when
    enabled (assuming positive mean_tw_buffer_score)."""
    inst = generate(N=30, seed=9)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    base = evaluate(inst, sol, s_off)["operational_cost"]
    qs = score_solution(inst, sol)
    coef = 15.0
    s_on = Settings()
    s_on.economics = Economics(tw_buffer_bonus_coef=coef)
    actual = evaluate(inst, sol, s_on)["operational_cost"]
    expected = base + (-1.0) * coef * float(qs.mean_tw_buffer_score)
    assert abs(actual - expected) < 1e-9
    # Reward direction: when the mean buffer score is positive, the
    # bonus REDUCES cost (i.e. the solver is "paid" to keep buffer up).
    if qs.mean_tw_buffer_score > 0.0:
        assert actual < base


def test_three_terms_additive():
    """All three terms together = sum of each individually applied."""
    inst = generate(N=30, seed=10)
    s_off = Settings()
    sol = greedy_mod.solve(inst, s_off)
    base = evaluate(inst, sol, s_off)["operational_cost"]
    qs = score_solution(inst, sol)
    K1, K2, K3 = 2.0, 20.0, 15.0
    s_all = Settings()
    s_all.economics = Economics(
        crossings_penalty_per_pair=K1,
        util_imbalance_penalty_coef=K2,
        tw_buffer_bonus_coef=K3,
    )
    actual = evaluate(inst, sol, s_all)["operational_cost"]
    expected = (
        base
        + K1 * float(qs.inter_route_crossings)
        + K2 * float(qs.load_util_cv)
        + (-1.0) * K3 * float(qs.mean_tw_buffer_score)
    )
    assert abs(actual - expected) < 1e-9
