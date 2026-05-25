"""Typed configuration. See SPEC-0-CFG-01."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Economics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wage_per_hour: float = 14.50
    cost_per_mile: float = 0.50
    hard_late_penalty: float = 1000.0
    early_wait_penalty_per_minute: float | None = None
    # SPEC-7-COST-01 — opt-in underutilisation + symmetry penalties.
    # Defaults are 0.0 so existing bench results stay bit-identical.
    underutil_penalty_per_route: float = 0.0
    underutil_exponent: float = 2.0
    underutil_target_util: float = 0.70
    symmetry_penalty_coef: float = 0.0
    # SPEC-7-COST-02 — opt-in per-route fixed cost (vehicle-day rental,
    # driver-shift overhead, etc.). Default 0.0 leaves cost identical.
    #
    # iter-5u WARNING: leaving this at 0.0 means cross-solver comparisons
    # are NOT K-fair. Solvers that use more vehicles (fast_construct_v2
    # uses 1.7-3.3x more than solve_auto on v1_large N=500/1000) get a
    # free subsidy. The wholesale leaderboard "fcv2 wins quality" framing
    # was driven entirely by this. For production benches set
    # per_route_fixed_cost to a realistic value (~$50-100); see
    # iter-5u session note for the re-leaderboard analysis.
    per_route_fixed_cost: float = 0.0
    # SPEC-F-COST-01 -- opt-in structural cost terms (iter-5l, Phase F).
    # Default 0.0 -> bit-identical legacy behaviour. Each term that the bandit can
    # optimize but PyVRP cannot is a structural lever that compounds the
    # architectural advantage.
    crossings_penalty_per_pair: float = 0.0       # $ per inter-route segment crossing
    util_imbalance_penalty_coef: float = 0.0      # $ * CV(route_utils)
    tw_buffer_bonus_coef: float = 0.0             # -$ * mean_tw_buffer_score (negative = reward)
    # iter-5v -- driver-shift / per-route duration overrun. PyVRP can't see this
    # (no per-route duration constraint in the cost model); solve_auto's bandit
    # can optimise for it. Gated on BOTH being non-zero -> bit-identical default.
    # Examples:
    #   shift_max_minutes=480, shift_overrun_penalty_per_min=1.0
    #     => 8-hour shift, $1 per minute over (~$60/hour overtime equivalent)
    #   shift_max_minutes=300, shift_overrun_penalty_per_min=0.5
    #     => 5-hour shift, $0.50/min over (mild break-violation surrogate)
    shift_max_minutes: float = 0.0               # per-route duration cap (depot-to-depot)
    shift_overrun_penalty_per_min: float = 0.0   # $ per minute that route_duration > cap
    # iter-5x -- peak-hour wage surcharge. Minutes of route activity (travel +
    # service + wait + late) that fall inside any peak window are charged at
    # base_wage * peak_wage_multiplier instead of base_wage. Real-world
    # equivalent: rush-hour overtime, congestion surcharge, demand-period pay.
    # PyVRP can't see this; solve_auto's bandit can shift route start times
    # / customer sequencing to push activity OUT of peak windows.
    # Gated on multiplier != 1.0 AND non-empty windows -> bit-identical default.
    # Format: pairs of (start_minute, end_minute) since midnight.
    # Example (8-10am + 5-7pm rush hours):
    #   peak_window_starts=(480, 1020), peak_window_ends=(600, 1140),
    #   peak_hour_wage_multiplier=1.5
    peak_window_starts: tuple[int, ...] = ()
    peak_window_ends: tuple[int, ...] = ()
    peak_hour_wage_multiplier: float = 1.0
    # iter-5y -- cross-route driver-fairness term. Penalty per $^2 of variance
    # in (wage_per_min * route_duration) across active routes. High variance =
    # some drivers work 8 hours while others work 2 hours; low variance = fair.
    # PyVRP optimizes pure ops cost and produces highly unequal route durations;
    # solve_auto's bandit has cross-route operators (relocate / swap /
    # two_opt_star / merge_routes) that CAN rebalance, so the iter-5x Step 0
    # operator-coverage check is satisfied for this term.
    # Gated on coef != 0 -> bit-identical default. Units: coef is dimensionless
    # multiplier on variance ($-squared minutes-squared); a coef of ~0.001
    # produces meaningful gradients without dominating the cost.
    driver_time_variance_penalty_coef: float = 0.0
    # iter-6a-1 -- EU Regulation 561/2006 lite: max continuous driving before
    # a mandatory rest period. First-cut implementation caps total driving
    # time (sum of T[prev, cid] travel-only minutes, excluding service+wait)
    # per route at `driving_max_minutes`. Excess minutes are charged at
    # `break_violation_penalty_per_min`. This conflates "continuous driving"
    # with "total driving per route" -- a 5-hour route with a 45-min lunch
    # break in the middle is technically EU-561-legal but penalised here.
    # Tractable first cut; refine to continuous-stretch tracking later.
    # PyVRP cannot model break rules at default settings; solve_auto's
    # bandit (with split_route operator in its arm set) CAN split overlong
    # driving routes into break-compliant ones.
    # Examples:
    #   driving_max_minutes=270, break_violation_penalty_per_min=2.0
    #     => EU 561 lite (4.5h cap, $2/min over)
    #   driving_max_minutes=480, break_violation_penalty_per_min=0.5
    #     => US HOS lite (8h cap, mild penalty)
    driving_max_minutes: float = 0.0
    break_violation_penalty_per_min: float = 0.0
    # iter-6a-2 -- hard zones / time-restricted areas (first cut: GLOBAL embargo
    # windows, not per-customer). Any customer visit whose ARRIVAL time falls
    # within any (start, end) embargo window incurs a per-visit penalty.
    # Real-world equivalents: school zones closed 8-9am, residential areas
    # closed after 9pm for trucks, historic districts closed during business
    # hours. PyVRP can't see embargo windows; solve_auto's bandit (relocate /
    # swap reorder customers, shifting arrival times) can route around them
    # IF customer time-windows allow flexibility.
    # Gated on (any non-zero penalty AND non-empty windows AND matching lens).
    # Future iteration: per-customer embargo (requires Instance schema change).
    embargo_window_starts: tuple[int, ...] = ()
    embargo_window_ends: tuple[int, ...] = ()
    embargo_violation_penalty_per_visit: float = 0.0
    # iter-6a-4 -- electric vehicles (first cut: per-route distance range).
    # If a route's total distance exceeds vehicle_range_miles, charge linear
    # penalty per mile over. Real-world equivalent: EV with 200-mile battery
    # incurs charging-stop overhead beyond range. PyVRP can't model EVs;
    # solve_auto's bandit (split_route operator) can break long routes.
    # Gated on BOTH coefs being non-zero -> bit-identical default.
    # Future iteration: charging-station node insertion (changes Instance).
    vehicle_range_miles: float = 0.0
    range_violation_penalty_per_mile: float = 0.0
    # iter-6a-3 -- mixed fleets (additive class-premium model). Per-route load
    # is matched to the smallest viable vehicle class (class_capacity >= load).
    # Each class has a fixed-cost premium AND a per-mile premium added on top
    # of the baseline wage/dist costs. PyVRP can't model class assignment;
    # solve_auto's bandit (relocate / swap reshuffle loads across routes) can
    # consolidate small routes onto larger classes to save fixed premium OR
    # split big routes onto smaller classes to save per-mile premium.
    # Gated on ALL THREE tuples non-empty AND matching length AND any premium > 0.
    # Tuple format: ascending capacity (smallest first). Baseline is "class -1"
    # = inst.vehicle_capacity, no premium. Classes act as UPGRADES that cost
    # more but offer larger capacity.
    vehicle_class_capacities: tuple[float, ...] = ()
    vehicle_class_fixed_premiums: tuple[float, ...] = ()
    vehicle_class_per_mile_premiums: tuple[float, ...] = ()
    # iter-6a-6 -- pickup-and-delivery precedence pairs (first cut: penalty
    # only, not hard constraint). Each pair = (pickup_cid, delivery_cid).
    # Violation = pickup and delivery in DIFFERENT routes OR delivery comes
    # BEFORE pickup in the same route. Per-pair penalty added to cost.
    # Real-world equivalents: package pickup from warehouse A must precede
    # delivery to customer B; trailer drop-off precedes pickup; passenger
    # transport (pickup at home before dropoff at airport). PyVRP at default
    # settings doesn't see PD pairs; solve_auto's bandit (relocate / swap)
    # can move pickup-or-delivery between routes to satisfy precedence.
    # Gated on (non-empty pairs AND non-zero penalty) -> bit-identical default.
    # Pair format: flat tuple of (pickup, delivery, pickup, delivery, ...).
    # Using tuple-of-tuples breaks pydantic v1 validation; flat tuple works.
    pd_pairs_flat: tuple[int, ...] = ()
    pd_violation_penalty_per_pair: float = 0.0
    # iter-6a-8 -- minimum routes required (labor / union contract). Some
    # operations must keep at least N drivers active per day (collective
    # bargaining agreements, minimum-shift guarantees). Penalty if the
    # solution uses fewer than min_routes_required active routes.
    # Step 0: split_route operator can add routes by splitting -- bandit can
    # bring solution UP to the floor if cost term incentivises it.
    # Gated on BOTH non-zero -> bit-identical default.
    min_routes_required: int = 0
    under_min_routes_penalty_per_route: float = 0.0
    # iter-6a-7 -- driver skills / customer-vehicle matching (first cut: integer
    # skill levels, not string sets). Each customer has a required skill level;
    # each vehicle class provides a skill level. If the route's assigned class
    # provides a level LESS than the customer's required level, violation.
    # Real-world equivalents: hazmat certification, refrigerated vehicle for
    # cold-chain customers, lift-gate truck for heavy customers.
    # Depends on iter-6a-3 mixed_fleets being active for class assignment.
    # If vehicle_class_skill_levels is empty OR mixed_fleets is off, treat
    # every route as class 0 (smallest) for the check.
    # Format: customer_skill_levels_flat is alternating (cust_id, level, ...).
    # vehicle_class_skill_levels is per-class level (parallel to
    # vehicle_class_capacities order, ascending).
    # Step 0 check: relocate/swap can move a customer to a route whose class
    # provides the required level -- bandit-actionable.
    # Gated on (non-empty pairs AND non-empty class levels AND non-zero pen).
    customer_skill_levels_flat: tuple[int, ...] = ()
    vehicle_class_skill_levels: tuple[int, ...] = ()
    skill_mismatch_penalty_per_visit: float = 0.0

    @property
    def wage_per_minute(self) -> float:
        return self.wage_per_hour / 60.0

    @property
    def early_wait_per_minute(self) -> float:
        return self.early_wait_penalty_per_minute if self.early_wait_penalty_per_minute is not None else self.wage_per_minute


class TimeBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day_start_minute: int = 480
    day_end_minute: int = 960


class Stochastic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    travel_lognormal_sigma: float = 0.6
    service_normal_sigma: float = 8.0
    service_mean: float = 10.0
    days_per_instance: int = 30


class SolverCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["greedy", "ortools", "lkh3", "dqn", "auction", "mcts", "pomo"] = "greedy"
    params: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SVRPTW_")
    economics: Economics = Economics()
    time: TimeBudget = TimeBudget()
    stochastic: Stochastic = Stochastic()
    solver: SolverCfg = SolverCfg()
    seed: int = 0
    gart_artifact_path: str = "D:/VRP-Advanced-Estimator-Integration/estimators/lgbm_alpha_model_v4.joblib"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)
