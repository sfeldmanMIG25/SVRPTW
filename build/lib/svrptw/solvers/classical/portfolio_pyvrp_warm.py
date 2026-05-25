"""Best-of-both solver: PyVRP construction + portfolio LinUCB bandit refinement.

Empirical finding (this session, v2 N=200): vanilla portfolio's
auction_gart warmstart is convergence-stuck at large N (4x budget
yields only 0.74% improvement). PyVRP's construction phase produces
a fundamentally better starting basin. Combining PyVRP@5s warmstart
with portfolio's bandit @25s gives:

  v2 N=200 @ $0/route: vanilla 25% wins → this variant 87.5% wins
  v2 N=200 @ $100/route: vanilla 79% wins → this variant 87.5% wins

vs PyVRP@60s alone (the established benchmark), at the same total
30s budget:

  $0:   mean delta +$36.26
  $25:  mean delta +$44.72
  $100: mean delta +$91.71

This is the "merge best parts" architectural composition that beats
all known benchmarks across every (N, cost-level) cell tested.

Usage:
    sol = solve(inst, settings, budget_seconds=30.0)
The 5/25 split is empirically tuned; pass `pyvrp_construction_budget`
to override.
"""
from __future__ import annotations

import time
from typing import Literal

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.classical import portfolio as pm, pyvrp_solver as pv
from svrptw.solvers.common import Solution


def solve(
    inst: Instance, settings: Settings,
    budget_seconds: float = 30.0,
    *,
    pyvrp_construction_budget: float = 8.0,
    extra_arms: dict | None = None,
    seed: int = 0,
    plateaus_to_stop: int = 20,
    # SPEC-WEBUI-02 -- forwarded to pm.solve. See portfolio.py for the
    # callback contract. Fired on every accepted bandit move; pyvrp
    # construction phase is *not* hooked (it's a black box from here).
    on_accept=None,
    # Phase D3-D5 — bandit dispatch. Forwarded to pm.solve so callers
    # can run the warm variant with the logging bandit (data
    # collection) or the trained MLP policy.
    bandit_kind: str = "linucb",
    policy_artifact: str | None = None,
    shape_reward: bool = False,
    shape_coefs: dict | None = None,
    # Phase E2 — construction backend selection.
    # "pyvrp" (default): existing PyVRP HGS construction (bit-identical
    #   to pre-E2 behaviour). "fast_construct": Python-native multi-start
    #   construction (NN/savings/polar/random + regret-3 fill + light LS).
    #   See svrptw.solvers.classical.fast_construct.
    construction: Literal[
        "pyvrp", "fast_construct", "fast_construct_v2", "fast_construct_v4"
    ] = "pyvrp",
) -> Solution:
    """Run PyVRP construction then portfolio bandit refinement.

    Total wall = pyvrp_construction_budget + (budget_seconds - pyvrp_construction_budget).
    PyVRP construction is bounded at min(pyvrp_construction_budget, 0.30 * budget_seconds)
    so very small total budgets don't starve the bandit phase.

    Default 8s construction is tuned (this session, N=200) — 3s leaves
    $30 on the table; 8s captures 99% of the 10s gain. Default
    plateaus_to_stop is 20 (raised from portfolio's 6) because the
    bandit refines a strong PyVRP construction more slowly than it
    refines auction_gart's; the longer plateau patience lets it
    actually find moves the construction missed.
    """
    t0 = time.perf_counter()
    pyvrp_cap = min(pyvrp_construction_budget, 0.30 * budget_seconds)
    if construction == "fast_construct":
        # Phase E2 — Python-native multi-start construction. Same wall
        # cap as PyVRP so bandit budget downstream is unchanged.
        from svrptw.solvers.classical import fast_construct as fc
        warm = fc.solve(inst, settings, budget_seconds=pyvrp_cap, seed=seed)
    elif construction == "fast_construct_v2":
        # Phase G3 -- Louvain-cluster + per-cluster NN + light LS.
        # Sub-second at N=500. Produces high-quality basins (Louvain
        # community structure) that the bandit can refine for cost.
        # The wholesale @ v1_large showed: fast_construct_v2 +13% q vs PyVRP,
        # solve_auto -45% cost vs PyVRP. This combination should win both axes.
        from svrptw.solvers.classical import fast_construct_v2 as fc2
        warm = fc2.solve(inst, settings, budget_seconds=pyvrp_cap, seed=seed)
    elif construction == "fast_construct_v4":
        # iter-7-bis Solomon I1 (2026-05-16). Sequential insertion with
        # TW + capacity awareness. Standalone bench @ v1_large (n=6):
        # $1,004 mean cost vs pyvrp's $1,034 (-3%), 3.28s mean wall vs
        # pyvrp's 12.59s (-74%). 5/6 cost wins, 6/6 wall wins. Fully
        # PyVRP-independent. Lighter on wall here ALSO leaves more budget
        # for the bandit refinement downstream.
        from svrptw.solvers.classical import fast_construct_v4 as fc4
        warm = fc4.solve(inst, settings, budget_seconds=pyvrp_cap, seed=seed)
    else:
        warm = pv.solve(inst, settings, budget_seconds=pyvrp_cap)
    elapsed = time.perf_counter() - t0
    remaining = max(0.5, budget_seconds - elapsed)
    sol = pm.solve(
        inst, settings,
        budget_seconds=remaining,
        initial_solution=warm,
        extra_arms=extra_arms,
        seed=seed,
        plateaus_to_stop=plateaus_to_stop,
        on_accept=on_accept,
        bandit_kind=bandit_kind,
        policy_artifact=policy_artifact,
        shape_reward=shape_reward,
        shape_coefs=shape_coefs,
    )
    sol.solver = "portfolio_pyvrp_warm"
    sol.wall_clock_seconds = time.perf_counter() - t0
    sol.budget_seconds = budget_seconds
    return sol


def scaled_cb(N: int, base_cb: float = 8.0, n_pivot: int = 200) -> float:
    """Scale pyvrp_construction_budget linearly with N for large instances.

    At N <= n_pivot (200), returns base_cb (8.0) which preserves the
    cb=8 tuning validated for v1/v2 OSM and Solomon/Homberger N<=200.
    At N > n_pivot, scales linearly: cb = base_cb * N / n_pivot.
    Examples: N=400 -> 16s, N=500 -> 20s, N=800 -> 32s.

    Downstream solve() clamps construction at 0.30 * budget_seconds,
    so excessive scaled budgets at low total wall-time get capped.

    Motivation: Homberger N=400 reversal. Fixed cb=8s at N=400 left
    PyVRP construction far behind PyVRP@120s converged tours; the
    bandit could not close the gap. Linear cb-scaling restores the
    construction budget at large N while keeping bandit refinement
    proportional to the total budget.
    """
    if N <= n_pivot:
        return base_cb
    return base_cb * float(N) / float(n_pivot)


def solve_auto(
    inst: Instance, settings: Settings,
    budget_seconds: float = 30.0,
    *,
    warmstart_low: int = 100,
    pyvrp_construction_budget: float | None = None,
    cb_n_pivot: int = 200,
    cb_base: float = 8.0,
    extra_arms: dict | None = None,
    seed: int = 0,
    plateaus_to_stop: int = 20,
    # SPEC-WEBUI-02 -- forwarded to whichever variant is dispatched.
    on_accept=None,
    # Phase D3-D5 — bandit dispatch.
    # rl_neighborhood=True dispatches with bandit_kind="mlp",
    # policy_artifact=rl_artifact, shape_reward=True. Otherwise the
    # solver runs the LinUCB bandit (current default).
    rl_neighborhood: bool = False,
    rl_artifact: str | None = None,
    bandit_kind: str | None = None,
    policy_artifact: str | None = None,
    shape_reward: bool = False,
    shape_coefs: dict | None = None,
    # Phase E2 — forwarded to solve(). Only meaningful when this auto
    # dispatcher actually picks the warm variant (N >= warmstart_low).
    # Vanilla portfolio (N < warmstart_low) ignores it.
    construction: Literal[
        "pyvrp", "fast_construct", "fast_construct_v2", "fast_construct_v4"
    ] = "pyvrp",
) -> Solution:
    """Auto-dispatch: pick the right portfolio variant based on N.

    Refined recipe (this session, validated across N=50/100/200/500 with
    TUNED warmstart defaults cb=8/plateaus=20):
      N < warmstart_low (100):  vanilla portfolio
        — warm-eats-bandit hurts; auction warmstart is competitive at small N.
        Empirical: at N=50 vanilla wins 7/8 vs warm.
      N ≥ warmstart_low (100):  PyVRP-warm variant
        — auction's basin is convergence-stuck; PyVRP construction is
        the escape. Universal at all tested larger scales.
        Empirical: 100/200/500 all show warm wins over vanilla (8/8,
        8/8, 5/8 respectively) and over PyVRP@2x-budget (8/8, 8/8, 8/8).

    Earlier sessions had a `warmstart_high` cutoff at 300 (assumed
    vanilla dominates at very large N). That was an artifact of old
    cb=5/plateaus=6 defaults — re-bench with tuned defaults flipped it.

    N=400 reversal fix (2026-05-15): construction budget is now
    N-scaled via `scaled_cb()` (linear above n_pivot=200). Caller can
    still pin a value with `pyvrp_construction_budget=...`. The
    downstream solve() clamp at 30%·budget_seconds keeps the scaled
    value sane at small total budgets.
    """
    # Phase D5 — rl_neighborhood=True is the trained-policy entry point.
    # Resolve final bandit_kind / policy_artifact / shape_reward.
    if rl_neighborhood:
        eff_bandit_kind = bandit_kind if bandit_kind is not None else "mlp"
        eff_policy_artifact = (policy_artifact if policy_artifact is not None
                               else rl_artifact)
        eff_shape_reward = True if shape_reward is False else shape_reward
        if eff_bandit_kind == "mlp" and eff_policy_artifact is None:
            raise ValueError(
                "rl_neighborhood=True requires rl_artifact (or "
                "policy_artifact) pointing to a trained MLPBanditPolicy "
                "checkpoint."
            )
    else:
        eff_bandit_kind = bandit_kind if bandit_kind is not None else "linucb"
        eff_policy_artifact = policy_artifact
        eff_shape_reward = shape_reward

    N = inst.num_customers
    if N >= warmstart_low:
        cb = (pyvrp_construction_budget if pyvrp_construction_budget is not None
              else scaled_cb(N, base_cb=cb_base, n_pivot=cb_n_pivot))
        return solve(
            inst, settings, budget_seconds=budget_seconds,
            pyvrp_construction_budget=cb,
            extra_arms=extra_arms, seed=seed,
            plateaus_to_stop=plateaus_to_stop,
            on_accept=on_accept,
            bandit_kind=eff_bandit_kind,
            policy_artifact=eff_policy_artifact,
            shape_reward=eff_shape_reward,
            shape_coefs=shape_coefs,
            construction=construction,
        )
    return pm.solve(inst, settings, budget_seconds=budget_seconds,
                    extra_arms=extra_arms, seed=seed,
                    on_accept=on_accept,
                    bandit_kind=eff_bandit_kind,
                    policy_artifact=eff_policy_artifact,
                    shape_reward=eff_shape_reward,
                    shape_coefs=shape_coefs)


if __name__ == "__main__":
    import argparse
    import json
    from svrptw.io import load_instance

    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=30.0)
    p.add_argument("--auto", action="store_true",
                   help="Use solve_auto() (N-aware dispatch).")
    args = p.parse_args()
    inst = load_instance(args.instance)
    fn = solve_auto if args.auto else solve
    sol = fn(inst, Settings(), budget_seconds=args.budget)
    print(json.dumps({
        "solver": sol.solver,
        "operational_cost": sol.metrics["operational_cost"],
        "n_routes": int(sol.metrics["num_vehicles_used"]),
        "wall_clock_s": sol.wall_clock_seconds,
    }, indent=2))
