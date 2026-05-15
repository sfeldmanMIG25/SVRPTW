"""Portfolio solver: LinUCB bandit picks operators from a rich pool.

SPEC-3-PORTFOLIO-01 + SPEC-3-OPSEL-02.

The pool covers every operator from `svrptw.solvers.common.local_search`:
  - relocate, swap (built via 2-opt-star which subsumes swap-by-tail-swap),
    2-opt intra, 2-opt*, 3-opt, merge_routes, vehicle_kill, soft_drop,
    SISR destroy-recreate, ejection chain, cyclic 3-exchange.

The bandit's context is the 16-dim state-feature vector. After each op,
reward = (cost_before - cost_after) / max(elapsed, 1e-3).

This solver bootstraps from the same two-tier auction construction as
auction_gart, but then runs an ADAPTIVE schedule instead of a fixed chain.
"""
from __future__ import annotations

import time

from svrptw.config import Settings
from svrptw.io import Instance
from svrptw.solvers.classical import auction_gart
from svrptw.solvers.classical import greedy as greedy_mod
from svrptw.solvers.common import (
    Solution,
    cyclic_3_exchange,
    ejection_chain,
    merge_routes,
    relocate,
    sisr_destroy_repair,
    soft_drop,
    swap_star,
    three_opt_intra,
    two_opt_intra,
    two_opt_star,
    vehicle_kill,
)
from svrptw.solvers.learning.bandit import LinUCBBandit
from svrptw.solvers.learning.state_features import FEATURE_DIM, featurize

from svrptw.solvers.common.local_search_destroy import destroy_island, drop_leg, drop_route

_OPS = {
    "merge_routes":   lambda i, s, c, t: merge_routes(i, s, c, max_seconds=t),
    "relocate":       lambda i, s, c, t: relocate(i, s, c, max_seconds=t),
    "two_opt_intra":  lambda i, s, c, t: two_opt_intra(i, s, c, max_seconds=t),
    "two_opt_star":   lambda i, s, c, t: two_opt_star(i, s, c, max_seconds=t),
    "swap_star":      lambda i, s, c, t: swap_star(i, s, c, max_seconds=t),
    "three_opt_intra":lambda i, s, c, t: three_opt_intra(i, s, c, max_seconds=t),
    "sisr":           lambda i, s, c, t: sisr_destroy_repair(i, s, c, max_seconds=t, seed=int(time.time_ns()) & 0xFFFF),  # see seed_local_rng below
    "ejection_chain": lambda i, s, c, t: ejection_chain(i, s, c, max_chain_length=3, max_seconds=t),
    "cyclic_3":       lambda i, s, c, t: cyclic_3_exchange(i, s, c, max_seconds=t),
    "vehicle_kill":   lambda i, s, c, t: vehicle_kill(i, s, c, max_seconds=t),
    "soft_drop":      lambda i, s, c, t: soft_drop(i, s, c, max_seconds=t),
    # SPEC-7-OPS-DESTROY-01 — epsilon-greedy worst-util route removal
    # with capacity-aware regret-1 reinsertion. Complements vehicle_kill
    # which is greedy-best-first.
    "drop_route":     lambda i, s, c, t: drop_route(i, s, c, max_seconds=t),
    # SPEC-7-OPS-DESTROY-01 — Voronoi-clustered destroy (k-NN island
    # around a seed customer, capacity-aware regret reinsertion).
    # Complements drop_route by attacking cross-route geographic
    # spaghetti rather than under-utilised individual routes.
    "destroy_island": lambda i, s, c, t: destroy_island(i, s, c, max_seconds=t),
    # SPEC-7-OPS-DESTROY-01 — drop_leg: remove the worst-load-util edge
    # (two consecutive customers) and re-insert. Cheaper than destroy_island,
    # more surgical than drop_route.
    "drop_leg":       lambda i, s, c, t: drop_leg(i, s, c, max_seconds=t),
}


def _best_logic_perturbation(
    inst: Instance, sol: Solution, settings: Settings,
    logic_ensemble, *, top_k: int, per_op_seconds: float,
    authoritative_only: bool,
) -> Solution | None:
    """SPEC-6-BANDIT-PLATEAU-01 — try `top_k` destroy operators and
    return the candidate with the highest logic-ensemble score.

    Returns None if no candidate is authoritative under
    `authoritative_only`. Skip ops with cost > 2× current (sanity bound).
    """
    candidates = ["destroy_island", "drop_route", "drop_leg", "sisr"]
    cur_cost = sol.metrics["operational_cost"]
    try:
        cur_score = logic_ensemble.score(inst, sol)
    except Exception:
        return None
    if authoritative_only and not cur_score.authoritative:
        return None

    best: Solution | None = None
    best_logic = cur_score.mean
    for op_name in candidates[:top_k]:
        if op_name not in _OPS:
            continue
        try:
            cand = _OPS[op_name](inst, sol, settings, per_op_seconds)
        except Exception:
            continue
        # Skip wildly worse cost (don't pretend basin-jumps can break the
        # bank — cap drift to 2× current).
        if cand.metrics["operational_cost"] > cur_cost * 2.0:
            continue
        try:
            cand_score = logic_ensemble.score(inst, cand)
        except Exception:
            continue
        if authoritative_only and not cand_score.authoritative:
            continue
        if cand_score.mean > best_logic + 1e-3:
            best_logic = cand_score.mean
            best = cand
    return best


def _council_to_portfolio_op(council_op):
    """Adapter: convert a council operator (sol, ctx) -> Sol|None into the
    portfolio's (inst, sol, settings, max_seconds) -> Solution signature.

    No-regret on cost: if the council op returns None or a worse solution,
    return the input solution unchanged. This is the contract the bandit
    expects (an op never makes things worse than the input)."""
    from svrptw.council.proposal import OperatorContext
    def adapter(inst, sol, settings, max_seconds):
        ctx = OperatorContext(instance=inst, settings=settings,
                              rng_seed=0, deadline_seconds=max_seconds)
        try:
            cand = council_op(sol, ctx)
        except Exception:
            return sol
        if cand is None:
            return sol
        if cand.metrics.get("operational_cost", float("inf")) >= sol.metrics["operational_cost"]:
            return sol
        return cand
    return adapter


def solve(inst: Instance, settings: Settings,
          budget_seconds: float = 30.0,
          per_op_seconds: float = 1.5,
          plateaus_to_stop: int = 6,
          alpha: float = 1.2,
          eps: float = 0.10,
          # SPEC-6-BANDIT-PLATEAU-01 — opt-in logic-driven basin-jump.
          plateau_basin_jump: bool = False,
          logic_ensemble: object | None = None,
          basin_jump_max_count: int = 2,
          basin_jump_top_k: int = 3,
          logic_authoritative_only: bool = True,
          # SPEC-8-COUNCIL-02 — extra arms registered into the bandit pool.
          extra_arms: dict | None = None,
          # SPEC-9-FUSION-01 — optional caller-supplied warmstart.
          initial_solution: Solution | None = None,
          # SPEC-8-COUNCIL-02 paired-seeding — controls the bandit's
          # eps-greedy/tie-break RNG and SISR's internal RNG. Required
          # for the composability bench to compare baseline-vs-augmented
          # at zero exploration-variance noise.
          seed: int = 0) -> Solution:
    """Bandit-driven solve.  Construction = auction_gart's tier-1/2/3 bid loop
    (without its fixed improvement chain), then LinUCB rolls operators
    until budget exhausted or `plateaus_to_stop` non-improving picks in a row.

    With `plateau_basin_jump=True` + a `logic_ensemble`, the solver makes
    up to `basin_jump_max_count` logic-driven perturbations on plateau —
    chooses the operator whose result maximally improves the ensemble's
    logic score, then resumes cost exploitation. The returned solution
    is always the best-cost seen (no-regret on cost).
    """
    t0 = time.perf_counter()

    # Warm-start from caller-provided solution (SPEC-9-FUSION-01) or
    # from the auction's construction phase. Caller-provided is
    # re-evaluated under the current settings so .metrics is fresh.
    if initial_solution is not None:
        from svrptw.solvers.common import evaluate as _eval
        sol = initial_solution
        sol.metrics = _eval(inst, sol, settings)
        sol.feasible = bool(sol.metrics["feasible"])
    else:
        sol = auction_gart.solve(inst, settings, budget_seconds=0.5)
    sol.solver = "portfolio"

    greedy_cost = greedy_mod.solve(inst, settings).metrics["operational_cost"]

    # SPEC-8-COUNCIL-02: merge any extra arms into the operator pool. Council
    # operators (sol, ctx) -> Sol|None are auto-adapted; portfolio-shape ops
    # pass through unchanged.
    ops_pool = dict(_OPS)
    # Override SISR with the caller-controlled seed so paired baseline/
    # augmented runs share SISR RNG and only differ in the augmented arm.
    ops_pool["sisr"] = lambda i, s, c, t, _seed=seed: sisr_destroy_repair(
        i, s, c, max_seconds=t, seed=_seed
    )
    if extra_arms:
        import inspect
        for name, fn in extra_arms.items():
            sig = inspect.signature(fn)
            if len(sig.parameters) == 2:
                ops_pool[name] = _council_to_portfolio_op(fn)
            else:
                ops_pool[name] = fn
    bandit = LinUCBBandit(ops=list(ops_pool.keys()), feature_dim=FEATURE_DIM,
                          alpha=alpha, seed=seed)
    plateaus = 0
    ops_applied = 0
    basin_jumps_used = 0
    best_sol = sol
    best_cost = sol.metrics["operational_cost"]
    history: list[tuple[str, float]] = []
    deadline = t0 + budget_seconds

    while time.perf_counter() < deadline and plateaus < plateaus_to_stop:
        ctx = featurize(inst, sol, greedy_cost=greedy_cost,
                        ops_applied=ops_applied, plateaus_so_far=plateaus)
        op_name = bandit.choose(ctx, eps=eps)
        op = ops_pool[op_name]
        # Allocate a slice of remaining time, capped per-op.
        remaining = deadline - time.perf_counter()
        slot = min(per_op_seconds, max(0.2, remaining))
        cost_before = sol.metrics["operational_cost"]
        t_op_start = time.perf_counter()
        try:
            new_sol = op(inst, sol, settings, slot)
        except Exception:
            new_sol = sol
        elapsed = time.perf_counter() - t_op_start
        cost_after = new_sol.metrics["operational_cost"]
        improvement = cost_before - cost_after
        # Reward shaped as $/s improvement, clipped so a 0.5s exploration
        # doesn't dominate the linear posterior.
        reward = max(-100.0, min(200.0, improvement / max(elapsed, 1e-3)))
        bandit.update(op_name, ctx, reward)
        history.append((op_name, improvement))
        ops_applied += 1
        if improvement > 1e-6:
            sol = new_sol
            plateaus = 0
            if cost_after < best_cost:
                best_sol = new_sol
                best_cost = cost_after
        else:
            plateaus += 1

        # SPEC-6-BANDIT-PLATEAU-01 — basin-jump on plateau, opt-in.
        # Trigger conditions: enabled, ensemble present, not yet at max
        # jumps, plateau threshold reached (before classical exit fires).
        if (plateau_basin_jump and logic_ensemble is not None
                and basin_jumps_used < basin_jump_max_count
                and plateaus >= plateaus_to_stop
                and (deadline - time.perf_counter()) > per_op_seconds * 1.5):
            cand = _best_logic_perturbation(
                inst, sol, settings, logic_ensemble,
                top_k=basin_jump_top_k, per_op_seconds=per_op_seconds,
                authoritative_only=logic_authoritative_only,
            )
            if cand is not None:
                # Accept unconditionally (cost may regress) — best-of-run
                # tracked separately so the returned solution is no-regret.
                sol = cand
                history.append((f"basin_jump_{basin_jumps_used}", 0.0))
                basin_jumps_used += 1
                plateaus = 0   # reset to allow exploitation from new basin

    # SPEC-6-BANDIT-PLATEAU-01 — final no-regret on cost: return the
    # best-cost solution seen anywhere in the trajectory. With basin-jump
    # disabled this collapses to the original behaviour (best == current).
    if best_cost < sol.metrics["operational_cost"] - 1e-6:
        sol = best_sol

    # No-regret guarantee vs greedy.
    greedy_sol = greedy_mod.solve(inst, settings)
    if greedy_sol.metrics["operational_cost"] < sol.metrics["operational_cost"] - 1e-6:
        sol = greedy_sol
    sol.solver = "portfolio"
    sol.wall_clock_seconds = time.perf_counter() - t0
    sol.metrics["bandit_ops_applied"] = float(ops_applied)
    sol.metrics["bandit_history"] = history  # type: ignore[assignment]
    return sol


if __name__ == "__main__":
    import argparse
    import json

    from svrptw.io import load_instance
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True)
    p.add_argument("--budget", type=float, default=15.0)
    args = p.parse_args()
    inst = load_instance(args.instance)
    sol = solve(inst, Settings(), budget_seconds=args.budget)
    metrics = {k: v for k, v in sol.metrics.items() if k != "bandit_history"}
    history = sol.metrics.get("bandit_history", [])
    print(json.dumps({"solver": sol.solver, "metrics": metrics,
                      "wall_clock_s": sol.wall_clock_seconds,
                      "ops_applied": int(metrics.get("bandit_ops_applied", 0)),
                      "first_5_ops": history[:5]}, indent=2))
