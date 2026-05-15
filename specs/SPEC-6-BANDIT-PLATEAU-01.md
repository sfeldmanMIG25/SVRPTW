# SPEC-6-BANDIT-PLATEAU-01 — Plateau-driven basin-jump in the LinUCB portfolio

```
ID:            SPEC-6-BANDIT-PLATEAU-01
Title:         When the bandit plateaus on cost, perturb under the logic objective
Owner role:    Search Engineer
Status:        FROZEN
Inputs:        portfolio search state, LogicEnsemble.score()
Outputs:       extension to svrptw.solvers.classical.portfolio
Depends on:    SPEC-3-PORTFOLIO-01, SPEC-6-LOGIC-01
```

## Why

The LinUCB portfolio currently exits when it sees ≥ N_plateau (default 5)
consecutive non-improving operator applications. That exit is correct
when we are stuck in a local optimum **under the cost objective**, but
it is wasteful when the logic axis sees a clearly better-shaped
neighbour. Instead of *stopping*, we should *redirect* the search for
one perturbation: pick the operator with the largest expected
*logic-axis* improvement, apply it (even if the cost goes up), then
re-enter cost-driven exploitation.

This is a small change with one specific use: get out of a basin in a
direction the cost gradient cannot see, without giving up
cost-monotonicity over the whole run.

## Behaviour

```python
PortfolioConfig(
    plateau_basin_jump=True,        # new field, default False (back-compat)
    plateau_window=5,
    basin_jump_max_count=2,         # at most 2 logic-driven perturbations per run
    logic_authoritative_only=True,  # only basin-jump when ensemble.authoritative
)
```

Algorithm change:

```
state: consecutive_non_improving = 0
       basin_jumps_used = 0
       best_cost_seen = +inf

after each operator apply:
    if new_cost + 1e-9 < best_cost_seen:
        best_cost_seen = new_cost
        consecutive_non_improving = 0
    else:
        consecutive_non_improving += 1

    if (consecutive_non_improving >= plateau_window
        and basin_jumps_used < basin_jump_max_count
        and plateau_basin_jump):
            # try basin-jump
            cand = best_logic_perturbation(state, top_k_ops=3)
            if cand is not None:
                # No-regret guarantee: we restore if logic-driven move
                # leaves us worse than greedy on cost too.
                accept_unconditionally(cand)
                basin_jumps_used += 1
                consecutive_non_improving = 0
                continue
    if consecutive_non_improving >= plateau_window and not did_basin_jump:
        break   # original exit
```

`best_logic_perturbation`:

```
for each of the top-k bandit-suggested operators:
    apply to current state in a copy
    compute logic_ensemble.score(instance, candidate)
    skip if ensemble.authoritative == False  (when logic_authoritative_only)
pick argmax of (logic_score - current_logic_score)
return None if no candidate is authoritative
```

No-regret guarantee preserved:

- At the end of the run, the returned solution is `argmin_cost(best_cost_ever,
  best_logic_basin_solution_cost, current)`. We never *return* a worse
  cost than the best cost seen.
- We do allow the *trajectory* to move up in cost mid-run.

## Acceptance gates

1. With `plateau_basin_jump=False`, the portfolio is bit-identical to
   SPEC-3-PORTFOLIO-01 behaviour. (Verified by an existing test fixture.)
2. With `plateau_basin_jump=True` on the v1 N=100 instances, the
   reported `operational_cost` never regresses relative to the
   `False` setting (no-regret).
3. On instances where the bandit hits the plateau exit, the basin-jump
   branch is exercised in ≥ 80 % of runs (i.e. ensemble was
   authoritative often enough to fire).
4. Average wall-clock overhead of the basin-jump branch is ≤ 5 % of
   total portfolio time at N=100 (we are gated by student latency
   <50 ms × top-k=3 perturbations × ≤ 2 jumps = 0.3 s budget).

## Non-goals

- Not making logic a continuous co-objective in the LinUCB reward. The
  bandit's reward signal remains cost-delta. Logic only enters at the
  plateau-detection branch.
- Not adapting `plateau_window` or `basin_jump_max_count` per instance.
  We pick the defaults from the v1 bench and freeze them; tuning lives
  in SPEC-3-OPSEL-01 (Optuna) once we have the data.
- Not exploring "logic descent" mode — the basin-jump is a *single*
  perturbation, after which the bandit reverts to cost.

## Files this spec touches

| Path | Change |
|---|---|
| `svrptw/solvers/classical/portfolio.py` | add `PortfolioConfig` fields + basin-jump branch |
| `svrptw/solvers/classical/portfolio_state.py` | track consecutive-non-improving, basin_jumps_used |
| `tests/unit/test_portfolio_basin_jump.py` | no-regret invariant + back-compat |
