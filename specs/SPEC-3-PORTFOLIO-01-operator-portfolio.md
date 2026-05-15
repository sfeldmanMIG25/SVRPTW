# SPEC-3-PORTFOLIO-01 — Operator portfolio solver

```
ID:            SPEC-3-PORTFOLIO-01
Title:         Multi-operator metaheuristic solver with ALNS-style selection
Owner role:    OR Engineer + ML Engineer
Status:        DRAFT (operators land incrementally; bandit selector next)
Inputs:        Instance, Settings, time budget
Outputs:       svrptw.solvers.classical.portfolio.solve -> Solution
```

## Why

The current `auction_gart` chain (auction → merge_routes → relocate → 2-opt)
is a fixed sequence. ALNS-style portfolios (Ropke & Pisinger 2006) and
hyper-heuristics (Burke et al. 2013) show that adaptive operator selection
under a shared incumbent beats any single fixed chain on heterogeneous
instance distributions. The instance generator produces 8 city topologies
× 4 sizes — heterogeneous by design.

## Operators (in `svrptw.solvers.common.local_search`)

Each operator is a function `(Instance, Solution, Settings, max_seconds) -> Solution`
that returns a TW-feasible Solution with cost ≤ input cost (no-op if no
improvement found). All operators set `cand.feasible = bool(cand.metrics["feasible"])`.

Already landed:
- `relocate` — first-improvement, single-customer Or-opt (1-customer) inter/intra route.
- `two_opt_intra` — reverse a sub-segment of one route.
- `merge_routes` — evacuate the shortest non-empty route into other routes
  by best-insertion. Strongest move for cutting K when wage dominates.

To land:
- `or_opt_2`, `or_opt_3` — Or-opt with chain length 2 and 3 (relocate adjacent
  pairs/triples in one move). Tightens routes.
- `swap_intra`, `swap_inter` — exchange two customers' positions. Best-improvement variants.
- `cross_exchange` — swap a sub-segment between two routes. Strong for
  rebalancing very different-length routes.
- `ejection_chain` (SPEC-3-EJECT-01) — k-cyclic chain of moves that no single
  Or-opt could find. Glover (1996); the headline operator for this spec.
- `perturb_destroy_repair` — remove R% of customers (worst-by-marginal or
  random) and re-insert greedily. Restarts local search out of basins.

## Selection (the "trained backend on operator selection")

Two implementations gated by `cfg.solver.params.op_select`:

1. **Bandit** (`bandit`): UCB1 over operators. Reward = relative improvement
   per second. Adaptive within a single solve; no offline training needed.
2. **Policy** (`policy`): a small PyTorch MLP that maps a 32-dim state
   (current cost, num routes, mean route length, std route length, time
   remaining, ops applied so far, ...) to a softmax over operators. Trained
   offline by REINFORCE on the v1 instance distribution. Loaded from
   `models/op_policy/v1.pt`. See SPEC-3-OPSEL-01 for training details.

Operator selection uses the **bandit** when no checkpoint exists, and the
**policy** when `op_policy/v1.pt` is available — same interface, different
provider.

## Schedule

```python
def solve(inst, settings, budget_seconds):
    sol = warm_start(inst, settings)            # auction_gart's bid loop
    selector = make_selector(settings, ops)
    while wall_clock < budget_seconds:
        op = selector.choose(state_of(sol))
        sol_new = op(inst, sol, settings, max_seconds=...)
        selector.update(op, reward=(sol.cost - sol_new.cost) / dt)
        if sol_new.cost < sol.cost:
            sol = sol_new
    return sol
```

## Acceptance

- On `instances/v1` (when ready), `portfolio@10` Pareto-dominates
  `auction_gart` on ≥ 80% of instances at the 10-second budget.
- On the N=100 bin specifically, `portfolio@10` closes ≥ 50% of the gap to
  LKH-3's mean cost (currently 1390 vs 870 → target ≤ 1130).
- Bandit selector converges to a per-instance-class operator distribution
  that's distinct (e.g. ejection chain dominant on Manhattan grid;
  cross_exchange dominant on sprawl).

## Non-goals

- Beating LKH-3 on N=100 outright (still the open problem — likely needs
  the drop-customer mechanism from a separate spec).
- Population-based search (GA, scatter search). Single incumbent only.

## Dependencies

- SPEC-2-AUCTION-01 (warm start).
- SPEC-3-EJECT-01 (ejection chain operator).
- SPEC-3-OPSEL-01 (offline-trained policy, optional).
