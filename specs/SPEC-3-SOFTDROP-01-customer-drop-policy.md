# SPEC-3-SOFTDROP-01 — Customer-drop policy

```
ID:            SPEC-3-SOFTDROP-01
Title:         Drop customers whose insertion cost exceeds the miss penalty
Owner role:    OR Engineer
Status:        FROZEN
Inputs:        Instance, Solution, Settings, max_seconds
Outputs:       svrptw.solvers.common.local_search.soft_drop
```

## Why

LKH-3 wins our N=100 partial bench (892 vs auction_gart's 1390) by
**strategically dropping ~0.1 customers per instance**. Our cost evaluator
already prices that trade-off correctly: a missed customer costs
`hard_late_penalty = 1000` (in dollars). If serving a customer costs the
vehicle > 1000 minutes-equivalent of effort, we should drop it.

The Track-2 brief identified this as the explicit fix: "replicate as a
soft drop penalty operator."

## Behaviour

```python
def soft_drop(inst, sol, settings, max_seconds=2.0) -> Solution:
    """For each customer, compute the *marginal cost of serving it*
    (current route cost − cost without it). If that marginal exceeds
    `settings.economics.hard_late_penalty` by margin ε, drop the
    customer.  Iterate until no further drop improves total cost."""
```

A drop strictly improves total operational cost iff:

  `marginal_cost_to_serve(c) > hard_late_penalty - epsilon`

where `epsilon` is a safety margin (default 1.0 minute-wage). After a
drop, re-evaluate to ensure no infeasibility was introduced (depot
return shifts, etc).

The operator never drops more than `max_drops_pct` of the unserved
customers per call (default 5%) — this is a hard cap so the operator
can't degenerate into "drop everyone."

## Acceptance

- On `instances/v1_partial` N=100 (Manhattan + Paris, 10 instances), the
  improvement chain ending with `soft_drop` closes ≥ 60 % of the gap to
  LKH-3 mean cost. Current gap: 1390 vs 892 = 498. Target: residual
  gap ≤ 200.
- On N=50 instances (where LKH-3 doesn't drop), `soft_drop` never
  drops any customer (regression assertion).
- Worst-case time: O(N · K · best-insertion-cost) per pass, capped by
  `max_drops_pct · N` drops total. For N=100, K=22: ~22k ops per pass.
- Like every operator: `cand.feasible` propagates, TW preserved on every
  touched route.

## Non-goals

- Strategic re-insertion (a dropped customer never comes back in a
  later pass). That's `perturb_destroy_repair`'s job.
- Cost-weighted prioritisation across customers (just greedy threshold).
- Modifying the cost model. The drop threshold uses the existing
  `hard_late_penalty` directly.

## Dependencies

- SPEC-0-CFG-01 (`economics.hard_late_penalty`).
- SPEC-2-AUCTION-01 (the chain it lives in).
