# SPEC-2-AUCTION-01 — Auction with GART bid valuation

```
ID:            SPEC-2-AUCTION-01
Title:         Decentralized auction with GART-marginal bid cost
Owner role:    OR Engineer
Status:        FROZEN
Inputs:        Instance, Settings, TourLengthEstimator
Outputs:       svrptw.solvers.classical.auction_gart.solve -> Solution
```

## Behavior

Reimplement the existing `auction_solver.py` route-insertion-cost computation
(currently a full route re-optimization per bid — O(N²) per round) with GART
marginal estimates from SPEC-1-GART-01. Each vehicle bids on each unserved
customer with:

  `bid_cost = GART_marginal(my_route, candidate) + wage_per_minute * waiting`

The auction loop:

```
while unserved:
    bids = {}
    for v in vehicles:
        for c in unserved:
            if not v.capacity_ok(c) or not v.tw_feasible(c):
                continue
            bids[(v, c)] = est.estimate_marginal(v.route, c, dist_matrix=T)
    # Solve assignment with linear_sum_assignment
    pairs = solve_assignment(bids)
    for (v, c) in pairs:
        v.append(c)
        unserved.remove(c)
```

`solve_assignment` uses `scipy.optimize.linear_sum_assignment` (Hungarian) on
the bid matrix. Rounds continue until no feasible bid exists.

## Acceptance

- Wall-clock per auction round ≤ 20% of the baseline (legacy) auction's per-round time on N=100, N=200 instances.
- Operational cost within ±2% of the legacy auction on the v1 set;
  better-than-baseline is upside.
- GART cache hit rate ≥ 80% within a single solve.

## Non-goals

- Reproducing the legacy auction's stochastic risk-buffer (deferred per
  ADR-0001 D4).
- Hyperparameter tuning. That's a follow-up Optuna sweep.

## Dependencies

- SPEC-1-GART-01.
