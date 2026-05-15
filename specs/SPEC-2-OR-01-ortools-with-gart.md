# SPEC-2-OR-01 — OR-Tools with GART arc penalty

```
ID:            SPEC-2-OR-01
Title:         OR-Tools VRPTW solver augmented with GART arc cost
Owner role:    OR Engineer
Status:        FROZEN
Inputs:        Instance, Settings, budget_seconds, TourLengthEstimator
Outputs:       svrptw.solvers.classical.ortools_gart.solve -> Solution
```

## Behavior

Wrap the OR-Tools solver from SPEC-0-BENCH-01 (greedy + ortools) with a
two-pass enhancement that uses the GART estimator:

1. **Pre-cluster.** Compute a quick GART-driven k-medoid clustering of
   customers into `num_vehicles` groups, using GART(group) as the cost of
   each candidate cluster.  Within each cluster, customers are warm-started
   in a candidate route order that OR-Tools then refines.
2. **Re-score with GART marginal.** After OR-Tools converges within its time
   budget, run a "candidate-route re-scoring" pass that swaps the top-K
   customer-to-vehicle assignments where `GART_marginal(reassign) <
   marginal_under_current` improves total objective.  Re-scoring uses
   nearest-insertion deltas (SPEC-1-GART-01) for O(K·N) cost rather than
   re-running the full CP solver.

The two passes share the same wall-clock budget — by default 70/30
(OR-Tools / re-score).

## Acceptance

- On `instances/v1`, gart_on variant achieves ≥ 5% lower mean operational
  cost than gart_off at the same wall-clock budget, on the N ∈ {100, 200}
  bins.
- gart_on with `alpha=0, k=1` reproduces gart_off exactly (regression
  guard).
- Cache hit-rate on the GART estimator during a single solve ≥ 80%
  (most marginal queries are repeated swaps).

## Non-goals

- Modifying OR-Tools internals.
- Beating LKH-3 on N ≥ 500 (separate spec).

## Dependencies

- SPEC-1-GART-01.
- The baseline OR-Tools solver (already implemented in
  `svrptw.solvers.classical.ortools_solver`).
