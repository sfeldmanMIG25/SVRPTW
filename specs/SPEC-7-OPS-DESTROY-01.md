# SPEC-7-OPS-DESTROY-01 — Destroy-island, drop-leg, drop-route operators

```
ID:            SPEC-7-OPS-DESTROY-01
Title:         Three new destroy operators that target under-utilised
               components — Voronoi-island, single-leg, full-route
Owner role:    Search Engineer
Status:        FROZEN
Inputs:        current solution, granular neighborhood (SPEC-3-GRAN-01)
Outputs:       svrptw.solvers.common.local_search.{destroy_island,
                                                    drop_leg, drop_route}
Depends on:    SPEC-7-COST-01 (the cost gradient that motivates these);
               SPEC-3-PORTFOLIO-01 (bandit will choose between them)
```

## Why

SISR string-removal (SPEC-3-SISR-01) and 2-opt* are *route-local*
moves. They never delete a route outright. When the cost model
gains an underutilisation penalty (SPEC-7-COST-01), the steepest
descent is often **"delete an entire under-loaded component and
push its customers into neighbours"** — a move SISR cannot make
because SISR removes a *string within* a route, not the route
itself.

Three new destroy operators, picked by the LinUCB bandit:

1. **`destroy_island`** — pick one Voronoi (Thiessen) cluster and
   wipe its customers across whichever routes they were on.
2. **`drop_leg`** — pick one route segment `(i, i+1)` with low
   load-utilisation across it and redistribute the endpoints.
3. **`drop_route`** — pick one whole route (the worst utilisation
   wins with probability ∝ `(target - util)²`) and re-insert every
   customer via regret-k into surrounding routes.

Each is paired with the existing repair step (`_blink_greedy_insert`
or `_regret_k_insert`), so the operator signature matches the SISR
contract:

```python
def destroy_island(sol, inst, rng, settings, *, k_neighbors=20) -> Solution
def drop_leg(sol, inst, rng, settings) -> Solution
def drop_route(sol, inst, rng, settings) -> Solution
```

## Behaviour

### `destroy_island`

1. Compute customer-coords centroids per route → seed points for
   Voronoi.
2. `scipy.spatial.Voronoi(seeds)` (cached per-instance,
   invalidated when route count changes).
3. Pick a seed weighted by `(target_util - util_seed)²` (bias
   toward the worst route).
4. Remove all customers whose nearest seed is the picked one.
5. Re-insert via regret-2 into the remaining routes.

Cost vs SISR: O(k log k) Voronoi build + O(n) reinsertion. On
N=200 we expect ~50 ms per call — acceptable for the bandit's
inner loop.

Failure mode: if Voronoi is degenerate (collinear seeds, n_routes <
3), fall back to "remove the worst-util route entirely" — i.e.
`drop_route`.

### `drop_leg`

1. For each leg `(c_i → c_{i+1})` in each route, compute
   `leg_load_util = load_after_c_i / capacity`.
2. Pick the leg with the lowest `leg_load_util` (softmax with
   temperature 1.0 for tie-breaking).
3. Remove `c_i` and `c_{i+1}`.
4. Re-insert with blink-greedy into the *granular neighbours* of
   each removed customer (k=20).

This is the cheapest of the three — useful when the bandit is
fast-cycling.

### `drop_route`

1. Sort routes by `(target_util - util_route)²`, descending.
2. Pick the top-1 route with probability 1 − ε; with ε pick from
   the top-3 to maintain exploration.
3. Remove every customer; mark the vehicle slot as free.
4. Re-insert each removed customer via regret-3 into the remaining
   routes. If any re-insertion fails feasibility, restore the
   original route (no-regret).

This is the **strongest cost-reducing move** when SPEC-7-COST-01
penalties are active — it directly attacks the term the dispatcher
hates most.

## Acceptance gates

1. Each operator preserves feasibility *or* restores the prior
   solution. (Unit test with hand-crafted instances.)
2. On the v1 N=100 set with SPEC-7-COST-01 defaults enabled,
   `drop_route` reduces operational_cost on ≥ 30 % of instances
   relative to the SISR-only portfolio.
3. The bandit (SPEC-3-PORTFOLIO-01) sees the three new arms and
   does not crash. After 100 trials on a v1 N=100 instance, at
   least one of the three has been selected ≥ 5 times.
4. Operator wall-time at N=200: P95 ≤ 80 ms each.

## Why this is not "just SISR with different parameters"

SISR removes a *contiguous string* within a route. The new
operators remove either a *geographic cluster crossing routes*
(`destroy_island`), a *single bad edge* (`drop_leg`), or *an
entire route* (`drop_route`). They expose different cost gradients
to the bandit. The bandit needs decorrelated arms to learn
something useful — three near-duplicate arms gives it nothing.

## Files this spec creates / touches

| Path | Change |
|---|---|
| `svrptw/solvers/common/local_search.py` | add 3 operators |
| `svrptw/solvers/common/voronoi.py` (new) | cached scipy Voronoi |
| `svrptw/solvers/classical/portfolio.py` | register 3 new arms in op list |
| `tests/unit/test_destroy_operators.py` | feasibility + no-regret invariants |
