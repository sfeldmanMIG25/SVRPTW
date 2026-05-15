# SPEC-7-COST-01 — Exponential underutilization + symmetry cost penalties

```
ID:            SPEC-7-COST-01
Title:         Penalize under-loaded routes and asymmetric route balances
               so the solver stops opening routes "for free"
Owner role:    Cost-model Engineer
Status:        FROZEN
Inputs:        Settings.economics — new fields (default off for back-compat)
Outputs:       svrptw.solvers.common.solution.evaluate() returns an
               operational_cost that includes route/leg utilization
               and symmetry terms when enabled
Depends on:    nothing (Settings extension is additive)
```

## Why

Current cost model:

```
operational_cost = wage_per_minute · total_time
                 + cost_per_mile   · total_distance
                 + early_wait_per_minute · early_wait
                 + hard_late_penalty · missed
                 + wage_per_minute · late
```

There is **no fixed cost or utilization penalty per route**. A
1-customer route close to the depot costs ~ wage·(depot→cust→depot
time) + miles·distance — which is small. The solver therefore has
zero incentive to consolidate when a near-by customer could be added
to an existing route. This is the mechanism behind the
ViVRP-Gemini smoke divergence (`bench/figures/v1_vivrp_gemini_smoke.md`):
portfolio@10 wins cost by spawning many small routes; LKH-3 wins the
dispatcher score by running fewer, fuller routes.

The dispatcher heuristic is simple: **a route exists to be utilized.**
If it isn't, the dispatcher will not pay for it.

## Behaviour

Two new economics terms, both opt-in:

```python
class Economics(BaseModel):
    # existing fields ...
    underutil_penalty_per_route: float = 0.0   # default 0 → no change
    underutil_exponent: float = 2.0             # exponent on (1 - util)
    underutil_target_util: float = 0.70         # below this → penalty grows
    symmetry_penalty_coef: float = 0.0          # default 0 → no change
```

### Per-route underutilization

For each route `r` with load `L_r` and capacity `C_r`:

```
util_r        = L_r / C_r                    # in [0, 1]
shortfall     = max(0.0, target_util - util_r)
penalty_r     = underutil_penalty_per_route * shortfall ** underutil_exponent
```

Exponent ≥ 1; with the default 2.0 a 50%-utilised route incurs
`penalty * (0.70 - 0.50)² = penalty * 0.04`, but a 10%-utilised route
incurs `penalty * (0.70 - 0.10)² = penalty * 0.36` — 9× worse, as
the user's "exponentially based on utilization" intent demands.

### Cross-route symmetry

Dispatchers also reject route sheets where one driver has a 9-hour
day and another has 90 minutes. We penalise the spread:

```
util_std       = stdev_over_routes(util_r)
symmetry_pen   = symmetry_penalty_coef * util_std ** 2
```

Squared so a small imbalance is cheap, a big imbalance is expensive.

### Per-leg utilization (deferred to SPEC-7-COST-02)

The user also called out **leg** utilization — penalising legs where
the on-board load is far from capacity. This requires tracking
`load_at_leg_l` and integrating across the route. It is a stronger
signal than route-level utilization but harder to compute
incrementally inside local-search; we ship route-level first, add
leg-level when the operator inner loop is ready.

## Defaults

The economically meaningful defaults (to enable in `defaults.yaml`
once the existing bench runs complete):

```yaml
economics:
  underutil_penalty_per_route: 40.0   # ~half a route-fixed-cost when 50% util
  underutil_exponent: 2.0
  underutil_target_util: 0.70
  symmetry_penalty_coef: 80.0
```

Picked so that on the ViVRP-Gemini smoke set (portfolio@10 N=50,
mean 11 routes, mean util ~0.35), the new total cost ≈ 629 +
11·40·(0.70-0.35)² ≈ 629 + 54 = 683 — closer to LKH's 774 — *but*
on LKH's fuller-route solutions (mean util ~0.65), the penalty is
~11·40·(0.70-0.65)² ≈ 1.1, essentially zero. The flip in ranking is
the intended behaviour. We verify after the spec lands.

## Back-compat

When all new fields are 0.0 (the file-default), `evaluate()` returns
bit-identical numbers to the pre-spec version. Every existing bench
result remains valid. The new behaviour is gated entirely on
non-zero penalty coefficients.

## Acceptance gates

1. With all penalties = 0, `evaluate()` is bit-identical to the
   pre-spec version on a 40-row sample. (Unit test.)
2. With non-zero penalties, a 1-customer route incurs strictly more
   cost than the same customer inserted into a half-loaded route
   in the same vicinity. (Unit test on a hand-crafted instance.)
3. On the ViVRP-Gemini smoke set, enabling the defaults re-orders the
   solvers so the gap to LKH-3 closes by ≥ 50 %. (Bench gate.)
4. No solver becomes infeasible because of the new term — feasibility
   is unaffected by reward shaping.

## Why exponent rather than linear

A linear penalty pushes the solver to mild improvements
everywhere, smearing the signal. An exponential penalty creates a
sharp gradient on the worst route specifically, which is exactly
what destroy-and-repair operators (SPEC-7-OPS-DESTROY-01) want to
target. The exponent value 2.0 is a hyperparameter; 3.0 produces
"penalize the worst route brutally and ignore everything else,"
2.0 is the middle ground.

## Files touched

| Path | Change |
|---|---|
| `svrptw/config/schema.py` | add 4 fields to `Economics` |
| `svrptw/config/defaults.yaml` | document defaults; keep 0.0 until bench reruns |
| `svrptw/solvers/common/solution.py` | add penalty computation behind flag check |
| `tests/unit/test_cost_underutil.py` | bit-equality with zero coefs; signed sanity test |
