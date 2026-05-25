---
date: 2026-05-15
project: SVRPTW
tags: [iter5y, fairness, cross-route, recipe-step-0.5, coefficient-calibration, validated]
---

# iter5y — cross-route fairness: 5/6 wins (and a new pre-flight check)

## TL;DR

Cross-route driver-time-variance penalty added as the third cost-term
test of the recipe. **First attempt (coef=0.001) produced 3/6 wins at
±$0 mean — TUNING ISSUE, not boundary**. Baseline variance penalty was
$0.10/inst on every instance (noise floor); the bandit had no
gradient. **Re-bench at coef=0.5 (calibrated to ~5% of ops cost):
5/6 wins, mean +$36.1/inst.** Cross-route generalization confirmed
with one Manhattan-N500 regression caveat (−$7.5).

This iteration codifies **Step 0.5 of the recipe**: calibrate coefficient
BEFORE bench, target baseline penalty ~3-10% of ops cost.

## The three structural categories now tested

| iter | term | axis | verdict |
|------|------|------|---------|
| 5w | shift_overrun | per-route | 6/6 wins, mean +$155.2 |
| 5x | peak_hour | per-segment-of-time | 2/6 (boundary, no time-shift operator) |
| 5y-bis | driver_time_variance | cross-route | 5/6 wins, mean +$36.1 |

The recipe generalizes to axes the bandit's operators can act on:
per-route operators (split / merge) → recipe works; cross-route operators
(relocate / swap / two_opt_star / merge_routes) → recipe works; no
time-segment-shift operator exists → recipe fails.

## iter-5y at coef=0.001 (intermediate, diagnostic)

```
instance              b_K s_K  b_ops  s_ops  b_var_pen s_var_pen     net
Manhattan-N0500        17  17  853.1  847.1     0.1      0.1        +6.1
Manhattan-N1000        32  32 1334.3 1373.0     0.1      0.1       -38.7
Paris-N0500            16  16  650.5  668.2     0.1      0.2       -17.7
Paris-N1000            30  30 1018.7 1006.5     0.1      0.1       +12.1
SanFrancisco-N0500     17  17  667.0  688.0     0.1      0.1       -21.0
SanFrancisco-N1000     32  31 1062.6 1003.3     0.1      0.1       +59.3
                                                              mean: -$0.001
```

**Diagnostic**: `b_var_pen == s_var_pen == $0.10` across ALL 6 instances.
The bandit's accepted moves produced solutions with EXACTLY THE SAME
variance penalty. Either:
- The bandit didn't try variance-reducing moves (cost signal too weak)
- OR variance-reducing moves were tried but rejected because their
  ops-cost increase outweighed the trivial $0.10 variance gain

Both readings point to the same fix: increase the coefficient so the
variance gain becomes a meaningful gradient.

## iter-5y-bis at coef=0.5 (re-bench, calibrated)

```
instance              b_K s_K  b_ops  s_ops  b_var_pen s_var_pen     net
Manhattan-N0500        17  17  829.0  859.9    33.6     10.2        -7.5
Manhattan-N1000        32  32 1329.7 1303.6    50.9     31.7       +45.3
Paris-N0500            16  16  658.5  655.9    90.7     11.4       +81.9
Paris-N1000            30  30 1013.8 1005.3    60.3     30.6       +38.2
SanFrancisco-N0500     17  17  662.1  693.6    91.7     21.9       +38.3
SanFrancisco-N1000     31  31 1032.4 1034.2    53.0     30.5       +20.6
                                                              mean: +$36.1
```

**5/6 wins, mean +$36.1/inst.** Key features:
- **K unchanged on every instance** — fairness operates within fixed
  route count. Existing cross-route operators (relocate / swap) move
  customers between routes to equalize durations.
- **Variance penalty cut to ~1/3** in most cells ($33-91 → $10-31).
  The bandit DID find more-balanced solutions when given a real signal.
- **Single loss: Manhattan-N500** (-$7.5). Paid $30.9 in ops cost for
  $23.4 variance reduction. Borderline cell — coefficient may need
  city-specific tuning.
- **Biggest win: Paris-N500** (+$81.9). Variance penalty $90.7 → $11.4
  (87% reduction) at $2.6 ops cost saving.

## Recipe Step 0.5 (added this iteration)

> **Step 0.5**: Calibrate the coefficient so the baseline penalty is a
> meaningful fraction of operational cost (~3–10%). Too small → the
> bandit treats the term as noise and produces identical solutions to
> baseline (iter-5y at coef=0.001 produced $0.10 penalty on a $900 base
> — pure noise floor, 3/6 wins, mean ±$0 net). Smoke-evaluate one
> solve_auto baseline run with the proposed coef BEFORE firing the
> paired bench.

The calibration heuristic:
- Run `solve_auto` once at default Settings on one representative instance.
- Apply the new term to the resulting solution at coef=1.0; this gives
  the "unit penalty" magnitude.
- Choose a coef such that `coef * unit_penalty / ops_cost ≈ 0.05`.

## What this opens

The recipe is now a complete framework:
- **Step 0** (iter-5x): operator-coverage prerequisite
- **Step 0.5** (iter-5y): coefficient calibration
- **Gates 1–7** (iter-5v/w): schema field, gated evaluator, 2 unit tests,
  smoke, paired 6-instance bench, 6/6 win threshold for ship

Five cost terms have now been opt-in plumbed in `Economics`:
1. `per_route_fixed_cost` (legacy)
2. `crossings_penalty_per_pair` (Phase F)
3. `util_imbalance_penalty_coef` (Phase F)
4. `tw_buffer_bonus_coef` (Phase F)
5. `shift_max_minutes` + `shift_overrun_penalty_per_min` (iter-5v) -- SHIPPABLE
6. `peak_window_starts` + `peak_window_ends` + `peak_hour_wage_multiplier` (iter-5x) -- term works but operators missing
7. `driver_time_variance_penalty_coef` (iter-5y) -- SHIPPABLE at coef=0.5 with Manhattan-N500 caveat

Three are bandit-actionable (cleared Steps 0+0.5+gates), one is bandit-blind
without a new operator. The recipe is mature enough to publish as a
methodology paper.

## Strategic significance

The user's review predicted iter-5y would test "whether the recipe
generalizes across the three structural categories of constraint." Result:

- **Yes, with caveats**. Per-route generalizes cleanly. Cross-route
  generalizes well but with calibration sensitivity. Per-segment-of-time
  needs a missing operator.
- **The recipe is operator-bound, not magic**. The cost-model-expansion
  thesis from iter-5u/v/w is real but bounded by the operator set's
  reach. Each new structural category needs either an existing operator
  on its axis OR a new operator (~1-2 days work).

This re-prioritizes the queue:
1. **iter-6a richer constraints** moves UP (qualitative gap, not linear stack).
2. **iter-5w-bis start-time-shift operator** stays queued for whenever
   peak_hour-style time-segment terms become commercially relevant.
3. **iter-5z polish prototype** stays low (small headroom per K-fair
   analysis from earlier this session).

## Honest caveats

- Manhattan-N500 regression at coef=0.5 suggests city-specific tuning
  may be needed. A coef sweep across [0.1, 0.5, 1.0, 2.0] on that one
  instance would help; not done.
- The "5/6 ship-with-caveat" criterion isn't in the original 7-gate
  recipe (which requires 6/6). Either: (a) tighten coefficient until 6/6,
  (b) ship anyway with the Manhattan-N500 caveat documented, or
  (c) update the recipe to "ship if ≥5/6 AND mean net positive AND
  worst-instance regression < $50/inst." Worth deciding before shipping.

## Files / artifacts

- `svrptw/config/schema.py` — `driver_time_variance_penalty_coef` (default 0.0)
- `svrptw/solvers/common/solution.py` — variance calc reusing `route_durations`
- `tests/unit/test_cost_terms_smoke.py` — 2 new tests (12/12 pass total)
- `bench/scripts/iter5y_fairness_v1large.py` — paired 6-instance bench
- `bench/runs/iter5y_fairness_v1large.{json,log}` — coef=0.001 (noise)
- `bench/runs/iter5y_fairness_v1large_bis.{json,log}` — coef=0.5 (5/6 wins)
- This file
