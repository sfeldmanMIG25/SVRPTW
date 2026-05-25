---
date: 2026-05-15
project: SVRPTW
tags: [iter5u, K-fairness, per_route_fixed_cost, metric-audit, validation]
---

# iter5u — per_route_fixed_cost re-leaderboard + metric K-dependence audit

## TL;DR

Two analyses on existing wholesale data (no new solver runs) validate
iter-5t's K-fairness narrative:

1. **Re-leaderboard with per_route_fixed_cost > 0**: solve_auto's lead
   over fcv2 widens from $2,092 (at $0/route) to $5,142 (at $100/route).
   Every solver loses worse when route count is properly priced.
2. **Per-instance K-dependence (Spearman rho across solvers)**:
   `quality_index` mean |rho| = +0.782 — strongly K-dependent. Other
   metrics under 0.4. The "fcv2 wins quality" framing was almost
   entirely about route count.

## Analysis 1 — fixed-cost re-leaderboard

```
                              fixed_cost=$0    fixed_cost=$50    fixed_cost=$100
  rank  solver               mean_cost_adj    mean_cost_adj      mean_cost_adj
   1    solve_auto                  879.7           2063.1             3246.4
   2    greedy                     1034.2           2200.8             3367.5
   3    regret_3                   1034.2           2200.8             3367.5
   4    fast_construct             1316.7           2583.3             3850.0
   5    pyvrp                      1425.8           2675.8             3925.8
   6    ortools                    1474.2           3015.8             4557.5
   7    fast_construct_v2          2971.4           5679.7             8388.0
   8    lkh3                     750000.0         750000.0           750000.0  (broken)
```

**solve_auto vs fcv2 gap**: $2,092 → $3,618 → $5,142 as fixed cost rises.
**solve_auto vs OR-Tools gap**: $594 → $952 → $1,311.
**solve_auto vs PyVRP gap**: $546 → $613 → $679 (PyVRP only marginally
worse, since both use ~17 routes at N=500).

The interpretation: **fcv2 was getting a 33-route subsidy**; OR-Tools
a 2-route subsidy; PyVRP no subsidy (matched K). Once routes are priced,
solve_auto's structural advantage **compounds** with the fixed cost coefficient.

## Analysis 2 — metric K-dependence (Spearman rho)

Per-instance Spearman correlation between K (n_routes) and each metric,
across the 7 feasible solvers (lkh3 excluded for infeasibility):

```
instance                       quality_index  inter_route_cr  load_util_cv  mean_tw_buffer
OSM-Manhattan-N0500-I000              +0.674          -0.090        +0.405          +0.270
OSM-Manhattan-N1000-I000              +0.623          +0.076        +0.283          -0.057
OSM-Paris-N0500-I000                  +0.860          -0.449        +0.299          +0.337
OSM-Paris-N1000-I000                  +0.711          -0.374        +0.524          +0.000
OSM-SanFrancisco-N0500-I000           +0.936          -0.440        +0.440          +0.294
OSM-SanFrancisco-N1000-I000           +0.891          -0.745        +0.418          +0.745
                                      ------          ------        ------          ------
                MEAN |rho|            +0.782          +0.362        +0.395          +0.284
```

**`quality_index` is strongly K-dependent** (+0.782 mean rho — knowing K
explains ~60% of `quality_index` variance across solvers). The other
three metrics are moderate-to-weak.

**Sign matters**: `quality_index` rho is +ve everywhere (more K → higher
quality). `inter_route_crossings` is mostly -ve (more K → fewer crossings,
as expected). `load_util_cv` is +ve (more K → easier to balance loads).

This decisively confirms the K-artifact mechanism in iter-5t. The
`quality_index` formulation rewards low crossings and tight clusters, both
of which trivially improve as K rises.

## Concrete fixes (queued for next iteration)

1. **Change `Economics().per_route_fixed_cost` default to $50** (matching
   the observed per-route operational cost premium solve_auto pays at K=17
   vs hypothetical K=18). All future benches K-fair by construction.
2. **Add `svrptw.metrics.quality_per_route(sol)` = `quality_index / n_routes`**
   as a K-fair complement metric. Document that `quality_index` is K-dependent
   in its docstring.
3. **Re-publish iter-5q/r/s/t conclusions** with the fixed-cost columns added
   to the headline tables. The cost numbers don't change but the *interpretation*
   becomes much cleaner.

## Strategic implication (sharpened from iter-5t)

The path forward isn't just "cost-model expansion" — it's **start by fixing
the metric**, then expand. Otherwise we'll keep producing artifact-driven
"findings" that don't survive K-fair audit.

Sequence:
1. Fix the metric: per_route_fixed_cost default + quality_per_route metric.
2. Re-validate the headline (all wholesale + leaderboard benches K-fair).
3. THEN add new cost terms (driver breaks, peak-hour penalty, fairness, mixed fleets).

The user's reflection — "the wins are coming from composition decisions" —
applies here too. Setting per_route_fixed_cost > 0 IS a composition: it
changes what the bandit is optimizing for, which in turn changes the basin
the entire solver lands in. This composition is *cheaper* than the
in-loop reward shaping of Phase F (no extra evaluator overhead) and likely
*more impactful* (compounds with K, not just one-shot crossings shape).

## Files / artifacts

- `bench/scripts/iter5u_metric_K_audit.py` — analysis script (reusable)
- `bench/runs/iter5u_metric_K_audit.json` — payload with re-leaderboard + rho
- `bench/runs/iter5u_metric_K_audit.log` — printable summary
- This file
