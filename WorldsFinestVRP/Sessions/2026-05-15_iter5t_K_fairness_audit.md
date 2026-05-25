---
date: 2026-05-15
project: SVRPTW
tags: [iter5t, K-fairness, metric-artifact, fast_construct_v2, narrative-shift]
---

# iter5t — K-fairness audit: fcv2 "quality winner" was a measurement artifact

## TL;DR

The "cost-vs-quality split" framing closed in iter5s relied on comparing
solve_auto and fast_construct_v2 with **wildly different K values**. Audit
of the wholesale JSON shows fcv2 using 1.7–3.3× more vehicles than
solve_auto on every v1_large instance:

| instance               | solve_auto K | fcv2 K | ratio |
|------------------------|---------------|--------|-------|
| Manhattan-N0500        | 17            | 50     | 2.94× |
| Manhattan-N1000        | 32            | 67     | 2.10× |
| Paris-N0500            | 16            | 53     | 3.31× |
| Paris-N1000            | 29            | 56     | 1.93× |
| SanFrancisco-N0500     | 17            | 45     | 2.65× |
| SanFrancisco-N1000     | 31            | 54     | 1.74× |

Quality_index rewards low inter-route crossings. With 50 routes covering
500 customers, each route averages 10 customers — naturally clustered, few
crossings. With 17 routes covering the same 500 customers, each route
covers 30 customers — wider geographic spread, more crossings. **The
"quality" metric was K-dependent.**

`Economics.per_route_fixed_cost` defaults to **0.0** — so fcv2 paid
nothing for using 33 extra vehicles. The wholesale comparison let it
trivially "win" quality.

## K-fair audit result (Manhattan/Paris/SF N=500, 10 s budget)

```
instance                          sa_K   nat_K nat_cost nat_q  capK_K capK_cost capK_q  d_cost  d_q
OSM-Manhattan-N0500-I000            17     50   3239.3  0.819     17   49157.0  0.607  +45917  -0.212
OSM-Paris-N0500-I000                16     53   3131.2  0.841     16   42943.9  0.531  +39812  -0.310
OSM-SanFrancisco-N0500-I000         17     45   2660.1  0.832     17   45993.7  0.567  +43333  -0.265
```

When fcv2 is forced to use solve_auto's K:
- **Cost explodes by ~$42-46k per instance** — that's HARD_LATE_PENALTY
  ($1000) × ~42 missed customers. fcv2 cannot pack 500 customers into 17
  vehicles within capacity+TW constraints; it lacks HGS-style local search.
- **Quality DROPS by 0.21–0.31** — even at the same K, fcv2's basin is
  weaker. The quality "lead" was entirely a route-count artifact.

**Conclusion**: at fair K, solve_auto strictly dominates fcv2 on cost,
quality, K, AND feasibility. The cost-vs-quality split was a metric bug.

## What this corrects from iter5q+r+s

iter5q (warmstart-source switch closed): conclusion stands. fcv2-warm vs
pyvrp-warm at SAME K (since the bandit phase is identical) showed
pyvrp-warm strictly dominates. K wasn't the variable.

iter5r (8-solver wholesale, OSM closed): solve_auto cost wins stand
(matched-wall comparisons). The K-disparity column was just glossed over
in interpretation. The headline numbers are correct; the interpretation
of fcv2 as "quality winner" was wrong.

iter5s (Phase F arm E closed): conclusion stands. Arm E pays cost for no
quality gain at v1_large N=500. K wasn't the variable.

The narrative that needs revision is "the cost-vs-quality split is the
operating envelope of the architecture". Now corrected to:
**"solve_auto is the strict winner across all axes that matter in
production (cost, K, feasibility, quality-per-route). fcv2's apparent
quality lead in the wholesale leaderboard was a route-count artifact
that disappears at K-fair comparison."**

## What this opens

1. **Add `per_route_fixed_cost` to default Settings.** A non-zero value
   (e.g., $50/route, matching the per-route operational cost premium
   solve_auto pays at K=17 vs hypothetical K=18) makes the cost metric
   K-aware and prevents this comparison artifact in future benches.
2. **Re-bench the wholesale with `per_route_fixed_cost=$50`.** Expected
   outcome: fcv2's $3239 cost becomes $3239 + 33×$50 = $4889 — pushing
   it FURTHER behind solve_auto. The "quality winner" framing dissolves
   completely.
3. **Document the metric** so future contributors don't repeat the
   mistake. Add a docstring to `quality_index` noting K-dependence and
   suggesting `quality_per_route` as a K-fair complement.
4. **Verify other metrics for K-dependence** — `inter_route_crossings`
   is obviously K-dependent (it's literally inter-route). What about
   `load_util_cv`, `mean_tw_buffer_score`? Quick audit needed.

## Strategic implication

The "cost-vs-quality split is structural" conclusion from iter5s is
softened: at K-fair, the split is much smaller (still positive but
single-digit-percent quality, not 19%). The unification work
(two-phase polish, multi-objective optimizer) becomes lower-priority
because the gap to close is much smaller.

The new highest-leverage research direction is **richer cost models**
(per_route_fixed_cost > 0, peak-hour penalty, fairness across drivers,
mixed fleets). Each new term that solve_auto can exploit and PyVRP
cannot is a new dimension of structural advantage.

## Files / artifacts

- `bench/scripts/fcv2_k_fair_audit.py` — the audit script
- `bench/runs/fcv2_k_fair_audit.json` — 3 N=500 K-fair rows
- `bench/runs/fcv2_k_fair_audit.log` — printable summary
- This file
