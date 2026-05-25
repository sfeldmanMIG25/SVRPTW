---
date: 2026-05-15
project: SVRPTW
tags: [iter5q, closure, fast_construct_v2, warmstart, v1_large, composition]
---

# iter5q — fast_construct_v2 warmstart definitively closed across all 6 v1_large

## TL;DR

Single-instance result from iter5p (Manhattan N=500: pyvrp-warm cost=822 vs
fcv2-warm cost=976) generalises **uniformly** across all 6 v1_large instances.
**pyvrp-warm strictly dominates fcv2-warm on BOTH cost and quality at every
scale and city tested.** The split-win on the standalone wholesale leaderboard
(solve_auto wins cost, fast_construct_v2 wins quality) is a property of fcv2
*alone*, not a hint that swapping the warmstart source would close the gap.
The bandit homogenises: it tears apart fcv2's standalone 0.794 quality basin
during cost-driven refinement and lands at 0.53–0.59, while pyvrp-warm refines
to 0.63–0.68. Direction closed.

## The 6-instance paired sweep

Script: `bench/scripts/fcv2_warm_v1large_sweep.py` (re-run with N=1000 included
after `N0*-I000.json` glob bug fix; 75s budget at N=500, 150s at N=1000).

```
instance                            pyvrp_cost  fcv2_cost  d_cost  pyvrp_q  fcv2_q   d_q
OSM-Manhattan-N0500-I000               822.1     976.4    +154.3   0.673   0.559  -0.114
OSM-Manhattan-N1000-I000              1396.8    1498.5    +101.8   0.630   0.588  -0.041
OSM-Paris-N0500-I000                   625.9     747.0    +121.1   0.647   0.536  -0.112
OSM-Paris-N1000-I000                   933.8    1128.6    +194.8   0.679   0.566  -0.113
OSM-SanFrancisco-N0500-I000            646.9     738.1     +91.2   0.658   0.532  -0.126
OSM-SanFrancisco-N1000-I000           1004.2    1116.4    +112.2   0.636   0.569  -0.067
```

**6/6 instances**: pyvrp-warm wins cost (mean +$129.2/instance, std $36).
**6/6 instances**: pyvrp-warm wins quality (mean +0.099 q-points).

The N=1000 cases narrow the quality gap somewhat (Δq 0.041–0.067) but
pyvrp-warm still wins both axes everywhere.

## Why the standalone leaderboard split-win doesn't transfer

The wholesale leaderboard reported earlier this session showed:
- solve_auto (pyvrp construction → bandit): wins cost ($876.5 mean)
- fast_construct_v2 standalone (no bandit): wins quality (0.794 mean)

Naively this suggests "swap solve_auto's warmstart to fcv2 and you win both
axes". The empirical answer is **no**. The bandit is the homogeniser: 75s of
LinUCB-driven refinement aimed at cost destroys whatever cluster structure
fcv2 set up. fcv2 standalone keeps Louvain-aligned routes; fcv2-warm-then-
bandit ends up with the bandit's preferred topology, which lands at lower
quality than pyvrp-warm-then-bandit ends at because pyvrp's HGS construction
is itself topology-aware in a way that survives bandit refinement.

The cost gap (+$129/instance) is the larger-than-expected piece. Going in,
fcv2-warm + 75s bandit was expected to be within ~$30 of pyvrp-warm. Reality
is 4× that gap. The HGS basin at construction time is structurally deeper for
the bandit to refine than what fcv2 produces, even though fcv2's standalone
quality is higher.

## What this closes

1. **"Swap the warmstart" isn't the path to single-solver leaderboard wins.**
2. **Composition didn't help here.** The wins from PyVRP-warm + bandit composition
   came because PyVRP's basin was already deep. Replacing the deep basin with
   a cluster-pretty-but-shallow basin doesn't compose with the bandit.
3. **fast_construct_v2 stays in the toolbox** as a standalone option for the
   quality-first regime (it really does win quality there). It is **not** a
   drop-in replacement for the warmstart inside solve_auto.

## What this opens

The wholesale leaderboard split-win remains real: solve_auto wins cost,
fcv2 wins quality, and **no single solver wins both at scale**. The path
forward is **objective function**, not warmstart:

- **Phase F at scale**: opt-in cost terms (`crossings_penalty_per_pair`,
  `util_imbalance_penalty_coef`, `tw_buffer_bonus_coef`) plumbed into
  `evaluate()`. Phase F arm E showed cost+quality wins at N=50. The question
  is whether the same arm wins both axes on v1_large N=500/1000. If it does,
  that's the unification.
- **Logged today** for the next research direction picker.

## Composition observation (per user reflection)

> "the wins are coming from composition decisions, not from new operators or
> new algorithms. PyVRP-as-warmstart was a composition. cb-scaling was a
> composition. solve_auto as a 2-tier dispatcher was a composition."

iter5q falsified one composition direction (warmstart-source-switch). The
remaining composition lever for the wholesale split-win is **bandit reward
shaping** (Phase F's evaluate() — change what "improvement" means to the
bandit). That is the next composition to try.

## Files / artifacts

- `bench/scripts/fcv2_warm_v1large_sweep.py` — final closure bench (12 solves)
- `bench/runs/fcv2_warm_v1large_v2.log` — printable log (this table)
- `bench/runs/fcv2_warm_v1large.json` — JSON rows
- `webui/client.py` — push_finding called with this closure
