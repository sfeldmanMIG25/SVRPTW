---
date: 2026-05-16
iter: 7
topic: pyvrp-independent construction at v1_large scale
status: 2 bugs surfaced; Bug 2 fix landed (v2: cost -52%); Bug 1 fix partial
---

## Post-fix headline (iter-7-bis Bug 2 fix landed)

`fast_construct_v2` with the new merge_routes consolidation pass:

| builder | mean cost (n=6) | mean wall | cost vs pyvrp |
|---|---|---|---|
| pyvrp_warm (reference) | $1,034 | 12.81s | 1.00x |
| **fast_construct v2 (post-fix)** | **$1,414** | **13.68s** | **+37%** |
| fast_construct v1 | $1,380 | 55.34s | +33% (but 4x over budget) |
| fast_construct v3 | $6,690 | 1.39s | +547% (savings rejects 99.7% of merges) |

v2 is now a **viable pyvrp-independent construction at v1_large scale**: within
37% of pyvrp on cost at comparable wall. Pre-fix v2 was $2,975 (+188%), so
the merge polish recovered 52% of the cost gap. K dropped from 45-67 to 19-37
across instances (closer to pyvrp's 16-32 optimum).



# iter-7 — construction options at v1_large (N=500/1000)

## Goal

User asked: "make our own construction process instead of relying on pyvrp ...
focus on construction options, scalability and speed ... handle multiple 1000
customer problems at speed ... use items that are readily available like the
dual graph rather than translating to euclidean."

This iteration: bench `pyvrp_warm` (reference) vs `fast_construct` v1 (multi-
start NN+regret+polish) vs `fast_construct_v2` (Louvain-clustered NN) vs the
newly-built `fast_construct_v3` (Clarke-Wright savings on OSM travel-time matrix)
on 6 v1_large instances at N=500 and N=1000.

## Result (mean across 6 instances, b=8s @ N=500 / 16s @ N=1000)

| builder | mean cost | mean wall | mean K (vs pyvrp) | within budget? |
|---|---|---|---|---|
| **pyvrp_warm** (reference) | **$1,034** | 12.84s | 16-32 (optimal) | yes (HGS internals) |
| fast_construct v1 | $1,380 (+33%) | **52.24s** (3-4x over!) | 16-17 (good) | **NO -- 3-4x budget overrun** |
| fast_construct v2 | $2,975 (+188%) | 7.16s | 45-67 (3x too many) | yes |
| fast_construct v3 | $4,400-9,256 (~5-8x worse) | **0.44-2.29s** (very fast) | 72-139 (4-5x too many) | yes |

## Two distinct bugs found

### Bug 1: v1 budget overrun (deadline not enforced inside fill loops)

`fast_construct.solve()` sets a wall deadline and checks it BETWEEN multi-start
iterations, but the expensive inner work is not deadline-aware:

* `_seed_savings` is O(N^2) for the savings list
* `_greedy_fill` is O(N^2 * R) per outer pass with up to N outer passes -> 1.5e9 ops at N=1000

At N=1000 with budget=16s, _greedy_fill alone runs ~77-90 seconds, blowing past
the budget. The deadline check at line 611 only fires AFTER the current start
completes, which means the first start can run essentially unbounded.

**Fix (queued for iter-7-bis)**: thread `deadline` (a `time.perf_counter()`
epoch) through `_seed_*`, `_regret_fill`, `_greedy_fill`. Each inner loop checks
the deadline at top of body and returns whatever's been placed so far. Caller
can then either accept the partial routes or fall back to a faster builder.

### Bug 2: v2 Louvain communities become hard route boundaries

`fast_construct_v2._seed_nn_within_community` starts a fresh route for each
community group, never merging across community boundaries. At v1_large the
OSM Louvain partition produces 45-67 communities, so K_built = 45-67 even
though K_opt ≈ 16-17.

This makes v2 the fastest builder (4.7-10s at N=1000, vs pyvrp's 16-17s) but
wholly unusable for cost.

**Fix (queued for iter-7-bis)**: after the per-community NN sweep, run a
post-construction `merge_routes` pass that's allowed to merge across community
boundaries when TW+capacity permit. Communities should be a clustering signal,
not a partition constraint.

## v3 finding: Clarke-Wright savings alone doesn't work at tight TW

Built `fast_construct_v3.py` (Clarke-Wright savings using `inst.travel_time`
directly -- the graph distances we already have). Results: very fast
(0.44-2.29s) but K is even worse than v2 (72-139 vs v2's 45-67).

Diagnosis from the merge log: of 124,749 candidate savings pairs at N=500,
only 428 (0.34%) passed the merge-feasibility check (capacity + TW + endpoint-
adjacency). The classical Clarke-Wright endpoint-only merge rule is too
restrictive at OSM TW tightness. Once a route grows to length 5+, only its 2
endpoints can accept new merges, but most TW-feasible "next customers" for
those endpoints have already been claimed.

**This is fundamental to savings -- not a fix to v3.** Savings is the wrong
algorithm for tight-TW VRPs at large N.

## What landed this iter-7-bis

* **Fix B (v2 merge polish): LANDED.** `fast_construct_v2.solve()` now runs a
  bounded `merge_routes` polish loop after the per-community NN sweep and
  before the two_opt_intra. 60% of remaining budget is reserved for the
  polish; loop exits when K stops decreasing or budget runs out. K dropped
  from 45-67 to 19-37 across the 6-instance v1_large suite. Cost dropped
  from $2,975 to $1,414 mean (-52%). 11/11 existing fast_construct_v2 smoke
  tests still pass.

* **Fix A (v1 deadline in `_greedy_fill`): partial.** Added optional
  ``deadline`` param to `_greedy_fill` (cooperative cancellation). Threaded
  through `solve()`. Behaviour is bit-identical when deadline=None. At
  N=1000 it doesn't materially help wall (the v1 critical path is also in
  `_seed_savings` + multi-start LS, not just `_greedy_fill`). v1 still
  overruns 4x at large N. Verdict: defer further v1 fixes -- v2 is now the
  pyvrp-independent path of choice at scale.

* **v3 (Clarke-Wright savings): ruled out.** 99.7% of savings pairs were
  rejected by the TW + endpoint constraint. This is fundamental to savings
  at tight TW; not fixable without abandoning the endpoint-merge rule
  (which means it's no longer Clarke-Wright). v3 stays in the tree as a
  ~1.4s sub-second baseline (useful for super-tight-budget single-shot)
  but not as a serious pyvrp replacement.

## Decision: v2 ships as pyvrp-independent option

`fast_construct_v2` is now the recommended PyVRP-independent construction at
v1_large scale:
* Quality: within 37% of pyvrp on mean cost
* Wall: 13.7s mean (vs pyvrp's 12.8s) -- comparable
* Uses OSM dual graph directly (Louvain communities + graph distances)
* No PyVRP dependency
* Scales to N=1000 in under 16s

For a serious pyvrp-match, the next attempt would be Solomon I1 (regret-2
insertion with TW awareness, K close to optimum by construction). But the
current v2 is good enough to be production-viable as an alternative warmstart
when pyvrp isn't available.

## Reproducer

```bash
PYTHONPATH=. python bench/scripts/iter7_construction_v1large.py
# -> bench/runs/iter7_construction_v1large.json
```

## Links

* [[16 - Construction at Scale]]
* [[05 - Decisions & Next Steps]]
