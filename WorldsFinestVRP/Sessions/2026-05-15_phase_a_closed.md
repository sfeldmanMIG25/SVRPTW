---
date: 2026-05-15
project: SVRPTW
tags: [phase-A, crystallization, writeup, cb-scaling, complete]
---

# Phase A — Crystallization Closed (writeup data complete)

## TL;DR

Phase A is done. Three benches confirm the architectural win is **bigger and cleaner**
than the prior session reported, AND it now extends to N=500 cleanly:

```
v1 OOD       (held-out I003+I004): 48/48 wins (100%) mean +$165.65/instance
v1 leaderboard (I000+I001 mix):    24/24 wins (100%) mean +$102.96/instance
Homberger N=400 (academic):        14/24 wins ( 58%) mean +$129.54/instance
TOTAL real-world OSM:             72/72 wins (100%) mean +$143/instance
```

**cb-scaling fixed the N=400 reversal**: prior 7/24 (29%) mean −$283 → 14/24 (58%) mean +$129.

## A1 — Scale pyvrp_construction_budget with N (DONE)

Implementation: `svrptw/solvers/classical/portfolio_pyvrp_warm.py::scaled_cb()`
- N ≤ 200: returns base cb=8.0 (preserves prior tuning)
- N > 200: linear scale `cb = 8.0 * N / 200` (so N=400 → 16s, N=500 → 20s, N=800 → 32s)
- Downstream `solve()` clamps at `min(cb, 0.30 * budget_seconds)` to keep it sane

## A2 — Re-bench at all 4 N on truly held-out OSM (DONE)

```
N      n  W  L   mean Δ ($)
100   16 16  0  +144.53
200   16 16  0  +105.89
500   16 16  0  +246.52
total 48 48  0  +165.65 mean
```

**100% wins everywhere.** Compared to prior session's 94% at N=500 (11/16):
- Win rate at N=500: **11/16 (69%) → 16/16 (100%)**
- Mean Δ at N=500: **+$114 → +$246** (more than doubled)

cb-scaling didn't just close the gap; it grew the lead substantially.

### Homberger N=400 (the original failure case)

Prior session: 7/24 wins (29%), mean −$283. With cb-scaling:

```
N=400  n=24  W=14  L=10  mean +$129.54

per-class:
  C1   1/3  −$35    PyVRP holds tight-cluster
  C2   2/2  −$4     tied
  R1   2/2  −$1667  bimodal: 2 wins + 2 catastrophic losses
  R2   4/0  +$1399  warm dominates
  RC1  1/3  +$134   mixed
  RC2  4/0  +$950   warm dominates
```

**Honest reading**: warm wins decisively on loose-TW classes (R2, RC2, 8/8 W). PyVRP holds tight-clustered (C1, RC1, 2/6 W). R1 is bimodal — R1_4_2 (−$3.8k) and R1_4_4 (−$3.1k) are the two catastrophic losses dragging the mean. Worth a focused investigation in a follow-up. The OSM v1 OOD set doesn't have these failure modes.

## A3 — v1 leaderboard refresh via solve_auto + tuned defaults (DONE)

```
N      n  W  L   mean Δ ($)
50     8  8  0  +87.20
100    8  8  0  +89.33
200    8  8  0  +132.34
total 24 24  0  +102.96 mean
```

**100% wins.** This refreshes the original session's 95/65/45/72.5% Pareto numbers
(which were generated before the architectural win + cb-scaling). The 2-tier
dispatcher (`solve_auto` with vanilla N<100 + warm N≥100 + cb-scaling above N=200)
is the production recipe.

## What Phase A leaves open (and that's OK)

- **R1 bimodality at Homberger N=400**: 2/4 catastrophic losses on tight-random.
  Likely something about the bandit's basin escape on those specific instances.
  Worth a focused look in a follow-up but doesn't block the writeup.
- **fast_construct doesn't strictly dominate PyVRP**: iter-5h grid bench confirmed
  PyVRP's basin quality is real and hard to replicate in Python-native at matched
  wall. Construction-bypass is HALF won (cost) but not full won (quality).

## What Phase A unblocks

Per the user's pasted reflection (item 4): "only after the first three are done, pick the
next research direction." The two candidates:

### Option 1: Cost-model exploration ← recommended
- The iter-5f metrics suite at `svrptw/metrics/quality.py` already operationalizes 14
  per-route + 11 per-solution metrics (silhouette-like, intra/inter, hull overlap,
  load CV/gini, route crossings, detour ratio, TW buffer, etc).
- Each new term that LinUCB-bandit can optimize but PyVRP cannot is a structural
  lever that compounds the architectural advantage. `per_route_fixed_cost` is one;
  the metrics suite has 14 more candidates.
- Concrete next move: pick 3 cost terms (crossings penalty, util-imbalance penalty,
  TW-buffer bonus) and turn them into opt-in fields on `Settings.economics`. Bench
  `solve_auto` with each enabled vs PyVRP.

### Option 2: Composition exploration
- Composition is the layer where the architectural win actually came from
  (PyVRP construction → bandit refinement is itself a composition).
- Repurpose the LLM proposer to generate meta-moves: "run swap_star until plateau,
  then trigger destroy_island once, then resume" rather than new operators.
- The 25-proposal rejection corpus is a dataset; the proposer plumbing works.

I'm picking **Option 1 (cost-model)** for iter-5l because:
1. The metrics suite is fresh and we know its terms work
2. Each term that compounds is a publishable result on its own
3. The bench harness is identical to what just ran for Phase A

## Files
- `bench/runs/cb_scaled_rebench.json` — v1 OOD 48 results
- `bench/runs/cb_scaled_rebench_homberger_n400.json` — Homberger N=400 24 results
- `bench/runs/v1_leaderboard_solve_auto.json` — v1 leaderboard 24 results
- `WorldsFinestVRP/Progress_Report.html` — full writeup with all 3 leaderboard tables embedded
- `WorldsFinestVRP/15 - Cost-Model Exploration.md` — next-direction spec (writing now)

## Cross-references
- [[01 - Progress Report]] — high-level snapshot
- [[03 - Benchmarks]] — leaderboard (will update with iter-5i/j/k numbers)
- [[12 - Architecture Review]] — pre-A4 architecture state
- [[15 - Cost-Model Exploration]] — next-direction spec
- [[Progress_Report.html]] — canonical writeup (iframe in dashboard)
