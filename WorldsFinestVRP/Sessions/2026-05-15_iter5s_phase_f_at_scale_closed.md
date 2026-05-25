---
date: 2026-05-15
project: SVRPTW
tags: [iter5s, phase-F, reward-shaping, closed, cost-quality-split, structural]
---

# iter5s — Phase F at scale: bandit reward shaping does NOT close the split

## TL;DR

Phase F arm E (`crossings_penalty_per_pair=2.0` + `util_imbalance_penalty_coef=20.0`
+ `tw_buffer_bonus_coef=15.0` all combined) at v1_large N=500 across
3 instances: **arm E pays mean +$54/inst cost for ZERO mean quality gain**
(both arms land at exactly q=0.675 mean). The promising N=50 smoke result
from iter-5l (where arm E matched PyVRP quality at $6 less than arm A on
cost) does NOT scale to N=500.

**This is the third composition lever closed in this session.** Combined with
iter-5q (warmstart-source switching closed) and iter-5r (no public solver
beats solve_auto on cost), the **cost-vs-quality split at scale is
structural** — not addressable from inside the LinUCB-bandit + PyVRP
construction architecture.

## Per-instance breakdown (v1_large N=500, 75s budget per cell)

```
instance                         A_cost  A_q     E_cost  E_q   d_cost  d_q
OSM-Manhattan-N0500-I000          766.8  0.725   825.6  0.699  +58.8  -0.026
OSM-Paris-N0500-I000              666.0  0.654   734.9  0.661  +68.9  +0.007
OSM-SanFrancisco-N0500-I000       669.2  0.648   703.2  0.666  +34.0  +0.018
                                  -----  -----   -----  -----  -----  ------
mean                              700.7  0.675   754.6  0.675  +53.9  +0.000
```

Arm E loses all 3 cells on cost. Wins quality on 2/3 cells but the gains
(+0.007, +0.018) are smaller than Manhattan's loss (-0.026), netting zero.

## Why arm E doesn't scale

The N=50 result (iter-5l) had arm A starting at q=0.606 (very low — small
instance with 11 routes, lots of crossings) and arm E pulling it up to
q=0.769 (matches PyVRP). At N=500, arm A's quality is already 0.675 — the
bandit's cost-driven refinement on a PyVRP-warmstart basin lands much closer
to the quality frontier to begin with. Arm E's reward shaping has less to
work with: it can't pull quality up from a low starting point because there
isn't a low starting point at scale.

The intuition: **arm E was correcting for arm A's degraded quality at very
small N**. At large N where arm A is already near the quality frontier of
its basin, arm E only adds friction.

## Three composition levers closed this session

| Iter | Lever | Result |
|------|-------|--------|
| 5q | Warmstart-source switch (fcv2 instead of pyvrp) | pyvrp-warm STRICTLY DOMINATES fcv2-warm both axes 6/6 |
| 5r | Public-solver wholesale | solve_auto strict cost winner vs all (PyVRP 5/6, OR-Tools 6/6, LKH-3 6/6) |
| 5s | In-loop reward shaping (Phase F arm E) | +$54/inst cost for 0 mean quality gain |

Per the user's reflection ("the wins are coming from composition decisions"):
the composition layer is itself now exhausted at this regime. The remaining
composition levers I can think of (untested):
- **Two-phase: cost first then quality polish** — solve_auto for cost, then a
  separate quality-only refinement that only accepts moves preserving cost
  AND improving quality. Architecturally clean; untested.
- **Quality-warmstart with quality-preserving bandit** — fcv2 warmstart, then
  a bandit variant that punishes any quality regression > epsilon. Different
  from arm E because the constraint is hard, not a soft penalty.
- **True multi-objective optimizer** (SPEA2, NSGA-III) — different
  architecture; would replace the LinUCB layer entirely.

## Strategic conclusion

The cost-vs-quality split at scale is **the operating envelope** of the
solve_auto architecture, not a bug to fix. Production recipe is now:
- **Cost-first deployment**: solve_auto (PyVRP construction → LinUCB bandit).
- **Quality-first deployment**: fast_construct_v2 standalone.
- **No single solver wins both at scale on this regime.**

This is publishable as-is. The honest writeup is: composition advantage is
real on cost (40% vs OR-Tools); the cost-vs-quality unification needs a
fundamentally different architecture (multi-objective optimizer or two-phase
polish) that isn't a 1-day experiment.

## What this opens

The next regime is one of:
1. **Two-phase polish prototype** — small experiment (1-2 day): take
   solve_auto's output, then run a quality-only-improving refinement. If
   quality lifts to fcv2's level without cost regression, the unification
   is found via a different composition.
2. **New regime entirely** — multi-objective with explicit logic axis once
   the VLM committee is solid (Phase C revival).
3. **Richer constraints** — driver breaks, hard zones, mixed fleets; this
   would change the cost model in ways that make solve_auto's structural
   advantage compound.

Recommendation: **(1) two-phase polish prototype** is the smallest experiment
that could change the headline. If it works, the writeup gets a unification
result; if it doesn't, the structural-envelope conclusion above is the
final word and we move to (2) or (3).

## Files / artifacts

- `bench/runs/phase_f_at_scale_N500_AE.json` — 6 arm rows (3 instances × 2 arms)
- `bench/runs/phase_f_at_scale_N500_AE.log` — printable log
- `bench/runs/phase_f_at_scale_N500_AE_summary.txt` — analysis output
- `bench/scripts/phase_f_at_scale.py` — launcher
- `bench/scripts/summarize_phase_f_at_scale.py` — analyser (unicode-safe now)
