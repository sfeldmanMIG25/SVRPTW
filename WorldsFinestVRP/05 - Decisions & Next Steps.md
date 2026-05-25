# Decisions & next steps

## Latest finding (2026-05-16 iter-7-bis: pyvrp-independent construction shipped)

`fast_construct_v2` now ships as a viable PyVRP-independent construction at
v1_large scale. Mean cost on 6-instance N=500/1000 v1_large suite: **$1,414
vs pyvrp's $1,034 (+37% cost) at 13.7s vs 12.8s wall (comparable).**

Pre-fix v2 was $2,975 (+188%) -- the Louvain community boundaries were acting
as hard route partitions, fragmenting K to 45-67 routes vs pyvrp's 16-32.
The fix: add a bounded `merge_routes` polish pass after the per-community
NN sweep that's free to merge across community boundaries when TW+capacity
permit. K dropped from 45-67 to 19-37. Cost dropped 52%. Wall actually
INCREASED slightly (7.1s -> 13.7s) because the polish does real work.

This addresses the unfinished half of the original /loop ask: "make our own
construction process instead of relying on pyvrp ... handle multiple 1000
customer problems at speed." v2 uses the OSM dual graph directly (no
Euclidean translation) and scales to N=1000 in under 16s. See
[[Sessions/2026-05-16_iter7_construction_at_scale]].

Two other construction options also benched:
* `fast_construct` v1: $1,380 (+33% cost) but **4x budget overrun** at large N
  (deadline only enforced between multi-starts, not inside fill loops).
  Partial deadline-fix landed in `_greedy_fill` but doesn't materially help
  -- the v1 critical path is also in `_seed_savings`. Defer further v1
  fixes; v2 is now the pyvrp-independent path of choice.
* `fast_construct_v3` (Clarke-Wright savings on OSM travel-time matrix):
  RULED OUT. 99.7% of savings pairs were rejected by TW + endpoint
  constraint. Fundamental to savings at tight TW; not fixable without
  abandoning the endpoint-merge rule. Stays in tree as a 1.4s sub-second
  baseline.

## Latest finding (Homberger N=400 reversal)

The architectural advantage peaks at N=200 and **reverses at N=400**: warm wins only 7/24 (29%), mean -$283/instance. Largest losses on R class. Fixed `pyvrp_construction_budget=8s` is undersized at large scale — PyVRP@120s converges fully while warm spends 8s on construction + 52s on bandit, insufficient to catch up.

This is honest external-validation noise — the architectural advantage is real at N=100-200 but doesn't extend monotonically to N=400. **Production recipe needs N-aware tuning** (either cb-scaling or vanilla fallback at very large N).

## Decision log (this session, in order)

1. **Composability bench spec amendment** (SPEC-8-COUNCIL-02) — user signed off. Tests "13 arms + new arm vs 13 alone at same wall budget" instead of "operator beats portfolio@10s in one shot."
2. **Drop the unit-test gate** from council proposal validation — LLM-generated tests hallucinated constructor signatures. Smoke gate is authoritative.
3. **Add API CHEAT SHEET** to LLM proposer prompt — fixed n=10 batch from 0/10 to 2/10 schema pass.
4. **Paired seeding in composability bench** — threaded `seed=` through `portfolio.solve()` → `LinUCBBandit` → `SISR`. Removed the ~$8 exploration-variance noise floor.
5. **Bumped composability subset 8 → 16 instances** — defeats selection bias.
6. **Wrote off logic-axis** — single-judge labelling lacks resolution to discriminate strong solvers.
7. **Wrote off POMO/EAS as warmstart** (fusion) — checkpoint structurally weaker than auction_gart's.
8. **Adopted PyVRP-warmstart** as the architectural win. Validated across N×instance regime.
9. **Tuned cb=8 + plateaus_to_stop=20** — captures full 30s budget; +$16 per N=200 instance over old defaults.
10. **Refined N-recipe to 2-tier** (was 3-tier): vanilla N<100, warm N≥100. Earlier "vanilla at N≥300" was a pre-tuning artifact.
11. **Mandated parallel-bench harness** — `bench/parallel.py` with max_workers=4; bench wall time cut 3-4×.
12. **External benchmark validation** — Solomon C101 hits published optimum, Homberger N=200 wins 83% at half-budget.

## 2026-05-15 iter 5 — STRATEGIC PIVOT: algorithm primary, dash locked

User direction: "developing dash should not interrupt algorithm development from harness side + RL training, I am convinced we can still optimize construction to bypass the need for pyvrp convergence."

Two parallel sub-agents running in background:
1. **Phase E** (new): construction-bypass research — characterize all existing constructions, prototype `fast_construct` multi-start. Spec at [[13 - Construction Bypass]].
2. **Phase D pipeline run**: collect bandit logs → train MLP → A/B vs LinUCB end-to-end.

Dash is locked in (action bar + 6 buttons + viewer + control endpoints + chart panel + iframe report). No more dash cycles.

## 2026-05-15 iter5y -- cross-route fairness: 5/6 wins (and Step 0.5 added)

First attempt at coef=0.001: 3/6 wins, mean -\$0.001 (baseline variance
penalty \$0.10 across ALL 6 -- noise floor, no gradient). Re-bench at
coef=0.5 (calibrated to ~5% of ops cost): **5/6 wins, mean +\$36.1/inst**.

```
instance              b_K s_K  b_ops  s_ops  b_var_pen s_var_pen     net
Manhattan-N0500        17  17  829.0  859.9    33.6     10.2        -7.5
Manhattan-N1000        32  32 1329.7 1303.6    50.9     31.7       +45.3
Paris-N0500            16  16  658.5  655.9    90.7     11.4       +81.9
Paris-N1000            30  30 1013.8 1005.3    60.3     30.6       +38.2
SanFrancisco-N0500     17  17  662.1  693.6    91.7     21.9       +38.3
SanFrancisco-N1000     31  31 1032.4 1034.2    53.0     30.5       +20.6
                                                              mean: +\$36.1
```

K unchanged across all 6 (fairness operates within fixed K via
cross-route operators). Variance penalty cut to ~1/3 on most cells.
Single loss: Manhattan-N500 (-\$7.5, borderline).

**Recipe Step 0.5 (iter-5y addition)**:
> "Calibrate the coefficient so baseline penalty is ~3-10% of ops cost.
> Too small -> bandit treats term as noise. Smoke-evaluate one solve_auto
> baseline with the proposed coef BEFORE firing the paired bench."

**Three structural categories now tested**:
| iter | axis | verdict |
|------|------|---------|
| 5w | per-route | 6/6 wins, mean +\$155 |
| 5x | per-segment-of-time | 2/6 boundary (no time-shift operator) |
| 5y | cross-route | 5/6 wins, mean +\$36 |

The recipe + Step 0 + Step 0.5 + 7 gates = mature framework.

See [[Sessions/2026-05-15_iter5y_fairness_term]].

## 2026-05-15 iter5x — peak_hour: COUNTEREXAMPLE -- recipe-boundary found

Recipe gate 7 FAILED for second cost term. 2/6 wins, mean -\$12.2/inst.
4 of 6 instances have IDENTICAL peak surcharge baseline vs peaked --
bandit could not reduce peak overlap at all.

```
instance              b_K s_K  b_ops  s_ops  b_peak  s_peak     net
Manhattan-N0500        17  17  768.8  770.2  246.5   246.5     -1.3
Manhattan-N1000        32  33 1276.4 1329.9  464.0   478.5    -68.0
Paris-N0500            16  16  606.7  623.3  232.0   232.0    -16.6
Paris-N1000            30  30  973.2  963.9  435.0   435.0     +9.3
SanFrancisco-N0500     16  16  637.0  627.8  232.0   232.0     +9.2
SanFrancisco-N1000     31  31 1011.8 1017.7  449.5   449.5     -5.9
                                                       mean:  -\$12.2
```

**Root cause**: depot.ready=480 (8am) for all v1 instances, exactly
inside the 8-10am peak window. All routes start in peak. The bandit's
13-arm operator set (relocate / swap / 2-opt / 2-opt-star / SISR /
merge / split) reorders customers within and across routes but NONE
shift route start times. Per-segment-of-time constraints need
time-segment-axis operators that don't exist in the current arm set.

**Recipe correction (Step 0 added)**:
> "Check whether the bandit's operator set acts on the axis the proposed
> cost term targets. If yes -> proceed with the 7 gates. If no -> either
> add the missing operator first, or pick a different term whose axis
> the existing operators DO cover."

**Strategic value**: this counterexample is more informative than another
6/6 confirmation. The recipe's boundary is located: it generalises to
constraints the bandit operators can ACT on, not all real-world cost
terms automatically.

**Re-prioritized queue**:
1. iter-5y cross-route fairness MOVED UP (cross-route property matches
   existing cross-route operators -- prerequisite met).
2. iter-5w-bis add start-time-shift operator (~2 days). Required before
   peak_hour-style terms become viable.
3. iter-5z two-phase polish unchanged.
4. iter-6a richer constraints unchanged.

See [[Sessions/2026-05-15_iter5x_peak_hour_recipe_boundary]].

## 2026-05-15 iter5w — shift_overrun: 6/6 paired wins on v1_large, mean +\$155.2/inst

Paired-seed bench across all 6 v1_large instances confirms iter-5v's
single-instance result generalises cleanly:

```
instance                     b_K  s_K  b_wShift  s_wShift     net
Manhattan-N0500-I000          17   18    2812.2    2696.5   +115.8
Manhattan-N1000-I000          33   34    5016.5    4964.4    +52.0
Paris-N0500-I000              16   17    2465.2    2300.3   +164.9
Paris-N1000-I000              30   33    4532.9    4305.8   +227.1
SanFrancisco-N0500-I000       17   18    2330.9    2237.1    +93.8
SanFrancisco-N1000-I000       30   33    4639.2    4361.5   +277.7
                                                            -------
                                                      mean: +\$155.2
```

6/6 wins, no regressions. N=500 cells add 1 route consistently; N=1000
cells add 1-3 routes. The bandit reliably finds the trade-off when the
cost signal is present.

**Cost-model-expansion recipe VALIDATED end-to-end** (iter-5u thesis):
opt-in `Economics` field + gated penalty in `evaluate()` -> bandit picks
it up at scale. shift_overrun is shippable as a production option.

**Next term (iter-5x queued)**: peak-hour penalty (8-10am, 5-7pm = 1.5x
wage). Real-world relevant, PyVRP can't see it, same recipe.

See [[Sessions/2026-05-15_iter5w_shift_overrun_6inst_validation]].

## 2026-05-15 iter5v — shift-overrun cost term: first concrete validation of cost-model expansion

Added opt-in `Economics.shift_max_minutes` + `shift_overrun_penalty_per_min`.
Per-route duration tracked in `evaluate()`; gated penalty applies only when
both coefs > 0 (bit-identical default). 8/8 cost-term tests pass.

Bench on Manhattan-N500 (cap=300min, pen=\$1/min, 2x75s):

```
solver     K   ops_cost  overrun_$  cost_w_shift
baseline  17    \$765.8    \$2039.1     \$2804.9
shifted   18    \$826.2    \$1869.1     \$2695.3   <- net -\$109.6
```

Bandit added 1 route (K 17->18) to spread load, paying +\$60 ops cost to
reduce overrun by \$170. Under shift-aware objective, shifted solver saves
\$109.6/inst. **First concrete validation of iter-5u thesis**: each new
opt-in term that solve_auto can optimize and PyVRP cannot is a new
dimension of structural advantage.

Caveats: 1-instance smoke; needs paired 6-instance validation (queued
iter5w). +8.3% overrun reduction is modest; tighter cap or higher penalty
might unlock larger gains.

See [[Sessions/2026-05-15_iter5v_shift_overrun_term]].

## 2026-05-15 iter5u — per_route_fixed_cost validates K-fairness; quality_index rho=+0.78

Two analyses on existing wholesale JSON (no new solver runs):

**Re-leaderboard at fixed costs $0/$50/$100**:
```
solver               $0      $50     $100
solve_auto         879.7   2063.1   3246.4
fast_construct_v2 2971.4   5679.7   8388.0   <- gap to solve_auto: 2092 -> 3618 -> 5142
ortools           1474.2   3015.8   4557.5   <- gap: 594 -> 952 -> 1311
pyvrp             1425.8   2675.8   3925.8   <- gap: 546 -> 613 -> 679 (matched K)
```

**solve_auto's lead WIDENS with per_route_fixed_cost** because every other
solver uses more vehicles. fcv2 the most (33 extra), OR-Tools middling
(~2 extra), PyVRP matched. Once routes are priced, solve_auto's structural
advantage compounds with the fixed-cost coefficient.

**Metric K-dependence (Spearman rho mean across 6 instances)**:
- `quality_index` = **+0.782** (strongly K-dependent -- the artifact)
- `inter_route_crossings` = 0.362 (moderate)
- `load_util_cv` = 0.395 (moderate)
- `mean_tw_buffer_score` = 0.284 (least)

`quality_index` is the smoking gun. Knowing K explains ~60% of variance
across solvers. Other metrics are tolerable.

**Concrete fixes (queued)**:
1. Change `Economics().per_route_fixed_cost` default 0.0 -> $50.
2. Add `svrptw.metrics.quality_per_route(sol) = quality_index / n_routes`
   as a K-fair complement.
3. Document `quality_index` K-dependence in its docstring.

**Strategic sequence**: fix the metric -> re-validate the headline ->
THEN add new cost terms (driver breaks, peak-hour penalty, fairness,
mixed fleets). Skipping (1) and (2) means future findings will keep
producing K-artifacts.

See [[Sessions/2026-05-15_iter5u_fixed_cost_validates_K_fairness]].

## 2026-05-15 iter5t — K-fairness audit: fcv2 "quality winner" was a measurement artifact

Audit of wholesale JSON shows fcv2 uses **1.7-3.3x more vehicles** than
solve_auto on every v1_large instance:

```
                       sa_K  fcv2_K  ratio
Manhattan-N0500          17     50  2.94x
Manhattan-N1000          32     67  2.10x
Paris-N0500              16     53  3.31x
Paris-N1000              29     56  1.93x
SanFrancisco-N0500       17     45  2.65x
SanFrancisco-N1000       31     54  1.74x
```

`quality_index` rewards low inter-route crossings -- with 50 routes
covering 500 customers each route averages 10 customers (naturally
clustered). With 17 routes each covers 30 (more spread, more
crossings). **The "quality" metric is K-dependent.** `Economics.per_route_fixed_cost`
defaults to **0.0** so fcv2 paid nothing for using 33 extra vehicles.

K-fair test (fcv2 capped at sa_K, 10s budget):

```
                          sa_K  capK_cost  capK_q   d_cost   d_q
Manhattan-N0500             17    49157.0   0.607  +45917  -0.212
Paris-N0500                 16    42943.9   0.531  +39812  -0.310
SanFrancisco-N0500          17    45993.7   0.567  +43333  -0.265
```

At fair K, **cost EXPLODES \$42-46k (infeasibility -- fcv2 cant pack tight)**
AND **quality DROPS 0.21-0.31**. fcv2's "quality lead" was entirely a
route-count artifact.

**Narrative correction**: the "cost-vs-quality split is structural" claim
from iter5s is softened. At K-fair, solve_auto strictly dominates fcv2
on cost, K, feasibility AND quality. The split was a metric bug.

Next moves:
1. Add `per_route_fixed_cost > 0` to default Settings (e.g., \$50/route)
   and re-bench wholesale; expected: fcv2 falls further behind.
2. Document `quality_index` K-dependence; add `quality_per_route` complement.
3. Audit other metrics for K-dependence (`load_util_cv`, etc.).
4. Pivot research direction to **cost-model expansion** (driver breaks,
   peak-hour penalty, fairness across drivers, mixed fleets) -- each new
   term that solve_auto can exploit and PyVRP cannot is a new dimension
   of structural advantage.

See [[Sessions/2026-05-15_iter5t_K_fairness_audit]].

## 2026-05-15 iter5s — Phase F at scale: in-loop reward shaping CLOSED

Phase F arm E (crossings_penalty + util_imbalance + tw_buffer_bonus combined)
on v1_large N=500 across 3 instances:

```
                              A_cost  A_q     E_cost  E_q   d_cost  d_q
OSM-Manhattan-N0500-I000      766.8  0.725   825.6  0.699  +58.8  -0.026
OSM-Paris-N0500-I000          666.0  0.654   734.9  0.661  +68.9  +0.007
OSM-SanFrancisco-N0500-I000   669.2  0.648   703.2  0.666  +34.0  +0.018
mean                          700.7  0.675   754.6  0.675  +53.9  +0.000
```

**Arm E pays mean +$54/inst cost for ZERO mean quality gain.** The N=50
smoke result (where arm E matched PyVRP quality at lower cost) does not
scale -- at N=500, arm A's quality is already near the basin frontier
(0.675), so arm E's reward shaping has nothing to lift up.

**Three composition levers closed this session:**
1. iter5q: warmstart-source switching (fcv2 dominated by pyvrp)
2. iter5r: 8-solver wholesale (solve_auto strict cost winner)
3. iter5s: in-loop reward shaping (arm E adds friction, no quality gain)

**Strategic conclusion:** the cost-vs-quality split at scale is the
*operating envelope* of the solve_auto architecture. Production recipe:
solve_auto for cost-first, fast_construct_v2 for quality-first. The
unification needs a fundamentally different architecture (two-phase
polish, multi-objective optimizer) -- not a 1-day experiment.

Next regime candidates (in order of attractiveness):
1. Two-phase polish prototype (1-2 day): solve_auto -> quality-only
   refinement that only accepts cost-preserving + quality-improving moves.
2. New regime: multi-objective with VLM committee (Phase C revival).
3. Richer constraints: driver breaks, hard zones, mixed fleets.

See [[Sessions/2026-05-15_iter5s_phase_f_at_scale_closed]].

## 2026-05-15 iter5r — 8-solver wholesale: OSM regime fully closed

Long-promised "world-class solver" question closed. 8 solvers × 6 v1_large
instances at matched wall budget (75s for N=500, 150s for N=1000), 55 min total:

```
solve_auto vs PyVRP    : 5/6 wins, mean +$546/inst
solve_auto vs OR-Tools : 6/6 wins, mean +$594/inst (-40% cost)
solve_auto vs LKH-3    : 6/6 wins (LKH-3 0% feasible -- WIRING BUG filed)
Pareto check: only PyVRP dominates_both on 1/6 (SF-N500 by $11)
```

**solve_auto sits on the cost frontier vs every public solver tested.** Mean
cost $879.7 vs OR-Tools $1474.2, vs PyVRP $1425.8 (PyVRP also infeasible 2/6
at N=1000). Composition advantage is real and externally validated.

Honest follow-ups: LKH-3 wiring bug (returned $1M = HARD_LATE_PENALTY × N
missed); regret_3 budget violation (1500s on N=1000, 10× over budget,
identical output to greedy); PyVRP feasibility regression at N=1000 (raw
PyVRP @ 150s misses 2/6, solve_auto 100%).

See [[Sessions/2026-05-15_iter5r_wholesale_OSM_closed]].

## 2026-05-15 iter5q — fast_construct_v2 warmstart definitively closed

Single-instance smoke from iter5p generalised to all 6 v1_large instances
via paired sweep (`bench/scripts/fcv2_warm_v1large_sweep.py`):

```
                            pyvrp_warm  fcv2_warm   d_cost      d_q
Manhattan  N0500  cost/q     822 0.673  976 0.559   +154.3   -0.114
Manhattan  N1000  cost/q    1397 0.630 1499 0.588   +101.8   -0.041
Paris      N0500  cost/q     626 0.647  747 0.536   +121.1   -0.112
Paris      N1000  cost/q     934 0.679 1129 0.566   +194.8   -0.113
SF         N0500  cost/q     647 0.658  738 0.532    +91.2   -0.126
SF         N1000  cost/q    1004 0.636 1116 0.569   +112.2   -0.067
```

**6/6 instances**: pyvrp-warm strictly dominates fcv2-warm on BOTH cost (mean
+$129.2/inst, $36 std) AND quality (mean +0.099 q-points). The wholesale
leaderboard split-win (solve_auto wins cost, fast_construct_v2 wins quality)
is a property of fcv2 standalone — the bandit homogenises. Refining fcv2's
Louvain-pretty basin with 75s of LinUCB lands at 0.53–0.59 quality, *worse*
than refining PyVRP HGS's basin to 0.63–0.68. Composition lever for the
split-win is now **bandit reward shaping (Phase F at scale)**, not warmstart.

See [[Sessions/2026-05-15_iter5q_fcv2_warm_closure]].

## 2026-05-15 epic — multi-phase build (in flight)

User /loop kicked off a multi-phase epic combining the inline reflection
("crystallize first") with Human Reflection 10 (web UI, multi-judge VLM,
RL for neighborhoods). Iteration 2 (this one) crystallized:
- **A1** ✓ cb-scaling in `solve_auto` via `scaled_cb()`
- **A2** ✓ `bench/scripts/cb_scaled_rebench.py`
- **A3** ✓ `bench/scripts/v1_leaderboard_solve_auto.py`
- **B0** ✓ `webui/` package with cross-process HTTP bridge
- **B1** ✓ `render_llm_compare()` LLM-comp safe render mode
Pending: B2 (per-iteration hook), C1-C4 (multi-judge), D1-D5 (RL),
A4 (writeup once benches run), B3 (time-lapse), B4 (OSM basemap).
See `Sessions/2026-05-15_iter2.md` and [[01 - Progress Report]].

## Open questions / next decision points

### Highest-value next moves
- [ ] **Homberger N=400 result** (in flight) — closes the academic-scale story. **2026-05-15: now bench-able via `cb_scaled_rebench.py`.**
- [ ] **C-class operator gap** — Solomon C is 29% wins (warm tied), Homberger C is 87% wins. Why the inversion at scale? Likely the clustered geography at N=100 falls within PyVRP's local-search neighborhood; at N=200+ the global structure helps the bandit. Worth a focused investigation.
- [ ] **Re-bench v1 leaderboard headline** with `solve_auto()` (2-tier refined recipe) — would update the original report's 95/65/45/72.5% Pareto numbers.
- [ ] **Publish-ready writeup** — the methodology + result are now solid enough for a paper draft.

### Open infrastructure questions
- [ ] **Larger Solomon-Homberger mirrors**: ML4VRP has only 48 of 120 instances; need to find the missing 72.
- [ ] **GART warmstart experiment**: GART estimator is ML-based, like POMO, but produces marginal-length estimates rather than full tours. Could it be used to bias the bandit's *operator choice*, not warmstart? Untested.
- [ ] **Better LLM proposer**: 25 proposals, 0 production wins. Either swap to a larger model (Claude/GPT-5) or pivot the proposer to *instance-generation* (find pathological inputs PyVRP fails on, then operate on those).

### Considered + parked
- LKH-3 as warmstart — tested negative.
- EAS-as-bandit-arm — tested negative.
- LLM committee with multi-model voting — OpenRouter free tier rate-limited too aggressively.
- POMO retraining at larger N — would take days of GPU time; weak evidence of payoff given current POMO checkpoint's structural limitations.

## What's working (don't change without strong reason)

- The 13-arm LinUCB bandit + PyVRP construction composition.
- The paired-seed composability bench at 16+ instances.
- Solomon + Homberger loaders.
- The parallel bench harness with max_workers=4.

## What's not working (revisit)

- The LLM proposer's yield-to-validation ratio (0/25 production-meaningful).
- POMO v3 checkpoint as a competitive baseline at N=200+.
- Logic-axis evaluation under single-judge VLM labelling.
