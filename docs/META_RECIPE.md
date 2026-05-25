# Meta-recipe: how to add a new operational constraint to svrptw

This is the procedure that built the entire 17-cost-term constraint catalog
in svrptw 0.1.0. Following it produces a constraint whose bandit-trained
solver reliably finds the right trade-offs, with a paired-bench-validated
shipping criterion.

## The two pre-flight checks

### Step 0 — operator-coverage check

> Verify the bandit's operator set acts on the axis the proposed cost term
> targets. If yes → proceed. If no → either add the missing operator first
> (canonical pattern below) or pick a different term whose axis the existing
> operators DO cover.

| axis | operators that act on it |
|------|--------------------------|
| customer assignment to routes | `relocate`, `swap`, `or_opt`, `swap_star`, `cyclic_3` |
| route count | `split_route`, `merge_routes`, `drop_route`, `vehicle_kill` |
| customer ordering within route | `two_opt_intra`, `three_opt_intra`, `or_opt` |
| cross-route load distribution | `relocate`, `two_opt_star`, `merge_routes` |
| **per-route start time** | `shift_start` (added iter-5w-bis) |
| **per-route vehicle class** | `class_shift` (added iter-6a-7-bis) |
| **per-route depot assignment** | `depot_shift` (added iter-6a-5-bis) |

If your constraint's axis isn't in the table, apply the canonical
operator-pair fix (last section).

### Step 0.5 — coefficient calibration

> Calibrate the coefficient so the baseline penalty is a meaningful fraction
> of operational cost (~3–10%). Too small → the bandit treats the term as
> noise and produces identical solutions to baseline. Smoke-evaluate one
> `solve_auto` baseline run with the proposed coef BEFORE firing the paired
> bench.

Procedure:
1. Run `solve_auto(inst, default_Settings, budget_seconds=N*0.15)` on one
   representative v1_large instance.
2. Apply the new term to the resulting solution at coef=1.0 via `evaluate()`
   under settings with the term enabled; this gives the "unit penalty"
   magnitude.
3. Choose a coef such that `coef * unit_penalty / ops_cost ≈ 0.05`.

This rule was added iter-5y after a coef=0.001 fairness bench produced
3/6 wins at noise-floor mean ±$0, then the same code at coef=0.5 produced
5/6 wins at mean +$36/inst.

## The 7 implementation gates

1. **Schema field(s)** in `svrptw.config.schema.Economics`, default 0 /
   empty / 1.0 (bit-identical when off).
2. **Gated evaluator block** in `svrptw.solvers.common.solution.evaluate()`,
   conditional on the schema field(s) being non-default.
3. **2 unit tests** in `tests/unit/test_cost_terms_smoke.py`:
   - Linearity in the coefficient (doubling coef doubles cost-delta).
   - Gating: setting only part of the required fields leaves cost identical
     to baseline.
4. **Runtime smoke**: a single-instance check that the cost term computes a
   non-trivial delta at `evaluate()` time on real data.
5. **Paired 6-instance bench** at v1_large (Manhattan/Paris/SF × N=500/1000).
   Use `bench/scripts/iter6a*_*_v1large.py` as templates.
6. **6/6 win threshold** under the term-aware objective.
7. **Ship as a production option** OR queue for retuning if 6/6 isn't hit.

## Single-term coefficient calibration (Step 0.5 worked examples)

| term | coef chosen | baseline_penalty / ops_cost |
|------|-------------|------------------------------|
| `shift_overrun_penalty_per_min` | $1.0/min | 4-8% |
| `driver_time_variance_penalty_coef` | 0.5 | 5-10% |
| `break_violation_penalty_per_min` | $2.0/min @ 90min cap | 12-18% (forced tight) |
| `embargo_violation_penalty_per_visit` | $50/visit @ 30min windows | 30-50% (heavy on purpose) |
| `vehicle_class_fixed_premiums` | (5, 20) | 5-15% per class |
| `range_violation_penalty_per_mile` | $1/mi @ 5mi | 30-40% (forced tight) |
| `pd_violation_penalty_per_pair` | $50/pair | 5-15% |
| `under_min_routes_penalty_per_route` | $100/missing | 50-100% (forced floor) |
| `skill_mismatch_penalty_per_visit` | $100/mismatch | scales with N affected |

## The canonical Step 0 remediation pattern

When a constraint's axis is **auto-decided by the evaluator** (not directly
controllable via the bandit's customer-reorder operators), Step 0 fails. The
canonical fix is always the same pattern:

1. **Add an explicit per-route field** in `Route` (default = sentinel
   meaning "auto").
2. **Make the evaluator honor the field** when set, fall back to auto
   otherwise.
3. **Add a dedicated operator** that sweeps candidate values for that
   field (greedy local search across {auto, val_1, val_2, ...}).
4. **Register the operator** in `svrptw.solvers.classical.portfolio._OPS`
   dict.

### Three worked examples from svrptw 0.1.0

| operator | route field | candidates swept | unblocks | bench verdict |
|----------|-------------|------------------|----------|---------------|
| `shift_start` | `start_offset_minutes` | {0, 30, 60, 90, 120} min | iter-5x peak_hour | partial (geometry-bound) |
| `class_shift` | `vehicle_class_idx` | {-1, 0..n_classes-1} | iter-6a-7 skills | 5/6 +$2,471/inst |
| `depot_shift` | `depot_idx` | {0..n_depots-1} | iter-6a-5 multi_depot | smoke −99.5% cost |

### When the remediation works vs not

The pattern **works cleanly** when:
- The decision axis has a small discrete candidate set (≤ 5 values).
- Each candidate evaluates in O(K) or better (sweep over routes only).
- The constraint geometry permits ≥ 1 candidate that actually helps
  (i.e., the answer isn't "no candidate satisfies any constraint").

The pattern is **geometry-bound** when:
- Constraint windows are wide AND aligned with depot.ready
  (iter-5x peak_hour: 2-hour peak starting at depot.ready=480 leaves
  no candidate that escapes peak without losing entire customer set).
- Multiple constraints push in opposing directions on the same axis.

### Empirical signature of geometry-bound vs budget-starved

Two failure modes look similar at first ("constraint doesn't reduce") but
have different fixes. The diagnostic is to **re-run at 2x budget**:

| symptom at 1x budget | symptom at 2x budget | diagnosis | fix |
|---------------------|---------------------|-----------|-----|
| 2-3/6 wins, high variance, mean ≈ 0 | 5-6/6 wins, narrower variance, mean clearly positive | **budget-starved** | recommend ≥ N × 0.3 s budget |
| 1-2/6 wins, low variance, mean ≈ 0 | 1-2/6 wins, low variance, mean STILL ≈ 0, per-instance penalties unchanged | **geometry-bound** | change constraint geometry (narrower windows / different alignment) or accept the bound |

Worked examples from this session:
- **Combined-stack-16 at N=500** (2026-05-16): 75s 2/6 mean −$45 ± $465 → 150s 3/3 wins mean +$482 ± $358 → **budget-starved**, fixed by recommending N×0.3s.
- **Single-term peak_hour at v1_large** (iter-5x / -bis / -ter): 75s 2/6, 75s+shift_start 2/6, 150s+shift_start 1/6 — peak surcharges identical baseline-vs-peaked across all 3 budgets → **geometry-bound**, fixed only by changing window geometry (cf. iter-6a-2 embargo 30-min windows 6/6 +$1796).

## Per-N budget rule of thumb

Multi-seed stability bench (iter-6a-perf-2, iter-6a-2026-05-16-multi-seed)
showed budget-starvation at full constraint stacks:

| N | recommended budget for full-stack solve |
|---|-----------------------------------------|
| 500 | ≥ 150 s |
| 1000 | ≥ 300 s |

Rule of thumb: **budget ≥ N × 0.3 seconds** for full 16-term stack solves.
Single-term solves can use the standard `solve_auto` budget (`N × 0.15s`)
since they're lower-dimensional.

## Honest-empiricism: when to falsify your own win

Throughout this session, several apparent wins were re-tested at scale and
falsified:

- **iter-5q**: fcv2-warm "won" cost on 1 instance → tested at 6 instances,
  pyvrp-warm strictly dominated → direction closed.
- **iter-5s**: Phase F arm E "won" on N=50 → tested at N=500, gained $0
  quality at +$54/inst cost → in-loop reward shaping closed.
- **iter-5t**: fcv2 "won" quality leaderboard → K-fairness audit showed
  it used 1.7–3.3× more vehicles → metric artifact, not real lead.
- **iter-6a-1 @ regulation cap**: driver_breaks "won" 0/6 inert → tested
  at tight cap, 6/6 +$179/inst → narrative clarified.
- **2026-05-16 multi-seed stack**: combined-stack +$281 single-seed →
  multi-seed at 75s showed mean ±$0 ± $465, multi-seed at 150s showed
  mean +$482 ± $358 → budget rule documented.
- **2026-05-16 architectural drift on embargo**: iter-6a-2 bench reported
  +$1,612/inst on Manhattan-N500 seed=0. Re-run on same instance + same
  seed today under current code (with crossings cache + Phase F lite +
  shift_start + class_shift + depot_shift operators all added since)
  shows +$756/inst. The bandit's arm pool changed → exploration changed
  → magnitudes changed. The direction (positive win) replicates; the
  magnitude doesn't. Catalog numbers documented as "at time of landing,"
  not "current."

Each cycle made the package stronger by bounding its claims to where they
actually hold.

## Three documented failure modes (with diagnostic protocols)

1. **Generalization failure** — a win on one instance / small N doesn't
   hold at more instances or larger N. *Diagnostic*: bench at 6 paired
   v1_large instances spanning Manhattan/Paris/SF × N=500/1000.
   *Examples*: iter-5q fcv2 single-instance, iter-5s Phase F arm E at
   N=500, iter-5t fcv2 K-fairness artifact.

2. **Seed-variance failure** — a win at one seed doesn't hold across
   seeds (mean ± std contains zero). *Diagnostic*: 3+ seeds on one
   instance via `iter6a_stack16_multi_seed.py` or
   `iter6a2_embargo_multi_seed.py`. *Example*: combined-stack +$281
   single-seed → null at 75s, recovered at 150s budget.

3. **Architectural-drift failure** — a win recorded when the bandit's
   arm pool had K operators doesn't replicate when K' ≠ K operators are
   in the pool. *Diagnostic*: re-run the bench under current arch.
   *Fix*: conditional registration of meta-recipe operators based on
   `Settings.economics` inspection at solve time (so e.g. `shift_start`
   is only in the pool when a time-segment cost term is active). Lives
   in `svrptw.solvers.classical.portfolio._filter_ops_pool` with 12
   guard tests in `tests/unit/test_conditional_op_registration.py`.
   *Examples*:
   - Single-term embargo: at-landing 6/6 +$1612 → drifted to 5/6 +$798
     unconditional pool → **fix → 6/6 +$1648** (92% of original).
   - Combined 16-term stack at N=500 b=150s 3-seed: pre-fix mean +$482
     ± $358 @ wall_x 1.24x → **post-fix mean +$816 ± $399 @ wall_x
     0.90x** (+69% mean savings, 27% faster than baseline because the
     pool is now axis-matched and the bandit gets more iterations on
     operators that have work).

All three are caught by **honest re-bench at higher rigor than initial
landing**. The discipline that built this catalog is to never trust a
landing-time number forever; treat it as a snapshot under that turn's
code state.

## Quick reference: where everything lives

- Cost terms: `svrptw/config/schema.py::Economics`
- Evaluator: `svrptw/solvers/common/solution.py::evaluate`
- Operators: `svrptw/solvers/common/{shift_start,class_shift,depot_shift}.py`
- Bandit pool: `svrptw/solvers/classical/portfolio.py::_OPS`
- Tests: `tests/unit/test_cost_terms_smoke.py` (30 tests)
- Single-term benches: `bench/scripts/iter6a*_*_v1large.py`
- Full-stack benches: `bench/scripts/iter6a_full_stack16_solve.py`
- Multi-seed: `bench/scripts/iter6a_stack16_multi_seed.py`
- Reproducer: `bench/scripts/headline_results.py`
- Constraint catalog docs: `svrptw/constraints.py`
- This file: `docs/META_RECIPE.md`
