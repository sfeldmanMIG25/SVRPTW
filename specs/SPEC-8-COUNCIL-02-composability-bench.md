# SPEC-8-COUNCIL-02 — Council bench: arm-pool composability test

**Status:** ACCEPTED (2026-05-13, user sign-off)
**Amends:** SPEC-8-COUNCIL-01 (Loop 1 shadow bench)

## Why

SPEC-8-COUNCIL-01's shadow bench tests "does this operator find improvement
on a portfolio@10s output?". Empirically (this session, 4 seeds: tw_anchor,
cap_rebalance, cross_route_2opt, gart_guided_ruin) **no single-shot operator
beats portfolio@10s** because PyVRP at production budget runs hundreds of
LNS iterations and converges to a local minimum that no one-shot heuristic
can escape. The bar is asking for the impossible.

The right council question is **complementarity**, not replacement:
> Does adding this operator as an extra arm to the production bandit
> improve the bandit's final cost at the same wall-budget?

A genuinely useful operator may be inferior on average yet improve the
ensemble by covering a region the existing 13 arms miss.

## What changes

`svrptw/council/shadow_bench.py` gains a `composability_run()` function
alongside the existing `run_shadow()`:

```python
def composability_run(
    proposal_id: str,
    operator: Callable,                # the candidate, (sol, ctx) -> Sol|None
    *,
    instances: tuple[str, ...] = COMPOSABILITY_INSTANCES,   # 6 fixed N=50 + 2 N=100
    budget_seconds: float = 10.0,      # same wall-budget for both runs
    n_repeats: int = 3,                # seeds 0,1,2 to denoise
    min_mean_abs_delta: float = 2.0,   # tighter than single-shot ($5)
    min_hit_rate: float = 0.40,
) -> dict
```

Per-instance protocol:
1. Run portfolio (13-arm bandit, no candidate) at `budget_seconds`, seeds 0..n_repeats-1. Record mean cost = `baseline_cost`.
2. Run portfolio with the candidate registered as a 14th arm at `budget_seconds`, same seeds. Record mean cost = `augmented_cost`.
3. Per-instance `delta = baseline_cost - augmented_cost` (positive = candidate helped).

Accept rule:
- `mean_abs_delta >= min_mean_abs_delta` across all instances
- `hit_rate >= min_hit_rate` (fraction of instances with delta > $1)
- No instance regressed by more than 1% (variance guard)
- No overload / infeasibility introduced

Wiring the candidate as a 14th arm requires `svrptw/solvers/classical/portfolio.py`
to accept an `extra_arms: list[Callable]` kwarg that prepends/appends to
its native arm pool. The bandit's LinUCB selection then naturally explores
the new arm without code change.

## Outputs

`bench/runs/council/composability/<proposal_id>.json` with per-instance
{baseline_costs[3], augmented_costs[3], mean_baseline, mean_augmented,
delta, regression}. `accept` boolean and `reasoning` summary.

## Backward compatibility

`run_shadow()` remains. Council orchestrator can run either or both. The
existing 4 seeds will be re-evaluated under composability; results recorded
to the sqlite corpus alongside their single-shot results so the LLM
proposer (downstream) sees both signals during reflection.

## Acceptance gates

- The composability test correctly distinguishes complementary operators
  from PyVRP-redundant operators on the existing 4 seeds.
- Runtime: ~`n_repeats × 2 × n_instances × budget_seconds` per proposal,
  capped at ~8 min for 8 instances × 3 seeds × 10s × 2 = 480s.
- All existing council unit tests stay green.
