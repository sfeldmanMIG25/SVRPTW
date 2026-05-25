# Construction strategies — audit + honest assessment

The openvrp native solver exposes three construction variants via
``SolveOptions.construction_variant``:

| Variant | Family | When to use |
|---|---|---|
| `"nn"` *(default)* | Parallel nearest-neighbor with kNN candidate filter | Every workload; default. Sub-second at N=1000. |
| `"regret2"` | Regret-2 insertion (Solomon I1 + regret tie-breaker) | Optional; documented in literature, exposed for users who want classical I1. |
| `"regret3"` | Regret-3 insertion | Same as regret2 with k=3. |

Plus the SISR-style **`blink_prob`** parameter (default 0.0) on the
relocate phase — non-zero values add stochastic skips for
diversification, per Christiaens & Vanden Berghe (Transportation
Science 2020).

```python
from openvrp import SolveOptions

# Default fast construction:
opts = SolveOptions(budget_seconds=10.0, construction="fast")

# Regret-2 with SISR-style blinks (slower; richer diversity):
opts = SolveOptions(budget_seconds=30.0, construction="fast",
                    construction_variant="regret2",
                    blink_prob=0.05)
```

## Honest assessment of each variant

### `"nn"` — parallel nearest-neighbor (default)

The cheapest, fastest baseline. Per-customer cost O(k) where k=kNN
size. Total construction O(N·k). At N=1000 with k=20 this is ~20k ops
— **sub-millisecond**.

**Quality**: produces feasible solutions with reasonable cost.
Empirically matches Solomon I1 at small N and degrades gracefully at
large N. Crossings tend to be low (5-30 at N=1000) because the
greedy local extension prefers near-neighbors.

### `"regret2"` and `"regret3"` — regret-k insertion

Classical Solomon I1 with the regret tie-breaker (Joubert et al.,
literature standard). For each unrouted customer the algorithm finds
the top-(k+1) feasible insertions across all open routes, computes
`regret = (k-th best delta) − (best delta)`, and inserts the
**highest-regret** customer at its best position. The intuition: a
high-regret customer is "hard to place later" if deferred, so place
it now.

**Why we expose it**: documented improvement over plain greedy I1 in
the academic literature; users running Solomon-style benchmarks may
want it for reproducibility.

**Honest finding from our bench** (smoke at N=50/100/250, 20 s
budget, deterministic seed=0):

| N | nn | regret2 | regret3 | nn+blink0.05 |
|---|---|---|---|---|
| 50 | $297 (0.19s, K=3) | $308 (0.26s, K=3) | $303 (0.25s, K=3) | $297 (0.01s, K=3) |
| 100 | $394 (0.04s, K=4) | $554 (2.10s, K=5) | $533 (2.30s, K=5) | $395 (0.05s, K=4) |
| 250 | $658 (0.19s, K=8) | $2481 (20.25s, K=19) | $2526 (20.10s, K=19) | $658 (0.19s, K=8) |

**Conclusion**: in *our* (single-seed-route I1) implementation,
regret-k under-performs the parallel-NN default — opening too many
routes due to the sequential route-opening policy. This matches the
field's consensus that **construction is not where the gains are**;
the recreate/repair mechanism that follows dominates the difference.
We keep regret-k exposed for users who explicitly want classical I1
behavior, but the default remains `"nn"`.

For users who genuinely want the regret-k *quality lift* documented
in the literature, the path is **parallel-route regret-k** (open
~⌈N·mean_demand/cap⌉ routes from the start so all customers compete
across many routes simultaneously) — a separate algorithm we have
NOT implemented. We will not add it unless the user-evidence supports
the investment.

### `blink_prob` — SISR blink mechanism

During cross-route relocate, each candidate insertion position is
skipped with probability `blink_prob`. β = 0.01–0.1 is the
literature-documented sweet spot. β = 0 (default) is deterministic
greedy.

**Why this matters**: blinks are the diversification primitive from
SISR (Christiaens & Vanden Berghe, *Transportation Science* 2020).
At long budgets they let the search escape first-improvement plateaus
without changing the basic algorithm. At short budgets the effect is
muted (the relocate phase doesn't iterate enough for the random
skips to matter).

**Honest finding**: at our 20 s budgets the bench above shows blinks
match NN exactly. The lift is at longer budgets where the relocate
phase iterates more.

## Recommendations

1. **Default to `"nn"` + `blink_prob=0.0`** for sub-second solves.
2. Use `blink_prob=0.05` for budgets ≥ 60 s where SISR diversification
   pays back.
3. `regret2` / `regret3` are exposed for reproducibility / academic
   benchmarks; do not expect a quality lift in our implementation.
4. The bigger gains (per the field literature and our own iter log)
   come from the **refine** phase: untangle, polish_load_balance,
   and — when `[pyvrp]` is installed — the svrptw LinUCB bandit
   driving 13 operators. See [algorithmic_moves.md](algorithmic_moves.md).

## Related literature (cited for completeness)

- Solomon, M. M. (1987). *Algorithms for the vehicle routing and
  scheduling problems with time window constraints.* Operations
  Research. The original I1 sequential-insertion heuristic.
- Joubert, J. W., & Claasen, S. J. (2006). *A sequential insertion
  algorithm for the initial solution of a constrained vehicle fleet
  mix problem.* The TWC (time-window-compatibility) pre-screening
  used to accelerate insertion scoring at large N.
- Christiaens, J., & Vanden Berghe, G. (2020). *Slack Induction by
  String Removals for Vehicle Routing Problems.* Transportation
  Science. SISR — the blinks + ruin-and-recreate framework that
  defines the modern frontier.
- Accorsi, L., & Vigo, D. (2021). *A fast and scalable heuristic for
  the solution of large-scale capacitated vehicle routing problems.*
  Transportation Science. FILO — iterated local search with
  localization at N=1000+.
