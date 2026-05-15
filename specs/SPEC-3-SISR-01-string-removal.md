# SPEC-3-SISR-01 — SISR string-removal operator

```
ID:            SPEC-3-SISR-01
Title:         Slack Induction by String Removals (Christiaens & Vanden Berghe 2020)
Owner role:    OR Engineer
Status:        FROZEN
Inputs:        Instance, Solution, Settings, max_seconds, avg_strings, max_string_len, blink_p
Outputs:       svrptw.solvers.common.local_search.sisr_destroy_repair
```

## Why

Track-2 research distilled SISR as "the highest single-operator ROI"
addition for asymmetric tight-TW instances. ~50 lines. Often dominates
relocate + 2-opt and is a near-drop-in destroy-repair move that fits
inside our existing operator pool.

## Behaviour

A SISR move:
1. **Pick a seed customer** uniformly at random from the routed set.
2. **Remove a "string"** of adjacent customers (in route order) containing
   the seed; string length is uniform in `[1, max_string_len]`.
3. **Spread to adjacent routes.** Determine `n_strings = ceil(avg_strings)`
   (default 10). For each, pick the route whose customers are closest in
   *time-aware* distance to the seed's coordinates; remove a string from
   it too.
4. **Reinsert** all removed customers via *blink-greedy*: for each removed
   customer (in random order), evaluate insertion cost at every position
   in every route; with probability `blink_p` (default 0.01), skip the
   current best position. Pick the best non-blinked position. Reject if
   no TW-feasible position exists; route the customer's drop or carry it
   to the next iteration.
5. **Accept** the new solution iff total operational cost strictly drops.

Default hyperparameters (Christiaens 2020, refined 2023-24):
- `avg_strings = 10`
- `max_string_len = 10`
- `blink_p = 0.01`

## Acceptance

- On N=100 instances/v1 (when ready), SISR after our current improvement
  chain finds ≥ 1 % additional cost reduction on at least 40 % of
  instances at a 3-second budget.
- On a synthetic N=20 instance with 4 routes and a known
  pessimum-but-locally-optimal layout, SISR escapes it in < 100 iterations.
- Preserves TW feasibility on all touched routes (regression assertion).
- Worst-case time per attempt: O(strings · N · K). For N=100, K=20,
  strings=10 → ~20k ops per attempt; 3-s budget allows ~1500 attempts.

## Non-goals

- Adaptive ruin / blink parameters. Static hyperparameters; ALNS-style
  adaptation is SPEC-3-OPSEL-01's job.
- Population search. Single-incumbent.

## References

- Christiaens & Vanden Berghe, "Slack Induction by String Removals for VRP," Transportation Science 2020.
- Wouda, Lan, Kool: PyVRP 0.9 SISR variants (2023-24).
