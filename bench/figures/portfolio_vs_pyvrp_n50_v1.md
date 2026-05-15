# Portfolio@10 vs PyVRP — full v1 N=50, capacity-aware evaluator

Date: 2026-05-13 (after SPEC-0-EVAL-01 capacity fix)

## Headline

| solver        | mean cost | mean wall |
|---------------|----------:|----------:|
| portfolio@10  |   **749.39** |     3.1 s |
| pyvrp@60      |   8 017.87 |    60.0 s |

Portfolio@10 wins on cost by **10×** and on wall by **20×**. *Why
this number is the way it is* matters more than the number.

## Single-instance diagnostic (Manhattan I000)

```
solver        cost   dist    time   veh    wall
portfolio@10  722.2  184.3   400.4   11     5.4 s
pyvrp@30      812.7  136.7   304.6   12    31.1 s
```

On a single instance, the gap is 10 %, not 10×. PyVRP finds a
shorter route in distance and time, but our cost includes
early-wait penalty (wage_per_minute × early_wait_minutes), which
PyVRP doesn't optimise for.

## Why the bench mean inflates 10×

The PyVRP wrapper's docstring (line 9): *"so PyVRP can drop
customers when serving them costs more than the prize."* Prize
collecting is on by default. Under tight TW + 60 s budget, PyVRP
drops ~7 customers per instance on average:

- pyvrp@60 N=50 reported mean **7.38 vehicles** — but v1 N=50
  needs ≥ 11 vehicles to cover all 50 customers within capacity.
- The 3-4 "missing" vehicles correspond to ~7 missing customers
  (PyVRP packs tighter on the routes it keeps).
- Each missed customer costs `hard_late_penalty = 1000`.
- 7 × 1000 + base cost (~1000) ≈ 8000 ≈ observed 8017.

So PyVRP isn't *wrong* — it's playing a different game. With
prize-collecting on, dropping customers is a legitimate move in
its objective. Under our cost model where missed delivery costs
$1000, that's economically dominated by serving them.

## Two takeaways

1. **Portfolio@10 is the legitimate cost leader at v1 N=50.** Not
   by 10× — by ~10 % per-instance.
2. **PyVRP wrapper fixed (2026-05-13).** `required=True` was
   flipped on; no more prize-collecting drops. Smoke at N=50 across
   16 instances (2 per city): mean cost **834.9**, mean miss **0**,
   mean veh **12.38**. Down from 8 017 → 835 = **10× drop**, all of
   which was missed-delivery penalty. This is the version that goes
   into the published bench going forward.

## Real apples-to-apples (post-fix)

| solver        | sample      | mean cost | mean s |
|---------------|-------------|----------:|-------:|
| portfolio@10  | 40 (full v1)|     749.4 |    3.1 |
| pyvrp@30      | 16 (2/city) |     834.9 |   30.0 |

Portfolio@10 wins ~10 % on cost at 10× the wall-clock. Matching-
sample run (pyvrp@30 on the same 40 instances) queued.

## What this does for the headline metric

Reframe of the 3-axis Pareto headline (SPEC-6-PARETO-3AXIS-01):
once the PyVRP wrapper is fixed to serve all customers, the
real cost comparison is portfolio 722 vs pyvrp 812 on Manhattan
— a 10 % cost win in our favour at 5 s vs 30 s. That's a
defensible "we beat PyVRP under our cost model" claim *without*
the misleading 10× gap.

The 3-axis Pareto comparison (cost, time, logic) still works:
- cost: portfolio wins by a small margin
- time: portfolio wins decisively (5 s vs 30-60 s)
- logic: unknown until the OpenRouter committee labels

Even before logic lands, portfolio Pareto-dominates PyVRP on at
least (cost, time) at N=50.
