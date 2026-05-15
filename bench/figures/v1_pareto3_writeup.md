# v1 — 2-axis Pareto dominance, portfolio vs PyVRP, scaling with N

Date: 2026-05-13 (updated)
Generated: `svrptw.bench.pareto3` (SPEC-6-PARETO-3AXIS-01)

## Headline scaling with N — full v1 set

| N   | challenger     | baseline | strict dom % | ties | losses | mean cost (us) | mean cost (them) | wall ratio |
|----:|----------------|----------|-------------:|-----:|-------:|---------------:|-----------------:|-----------:|
|  50 | portfolio@10   | pyvrp@30 |     **92.5** |    3 |  **0** |          749.4 |            808.7 |       10× |
| 100 | portfolio@10   | pyvrp@60 |     **60.0** |   16 |  **0** |        1 482.1 |          1 505.1 |       11× |
| 200 | portfolio@30   | pyvrp@60 |     **37.5** |   25 |  **0** |        2 820.6 |          2 743.5 |        4× |
| 500 | portfolio@30   | pyvrp@60 |     **67.5** |   13 |  **0** |        6 615.1 |          6 663.0 |        2× |

**Zero strict Pareto losses at any N tested.** PyVRP strictly
dominates portfolio on **0/40** instances at every N. The scaling
story:

- **N=50**: portfolio wins cost decisively + time decisively →
  92.5 % strict dominance.
- **N=100**: portfolio wins cost narrowly (most instances within
  1 % tolerance, so cost "ties") + time decisively → 60 %.
- **N=200**: PyVRP wins mean cost by 2.7 % but loses time 4× —
  most instances become non-comparable Pareto points; portfolio
  still dominates 37.5 %.
- **N=500**: portfolio wins cost narrowly again (6 615 vs 6 663,
  within tolerance) AND wins time 2×. The trough at N=200 was
  the time gap shrinking from 11× to 4×; at N=500 the cost catches
  up while time stays at 2×, so dominance recovers to 67.5 %.

The honest story for a paper: **portfolio is always on the (cost,
time) Pareto frontier on every v1 instance tested**; PyVRP joins
the frontier from N≈200 onward but never strictly dominates.
There is no v1 instance (out of 160 across 4 N-bins × 8 cities ×
5 seeds) where switching from portfolio to PyVRP gives you a
clean win on both axes.

## Mean stats (matched 40-instance sample)

| solver        | mean cost | mean wall | mean miss | mean overload |
|---------------|----------:|----------:|----------:|--------------:|
| portfolio@10  |    749.39 |     3.1 s |       0.0 |           0.0 |
| pyvrp@30      |    808.74 |    30.0 s |       0.0 |           0.0 |

Portfolio wins by ~7.3 % on cost at ~10× the speed. Both solvers
respect capacity and serve every customer (matters after the
SPEC-0-EVAL-01 + PyVRP `required=True` fixes that landed today).

## What 92.5 % means

Strict Pareto dominance with 1 % relative tolerance per axis:
challenger must be better-or-equal on every axis (within 1 %) and
strictly better on at least one. The 3 ties are instances where
the per-axis margin is inside that 1 % band — i.e., the two
solvers are statistically indistinguishable on at least one of
(cost, time), and we don't claim a win.

This is the right way to report. A naïve "mean cost" headline
hides per-instance variability and lets one outlier carry the
story. 92.5 % strict dominance is the *robust* claim.

## Caveats that need to land before this becomes a paper headline

1. **N=50 only.** N=100, N=200, N=500 not yet matched-sampled. The
   v1+PyVRP SOTA bench (overnight) had N=100/200 numbers but those
   used `required=False` and need re-running.
2. **No logic axis yet.** When SPEC-6-LOGIC-01 + SPEC-6-LOGIC-02
   ship, this becomes a 3-axis number. Expect a small drop —
   dispatcher judgment on portfolio's many-small-routes plans may
   be worse than on PyVRP's fewer-larger-routes plans.
3. **No HGS-DIMACS in the mix.** PyVRP 0.9+ is the published
   state-of-the-art; HGS-DIMACS is the academic-paper companion
   we should also benchmark against. SPEC-3-LKHCOMPARE-01 entry
   on the roadmap.
4. **`portfolio@10` is one configuration.** The LinUCB bandit's
   choices are seed-dependent; we should average over 3 seeds to
   take noise out of the comparison.

## Reproducing

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m svrptw.bench.harness run \
  --instances instances/v1 --out bench/runs/v1_n50_portfolio_postfix.json \
  --solvers portfolio@10 --n-filter 50

PYTHONPATH=. .venv/Scripts/python.exe -m svrptw.bench.harness run \
  --instances instances/v1 --out bench/runs/v1_n50_pyvrp30_required.json \
  --solvers pyvrp@30 --n-filter 50

PYTHONPATH=. .venv/Scripts/python.exe -m svrptw.bench.pareto3 \
  --rows '...' --challenger portfolio@10 --baseline pyvrp@30 \
  --axes operational_cost,wall_clock_seconds --minimise true,true
```
