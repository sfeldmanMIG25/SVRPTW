# Headline — portfolio Pareto-dominance over PyVRP on v1 (13-arm bandit)

Date: 2026-05-13 (post-drop_leg landing)
Module: `svrptw.bench.pareto3.dominance_report`

## Final result with the 13-arm bandit (3 SPEC-7-OPS-DESTROY-01 ops)

| N | dom %       | n_dom | ties | pyvrp dom | portfolio cost | portfolio wall | pyvrp cost | pyvrp wall |
|--:|------------:|------:|-----:|----------:|---------------:|---------------:|-----------:|-----------:|
|  50 | **95.0 %** | 38/40 |   2 |    0/40   |        748.71  |       4.27 s   |    808.74  |    30.0 s |
| 100 | **65.0 %** | 26/40 |  14 |    0/40   |      1 476.61  |       5.93 s   |  1 505.11  |    60.0 s |
| 200 | **45.0 %** | 18/40 |  22 |    0/40   |      2 804.29  |       9.48 s   |  2 743.46  |    60.1 s |
| 500 | **72.5 %** | 29/40 |  11 |    0/40   |      6 569.11  |      25.65 s   |  6 662.96  |    60.6 s |

**Across 160 v1 instances, PyVRP strictly Pareto-dominates portfolio
on 0. Portfolio strictly dominates PyVRP on 111 (69 %).**

## Cumulative gain from destroy-ops family

Three operators added on 2026-05-13: `drop_route` (whole route by
worst utilisation), `destroy_island` (geographic cluster via Voronoi/
k-NN), `drop_leg` (single worst-load-util edge).

| N | dom (10-arm) | dom (12-arm) | dom (13-arm) | total Δ |
|--:|-----:|-----:|-----:|-----:|
|  50 | 92.5 % | 95.0 % |  95.0 % | +2.5 pp |
| 100 | 60.0 % | 65.0 % |  65.0 % | +5.0 pp |
| 200 | 37.5 % | 42.5 % |  **45.0 %** | **+7.5 pp** |
| 500 | 67.5 % | 70.0 % |  72.5 % | +5.0 pp |

Mean cumulative +5.0 pp across N-bins. 0 strict-Pareto losses to
PyVRP at any N, in any configuration.

## Per-operator contribution (destroy_island vs drop_leg specifically)

- **drop_route + destroy_island (12-arm vs 10-arm)**: +3.75 pp mean
  across N. Most useful at N=100/200 where there's underutil to find.
- **drop_leg (13-arm vs 12-arm)**: +1.25 pp mean across N. Helps
  only at N=200 (+2.5) and N=500 (+2.5); neutral at N=50/100.
  Logical: leg-attack only pays when routes are long enough to have
  many possible legs.

## Why portfolio never loses Pareto

PyVRP strictly Pareto-dominates portfolio on 0 of 160 instances
because portfolio is *always* faster on wall:
- N=50: 4.3 s vs 30 s (7×)
- N=100: 5.9 s vs 60 s (10×)
- N=200: 9.5 s vs 60 s (6×)
- N=500: 25.7 s vs 60 s (2.3×)

Even when PyVRP wins on cost (N=200), the wall gap exceeds the
1 % tolerance, preventing strict dominance. The ties at every N
are the instances where one solver wins on cost outside tolerance
but loses on wall outside tolerance.

## Methodology notes

- Capacity-aware evaluator (SPEC-0-EVAL-01)
- PyVRP wrapper: `required=True` (no customer dropping)
- Strict Pareto with 1 % relative tolerance per axis
- 40 instances per N-bin (8 cities × 5 seeds)
- Portfolio: 13-arm LinUCB bandit, time budgets {10 s, 10 s, 30 s, 30 s}
  for N ∈ {50, 100, 200, 500}
- PyVRP budgets matched: {30 s, 60 s, 60 s, 60 s}
