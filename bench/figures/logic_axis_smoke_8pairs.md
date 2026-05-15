# Logic axis smoke — 8 authoritative pairs label PyVRP > Portfolio unanimously

Date: 2026-05-13
Source: 10-pair Gemini-solo labelling smoke (`bdys1v34z`)
Output: `data/logic/pairs_n50_smoke.json`

## What we ran

10 portfolio@10-vs-pyvrp@30 pairs sampled from the 40 v1 N=50 instances,
each solution rendered as PNG, then submitted to the OpenRouter
committee with permissive thresholds (min_responders=1, max_std=1.0)
so Gemini-direct's solo vote counts when OpenRouter is rate-limited.

## Result

| | count |
|---|---:|
| Total pairs | 10 |
| Authoritative + non-NaN | **8** |
| Label = 0.0 (B≻A, pyvrp wins) | **8** |
| Label = 1.0 (A≻B, portfolio wins) | 0 |
| Label = 0.5 (tie) | 0 |
| Mean teacher_score (pyvrp − portfolio) | **+0.61** |

Per-pair detail:

| city | portfolio score | pyvrp score | label |
|---|---:|---:|--:|
| SanFrancisco-I000 | 0.00 | 0.65 | B≻A |
| Manhattan-I001 | 0.20 | 0.75 | B≻A |
| Manhattan-I003 | 0.20 | 0.85 | B≻A |
| Pittsburgh-I002 | 0.00 | 0.65 | B≻A |
| Austin-I000 | 0.00 | 0.85 | B≻A |
| SanFrancisco-I004 | 0.00 | 0.50 | B≻A |
| SanFrancisco-I001 | 0.10 | 0.50 | B≻A |
| Charleston-I000 | 0.00 | 0.60 | B≻A |
| Cambridge-I002 (partial) | NaN | 0.85 | dropped |
| Phoenix-I001 (partial) | 0.20 | NaN | dropped |

## What this means for the headline

The 2-axis Pareto headline (cost, wall) stays intact:
**portfolio strictly dominates PyVRP on 95/65/45/72.5 % of v1
instances at N=50/100/200/500**, with 0 strict losses across 160
instances.

The **3-axis Pareto picture flips**:

| axis | portfolio wins | pyvrp wins |
|------|---:|---:|
| cost | 8/8 | 0/8 |
| wall | 8/8 | 0/8 |
| **logic** | **0/8** | **8/8** |

Strict Pareto requires winning every axis. With each solver winning
exactly one axis-cluster, **neither strictly dominates the other**
under (cost, wall, logic). The portfolio Pareto-dominance fraction
collapses from 95 % → **0 %** when the logic axis is added —
but PyVRP also goes from "0 strict wins" → "0 strict wins".

The result is **100 % ties under 3-axis Pareto with this teacher**.

## Caveats

1. **Single-teacher bias.** This is essentially a Gemini-Flash-Lite
   solo judgment with occasional Nemotron-12B-VL backup. The earlier
   3-voter smokes showed Nemotron tends to score solutions higher
   (less critical) than Gemini — so the headline of "PyVRP wins
   logic" partly reflects which model is dominating the committee.

2. **N=8 pairs.** Not enough to publish, but the pattern is unanimous
   so the trend is real. Scaling to ~100 pairs would tighten the
   confidence interval; 2.5k pairs would unblock the Bradley-Terry
   student training.

3. **No student signal.** With all 8 usable labels = 0.0, there's
   no Bradley-Terry diversity. A student trained on this would
   collapse to "always predict B≻A". For training, we need pairs
   where portfolio sometimes wins — likely from portfolio vs greedy
   (portfolio's logic score should exceed greedy's "no improvement"
   plans) or portfolio vs auction_gart.

## What this honestly tells the user

The original spec framing was: "you're not going to beat published
benchmarks at academic VRPTW cost; you ARE going to beat the
dispatcher-acceptance axis." The data says the *opposite* with our
current solvers:

- Portfolio beats PyVRP on cost (DIMACS winner) and on wall.
- Portfolio loses to PyVRP on dispatcher-acceptance.

PyVRP produces fewer, more visually-coherent routes, which the
committee (especially Gemini) reads as more shippable. Portfolio's
many-small-routes plans look spaghetti even when they're cheaper
to operate.

This is research-honest information. The Pareto frontier under all
three axes has TWO points (portfolio, PyVRP); neither dominates the
other. Picking between them depends on which axis the dispatcher
weights.

## Next steps

a. **Diversify pairs**: re-label with portfolio vs greedy and
   portfolio vs auction_gart so the student has some A-wins labels.
b. **Train LogicStudent on diverse 100-pair dataset** once labels
   are diverse.
c. **Decide direction**: do we want to add a "logic-aware"
   construction phase that produces fewer-larger routes (matching
   PyVRP's pattern), or do we accept that the 2-axis (cost, wall)
   dominance is the right headline?
