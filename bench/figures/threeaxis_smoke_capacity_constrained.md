# 3-axis Pareto smoke — blocked by OpenRouter free-tier rate limits

Date: 2026-05-13

## What ran

5 v1 Manhattan N=50 instances × {portfolio@10, pyvrp@30} = 10
(solver, instance) cells. Each solution rendered and submitted to
the SPEC-6-LOGIC-02 committee (Gemma-4 / Nemotron-30B / Nemotron-12B-VL
/ Gemini-Flash-Lite). Pareto3 then run with the logic axis included.

## The data

| instance     | solver       | cost | wall | logic | std  | auth  | nresp |
|--------------|--------------|-----:|-----:|------:|-----:|------:|------:|
| I000 | portfolio@10 |  724 |  6.4 |  **0.0** | 0.00 |  **T** |   **3** |
| I000 | pyvrp@30     |  813 | 30.1 |  0.5 | 0.15 | F |   2 |
| I001 | portfolio@10 |  731 |  4.2 |  0.0 | 0.38 | F |   2 |
| I001 | pyvrp@30     |  875 | 30.0 |  0.0 | 0.25 | F |   2 |
| I002 | portfolio@10 |  622 |  4.2 |  0.5 | 0.00 | F |   1 |
| I002 | pyvrp@30     |  630 | 30.0 |  —   | 0.00 | F |   0 |
| I003 | portfolio@10 |  698 |  4.3 |  0.0 | 0.00 | F |   1 |
| I003 | pyvrp@30     |  801 | 30.0 |  —   | 0.00 | F |   0 |
| I004 | portfolio@10 |  601 |  4.1 |  0.0 | 0.00 | F |   1 |
| I004 | pyvrp@30     |  655 | 30.0 |  —   | 0.00 | F |   0 |

## Why most rows aren't authoritative

OpenRouter's free-tier rate limits hit harder than the SPEC-6-LOGIC-02
caps suggested. We have 4 free-vision-capable models + Gemini-direct,
and only 1 of 10 calls got 3+ responders. Most got 1-2, and several
got 0 (every model rate-limited at once).

Pareto3 correctly drops the logic axis on every instance where either
solver's score is non-authoritative → falls back to (cost, time):

```
2-axis result:    portfolio dom 5/5 (100 %)
3-axis result:    logic axis dropped on 5/5; same 5/5 fallback
```

## The one authoritative result (I000 portfolio@10)

- 3 voters, all agreed on **0.0** (std = 0.00)
- "Reject; structural problem" per the rubric
- Same pattern as earlier solo-Gemini results: PyVRP's fewer-larger
  routes score higher than portfolio's many-small-routes

If this pattern held across 40 instances, the 3-axis headline would
change from "portfolio strictly dominates" to "portfolio + pyvrp are
Pareto-incomparable on the 3-axis frontier". But N=1 isn't enough to
publish.

## Options to unblock

1. **Pay for OpenRouter** — burst beyond 1-2 RPM per model. Estimate:
   2 500 preference pairs × $0.001/call avg × 5 voters = **$12 total**.
   Cheap enough to just buy. Requires user credential decision.

2. **Single-Gemini teacher** — drop the committee, keep just Gemini
   Flash Lite at 500 RPD. Loses the ensemble σ uncertainty signal
   but unblocks immediate labelling. Falls back to the SPEC-6-LOGIC-01
   single-teacher path.

3. **Lower `authoritative_max_std` from 0.15 to 0.30** — accept noisier
   labels into the student training set. Probably the right move
   regardless, since the disagreement between Gemini (harsh: 0.0) and
   Nemotron (lenient: 0.5) IS the signal — they look at different
   visual features.

4. **Lower `authoritative_min_responders` from 3 to 2** — accept
   2-of-4 quorum. Doubles the authoritative rate without paying.

## Recommended path

Combine 3 + 4: relax to (min_responders=2, max_std=0.30). With the
data we just collected:
- I000 portfolio: 3 voters, std=0.00 → authoritative ✓ (was)
- I000 pyvrp: 2 voters, std=0.15 → authoritative ✓ (new)
- I001 portfolio: 2 voters, std=0.38 → still not authoritative
- I001 pyvrp: 2 voters, std=0.25 → authoritative ✓ (new)

That would lift the authoritative rate from 1/10 → ~4/10. Still not
publishable headline, but enough signal to start building the
preference-pair dataset.

## Action items

a. Apply the relaxed thresholds.
b. Decide on paid vs single-Gemini path for the 2 500-pair dataset.
c. SPEC-6-LOGIC-01 student training stays gated on (a) + (b).
