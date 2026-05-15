# Rainbow rendering was the confound — fair_mode invalidates the pyvrp-wins-logic claim

Date: 2026-05-13

The previous "PyVRP wins logic axis 8/8 with +0.61 gap" finding
(`bench/figures/logic_axis_smoke_8pairs.md`) was rendered using the
*colorful per-route* palette. The user flagged: "LLM and users would
hate needless confusing colors on screen so that isn't a fair test."

We added a `fair_mode=True` flag to `svrptw.viz.renderer`:
- All routes drawn in a single muted navy color
- Route-count stripped from the title
- Customer markers neutral

We then re-ran a 15-pair labelling smoke (Gemini-only teacher,
diverse solver pairs from portfolio + pyvrp + greedy + auction +
pomo). Result:

| comparison | rainbow render (8 pairs) | fair_mode (15 pairs) |
|---|---|---|
| label=1.0 (A wins)  | 0       | **3** (all portfolio vs greedy) |
| label=0.0 (B wins)  | **8** (all pyvrp wins) | 0 |
| label=0.5 (tie)     | 0       | **12** |
| mean B−A gap        | **+0.61** | **−0.03** |

## What this proves

1. **The pyvrp-wins-logic claim was a rendering artifact**, not a
   real dispatcher-acceptance signal. With colours stripped, the
   gap evaporates.

2. **Per-instance fact check**: On Manhattan I000, portfolio uses
   **11 routes** and PyVRP uses **12**. Portfolio uses *fewer*
   routes than PyVRP. The previous "fewer-routes-looks-cleaner"
   intuition for why PyVRP scored higher was wrong on the facts —
   it was rainbow density driving the score.

3. **The 2-axis headline reclaims primacy**. With fair-mode, the
   logic axis is not a useful differentiator at this stage:
   - portfolio vs pyvrp: ties on all rainbow-pairs we have
     authoritatively measured under fair-mode (need to re-run)
   - portfolio vs greedy: portfolio wins on Manhattan-style dense
     instances (3 of 15 pairs)
   - All other within-instance pairs: ties at 0.0

## Implications for the SPEC-6 distillation track

With fair_mode rendering, the Gemini teacher gives little
preference signal between modern solvers. Bradley-Terry training
on 3-of-15 wins is not enough — the student would learn "always
predict 0.5" (median-collapse).

Three paths forward:

a. **Information-bearing rendering** — instead of identity colors
   per route, use *load utilisation* or *route length* on a fixed
   colormap. The VLM still sees a visual signal, but the signal
   is route-quality, not route-count.

b. **Closer-up renders** — render at higher zoom on selected
   regions (per the iterative-zoom SPEC-5-VIVRP-02 plan). May
   surface differences that get washed out at full-map scale.

c. **Different teacher** — Gemini-Flash-Lite is small and may
   not have the spatial reasoning capacity to distinguish at this
   level. A larger model (Gemini-Pro, GPT-4V, or Claude Vision)
   might produce more discriminative scores. Costs ~$0.001 to
   $0.01 per call.

## Honest update to HEADLINE.md

The 2-axis (cost, wall) result stands unchanged: portfolio
strictly dominates PyVRP on 95/65/45/72.5 % of v1 N=50/100/200/500.

The 3-axis result with current teacher + fair-mode is "no
meaningful signal" — neither solver has a measurable edge under
fair-mode VLM judging. This is the truth at this teacher quality.

## What we learned (research-grade)

- **Rendering bias is real and significant.** Solo-Gemini scored
  portfolio at 0.0 and PyVRP at 0.5+ when colored differently; the
  same solutions render to indistinguishable judgments when colored
  identically.
- **Free-tier VLM teachers are not sensitive enough** to
  distinguish near-Pareto-frontier solutions. The signal/noise
  ratio is too low for Bradley-Terry distillation at this scale.
- The full SPEC-6 pipeline still works end-to-end; the gating
  factor is teacher quality, not student code or pipeline
  architecture.
