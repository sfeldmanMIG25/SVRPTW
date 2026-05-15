# v1 ViVRP-Gemini smoke (10 instances × 4 solvers, N=50)

Date: 2026-05-12
Judge: gemini-3.1-flash-lite (Track 3 ViVRP backend)

| solver        | avg cost | gemini vivrp | wall s | feas |
|---------------|---------:|-------------:|-------:|-----:|
| portfolio@10  |   629.93 |     **1.70** |   3.24 | 1.00 |
| auction_gart  |   632.94 |         2.00 |   4.46 | 1.00 |
| greedy        |   745.93 |         2.00 |   0.00 | 1.00 |
| lkh3@1        |   774.27 |     **4.50** |   1.36 | 1.00 |

## What this says

Our portfolio wins the **cost** axis decisively (-18.7 % vs LKH-3) and
roughly ties the **time** axis (3.24 s vs 1.36 s — same order of
magnitude). LKH-3 wins the **logic** axis (Gemini ViVRP) by ~2.6×.

This is the divergence SPEC-6-LOGIC-01 / SPEC-6-PARETO-3AXIS-01 was
written to surface: at the cost frontier our solver is ahead;
under a dispatcher-acceptance objective the picture flips. The
3-axis Pareto framing is the only honest way to report this — a
single-cost headline would either falsely claim a clean win or
falsely cede the contest to LKH-3.

Two candidate explanations for the logic-axis gap (queued for
investigation as we land the student):

1. **Fewer routes / cleaner partition.** LKH-3's implicit drop +
   re-pack yields fewer vehicles per solution; the judge reads that
   as more shippable. If true, vehicle_kill and soft_drop need more
   bandit weight at small N.
2. **Geometry artifact.** Gemini may be biased by visual route-count
   density and not by the per-stop decisions a real dispatcher
   would scrutinise. The ensemble + isotonic calibration in
   SPEC-6-LOGIC-01 are exactly the defense against this.

We do not know which until the preference-pair labeling lands and we
can correlate vivrp scores against `num_vehicles_used` and
`total_distance_miles`. Either way, the headline number we ship at
v1 will be the 3-axis dominance fraction, not raw cost.
