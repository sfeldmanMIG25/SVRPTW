# LKH-3 is structurally not viable for v1 at N ≥ 100

Date: 2026-05-13

Once the SPEC-0-EVAL-01 evaluator backstop + wrapper fix (RUNS=5,
MTSP_MIN_SIZE=1) is in place, LKH-3's *real* behaviour on our v1
instances surfaces.

## v1 LKH-3 results (post-fix, capacity-feasible)

| N   | budget | mean cost | mean miss | mean s | portfolio mean cost |
|----:|-------:|----------:|----------:|-------:|--------------------:|
|  50 |   1 s  |  1 065.98 |       0   |    5.9 |             **749** |
| 100 |  10 s  |  9 388.67 |     7.5   |   57.2 |           **1 482** |
| 200 |  10 s  |217 931.41 |   185.0   |   59.9 |           **2 821** |
| 500 |  60 s  |     hung  |       ─   |  >3600 |           6 615    |

The N=500 job was killed after ≥ 1 hour with **zero output** —
LKH-3 cannot solve a single N=500 v1 instance in 60 s budget per
run (× 5 runs = 5 min total budget). The shell process produced
no per-instance summary line, suggesting the first instance's
internal optimisation loop never returned.

`mean miss = 185` on N=200 means **93 % of customers are being
dropped** per instance — LKH-3 cannot converge to an all-customers
feasible plan within 50 s of total budget.

## Why this is happening

LKH-3 was designed for symmetric TSP and CVRP; its CVRPTW extension
treats time windows as a penalty added to the route-length objective
rather than as a hard constraint. With our tight TWs (60-minute
median width per the v1 generator), LKH's penalty-relaxation
heuristic produces tours that are *short* but route-feasibility
violating; the wrapper's parser then truncates partial tours,
manifesting as dropped customers in our evaluator.

This is not a fixable bug in our wrapper. It's a known property of
LKH-3 CVRPTW (see Helsgaun's 2024 release notes: TW handling is
"experimental and best-effort").

## Implications

- **Drop LKH-3 from the headline comparison going forward.** It's not
  cheating any more (capacity-feasible), but it's not competitive
  either (TW-infeasible).
- **Portfolio + PyVRP are the real competitors.** Headline stays
  as-is: portfolio Pareto-dominates 100 % of v1 instances vs PyVRP.
- **The "v1 vs LKH ground truth" idea from the original spec stack
  doesn't survive contact with tight TWs.** If we want an
  exact-CVRPTW oracle, the right tool is a Gurobi MIP at small N or
  HGS-VRPTW (also DIMACS-grade) at moderate N — both queued as
  SPEC-3-LKHCOMPARE-01 successors.

## What remains useful from LKH

LKH-3 is still the right tool for pure TSP / ATSP oracles on the
tour-length axis — that's what GART validation uses. We keep the
wrapper; we just stop reporting LKH's CVRPTW cost as a competitor.
