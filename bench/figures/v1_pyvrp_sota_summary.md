# v1 PyVRP SOTA bench — completed 2026-05-13 00:30

Job `bfsrsec8e` ran ~6h overnight (started ~18:30 May 12). I earlier
wrote it off as dead — that was wrong; the output buffer stays empty
until the harness writes its summary table at the end.

| solver   | N   | n  | mean cost  | mean veh | mean s |
|----------|----:|---:|-----------:|---------:|-------:|
| pyvrp@30 | 200 | 40 |  16 813.50 |    14.38 |  30.12 |
| pyvrp@60 |  50 | 40 |   8 017.87 |     7.38 |  60.01 |
| pyvrp@60 | 100 | 40 |  10 372.13 |     9.10 |  60.04 |
| pyvrp@60 | 200 | 40 |  16 040.24 |    13.60 |  60.12 |

## Reading these against portfolio@10's 675 on Manhattan N=50

You can't, directly. The PyVRP run averaged across the full v1 city
set — Houston / Phoenix routes are several × longer than Manhattan's,
which alone explains a 10–12× cost gap before any solver-quality
comparison. The portfolio@10 = 675 number came from 10 Manhattan
N=50 instances.

A clean apples-to-apples comparison needs to re-run portfolio@10 on
the full v1 N=50 set, *under the capacity-aware evaluator*
(SPEC-0-EVAL-01). Queued.

## Whether these PyVRP numbers are tainted by the capacity bug

PyVRP is the DIMACS 2024 VRPTW winner; its internal feasibility
checks include capacity. We have no evidence it produces overloaded
routes — but we cannot prove the negative from the summary alone.
The proper audit is to re-evaluate each per-row solution under the
fixed `evaluate()` and check that `capacity_overload == 0`
everywhere. The bench JSON for this run wasn't written (only the
summary table), so the audit requires re-running.

## Implications

1. PyVRP@60 averages 7.38 vehicles at N=50 across the full v1 set.
   Portfolio@10 used 11.0 on Manhattan I000 alone. The vehicle-count
   gap is consistent with a city-mix story (Manhattan dense → many
   short routes; Phoenix sparse → fewer long routes), not necessarily
   solver quality.
2. The PyVRP run consumed 6 h for 160 (solver, instance) pairs.
   At 4 solvers × 160 = 640 cells, a full v1 bench in this style is
   a full-day run. We can do better with the 10s budget portfolio.

## Next

- Re-run portfolio@10 on full v1 N=50 set (40 instances, not 10),
  same evaluator, capture per-row JSON. Compare to pyvrp@60's 8017.
- Add `capacity_overload > 0` filter to leaderboard before publishing
  any number that has LKH-3 or PyVRP in it.
