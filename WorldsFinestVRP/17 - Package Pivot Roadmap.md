---
title: Package Pivot Roadmap
project: SVRPTW
tags: [pivot, package-build, complex-constraints, iter-6a-series]
date: 2026-05-15
---

# Package Pivot — full Python package for network-VRP with complex operational constraints

User direction (this session): "Pivoting objective, build a full python
package for solving network-based VRP with complex operation constraints."

## What we have (10 constraints / cost terms in `Economics`)

| # | term | iter | status |
|---|------|------|--------|
| 1 | Time windows (hard) | baseline | shipped |
| 2 | Capacity (hard, with overload-penalty) | baseline | shipped |
| 3 | Asymmetric network OD (osmnx + scipy Dijkstra) | iter-5n | shipped |
| 4 | `per_route_fixed_cost` | legacy | shipped (default 0.0; docstring warns about K-fairness) |
| 5 | `crossings_penalty_per_pair` (Phase F) | iter-5l | opt-in |
| 6 | `util_imbalance_penalty_coef` (Phase F) | iter-5l | opt-in |
| 7 | `tw_buffer_bonus_coef` (Phase F) | iter-5l | opt-in |
| 8 | `shift_max_minutes` + `shift_overrun_penalty_per_min` | iter-5v/w | shipped (6/6 wins, +$155/inst) |
| 9 | `peak_window_starts/ends` + `peak_hour_wage_multiplier` | iter-5x | plumbed but operator-bound (no time-shift op) |
| 10 | `driver_time_variance_penalty_coef` (fairness) | iter-5y | shipped at coef=0.5 (5/6 wins, +$36/inst, Manhattan-N500 caveat) |
| **11** | **`driving_max_minutes` + `break_violation_penalty_per_min` (EU 561 / US HOS lite)** | **iter-6a-1** | **NEW THIS TURN — plumbed, 14/14 tests pass, bench pending** |

## Built this session (iter-6a batch) — ALL 6/6 BENCHED AND SHIPPABLE

| # | constraint | iter | status | bench verdict |
|---|-----------|------|--------|---------------|
| 11 | driver_breaks (EU 561 / US HOS lite) | iter-6a-1-bis | ✅ shipped | **6/6 +$179.7/inst** at tight cap |
| 12 | hard_zones (global embargo windows, per-visit penalty) | iter-6a-2 | ✅ shipped | **6/6 +$1796.7/inst** (biggest win) |
| 13 | mixed_fleets (cheapest-class assignment, additive premiums) | iter-6a-3 | ✅ shipped | **6/6 +$64.4/inst** |
| 14 | electric_vehicles (per-route distance limit) | iter-6a-4 | ✅ shipped | **6/6 +$58.4/inst** |
| 16 | pickup_delivery (precedence pair penalty) | iter-6a-6 | ✅ shipped | **6/6 +$51.2/inst** (small) |
| 18b | min_routes_required (labor contract) | iter-6a-8 | ✅ shipped | **6/6 +$300.3/inst** |
| -- | Phase F lite refactor (util + tw_buffer inline) | iter-6a-perf | ✅ landed | 15-stack at 4.89x overhead |
| -- | wholesale comparison hard-timeout guard | iter-6a-cleanup-2 | ✅ landed | re-bench in flight |

**22/22 cost-term unit tests pass.** All 5 new constraints have:
- Opt-in `Economics` field(s), default 0/empty (bit-identical when off)
- Gated penalty block in `evaluate()` with sane non-zero checks
- 2 unit tests (linearity + gating)
- Runtime smoke confirming the term computes on real instances
- Step 0 (operator-coverage) check documented

## Still NOT built (next session)

| # | constraint | category | reason deferred |
|---|-----------|----------|-----------------|
| 15 | **Multi-depot** (multiple start/end depots) | structural | requires Instance schema change (depot → depots list) |
| 17 | **Driver skills / customer-vehicle matching** | matching | requires Customer.required_skills field + class-skill mapping |
| 18 | **Multi-day routing** | temporal | requires multi-day Solution schema |
| 19 | **Start-time-shift operator** (unblocks iter-5x peak_hour) | operator | real solver work in pm.solve / arms set |
| --- | per-customer embargo (refinement of iter-6a-2) | spatial+time | needs Customer.embargo_windows field |
| --- | charging-station insertion (refinement of iter-6a-4) | spatial | needs new node-type in routes |

## Pivot recipe (extends iter-5v/w/x/y framework)

The 7-gate recipe + Steps 0/0.5 is the proven flow. Each new constraint:
1. **Step 0**: identify which existing bandit operator(s) act on the
   constraint's axis. If none, queue the operator first.
2. **Step 0.5**: estimate the coefficient that makes baseline penalty
   3-10% of ops cost.
3. **Gates 1-7**: schema field + gated evaluator + 2 unit tests + smoke +
   paired 6-instance bench + 6/6 ship threshold.

## Package structure (target)

```
svrptw/
  config/           # schema + presets
  io/               # Instance loading, network OD generation
  metrics/          # quality_index, quality_per_route, graph_quality
  solvers/
    classical/      # solve_auto, pyvrp, fast_construct_v2, lkh3, etc
    common/         # operators, evaluate()
  constraints/      # NEW SUBPACKAGE -- explicit constraint catalog
    __init__.py     # registry + Constraint base
    breaks.py       # iter-6a-1 driver breaks (EU 561 / US HOS)  ← landing now
    hard_zones.py   # iter-6a-2 forbidden geographies
    fleets.py       # iter-6a-3 mixed fleets
    electric.py     # iter-6a-4 EVs
    multi_depot.py  # iter-6a-5
    pickup_delivery.py  # iter-6a-6
    skills.py       # iter-6a-7
  viz/              # renderer
  sim/              # stochastic simulator (parked but kept)
webui/              # dashboard
tests/              # unit + integration
bench/              # bench harness (prune ~50 → 15 reusable scripts)
examples/           # NEW -- one example per constraint
```

## Phasing (this session continues)

- iter-6a-1 driver_breaks (DONE this turn -- 14/14 tests pass, bench tomorrow)
- iter-6a-2 hard_zones (next iteration, this session if possible)
- iter-6a-3 mixed_fleets (next session)
- iter-6a-4 EVs (next session)
- iter-6a-5 multi_depot
- iter-6a-6 pickup_delivery
- iter-6a-7 skills
- iter-6a-8 multi-day (lower priority)
- iter-5w-bis start-time-shift operator (unblocks iter-5x; insert when convenient)

After all 7-8 land + each clears the recipe gates, we have a complete
shippable constraint catalog. Then: API stabilization (Problem/Solver/
Solution typed interface), `examples/` directory, README, version 0.1
release.

## What this changes about ongoing work

- **Wholesale re-bench** (in flight): still useful — provides the
  baseline cost-axis leaderboard for the package's README.
- **iter-5z polish prototype**: parked. K-fair analysis showed small
  headroom; pivot priority is higher.
- **iter-5w-bis start-time-shift operator**: re-prioritized — insert when
  natural (e.g., between iter-6a-3 and iter-6a-4) since it unblocks
  peak_hour (cost term already plumbed).
- **Council / LLM proposer / single-judge VLM / POMO / logic-axis**:
  remain closed.

## First package release target

`svrptw 0.1.0` ship criteria:
- All 7 iter-6a constraints implemented + tested
- Each constraint cleared recipe Step 0 + Step 0.5 + Gates 1-7
- Public API: `from svrptw import Problem, Solver, Solution, Constraint`
- README with quickstart + per-constraint examples
- `examples/` directory with one runnable per constraint
- pip-installable via `pip install -e .`
- Wholesale leaderboard table showing solve_auto vs PyVRP vs OR-Tools vs LKH-3 on the v1_large benchmark
