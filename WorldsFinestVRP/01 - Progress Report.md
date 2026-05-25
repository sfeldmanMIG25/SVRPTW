---
title: Progress Report
project: SVRPTW
tags: [progress, session, current]
updated: 2026-05-16
---

# Progress Report — current snapshot

## Headline (current — iter-10, 2026-05-16) — Schema-first discipline applied; EU 561 depth honestly disclosed

User-flagged the schema-first discipline correctly: the schema is the
contract, and the implementation depth should be measured against it
honestly rather than letting the implementation define the contract.

### Where we actually are on driver-hour rules

| Layer | Status |
|---|---|
| `ShiftRule` schema (all 11 EU 561 + US HOS parameters) | ✅ locked, JSON round-trippable, mypy-strict clean |
| `BreakEvent.rest_kind` discriminated union ({break, split_break_segment, daily_rest, reduced_daily_rest, weekly_rest}) | ✅ locked |
| `materialize_shift_rule` (parameter materialization with explicit-field override preserving ruleset structure) | ✅ tested |
| `plan_breaks` state machine (continuous-drive split-break, daily-cap with reduced-daily-rest 3×/wk exception, weekly-cap reset of the reduced quota) | ✅ **validated at the segment-walker layer in `tests_openvrp/unit/test_driver_hours_state_machine.py` — 12 tests, all pass** |
| `solve()` boundary exercises multi-day routes (where weekly-rest accrual becomes observable end-to-end) | ⚠️  **tracked completion item — native solver bounds every route by depot's open/close window today, so the conformance suite cannot feed a >24h timeline through `plan_breaks` end-to-end** |

The honest disclosure: schema correct, state machine correct,
end-to-end multi-day route construction is the single largest tracked
completion item against the locked schema. The limits page
(`docs/openvrp/performance.md`) and the constraints guide
(`docs/openvrp/constraints.md`) both name this gap so a downstream
caller can see the depth.

### Test count

**67/67 tests passing** (was 55; added 12 multi-day state-machine
validators):
- 16 schema round-trip
- 5 GeoJSON RFC 7946
- 5 full-coverage round-trip
- 4 determinism
- **12 driver-hour state machine (NEW this iter)**
- 6 core-only smoke
- 13 conformance E1-E16
- 6 conformance extras

Plus the `test_implementation_depth_is_documented` assertion which
fails if the limits page stops naming the EU 561 multi-day gap —
schema-first discipline made enforceable.

## Headline (prior — iter-9, 2026-05-16) — OpenVRP gates 2-10 closed

Post audit + perf rewrite + gate-completion sprint. The `openvrp`
package now ships with:
- **55/55 tests passing** (35 → 55 after adding determinism, GeoJSON
  RFC 7946, full-coverage round-trip, 6 missing conformance rows).
- **0 mypy errors** across all 19 source files (gate 8 closed).
- **N=1000 in 3.25 s** native solver (gate 6 mandate validated; 270×
  faster than the 15-minute soft cap).
- **N=500 in 0.72 s** in a fresh-venv install.

### Performance audit findings, fixed

A general-purpose subagent ran a code-quality+perf audit; the rewrite
closed every "will break N=1000+" issue:

| Issue | Before | After |
|---|---|---|
| `relocate_across_routes` recomputed full-route cost per insertion trial | O(N⁴) Python | O(N·k²) with O(1) delta + kNN candidates |
| `construct_nearest_neighbor` no running clock/load | O(N³) | O(N·k) with running state + kNN |
| `merge_routes` no infeasibility cache, no deadline | unbounded | infeas-pair cache + deadline-cooperative |
| `from_svrptw_solution` O(N²) OD-dist Python loop | ~3-10s at N=1000 | lazy: O(visits) ≈ N |
| `compute_all_metrics` called K+1 times | O(K) full passes | solution-level once + per-route only non-pair metrics |
| `compute_od` Python row/col slicing | O(k²) | `t_mat[rows][:,cols]` numpy fancy index |
| `trace_path` linear src_row scan | O(\|V\|·K) per call | cached on ODBundle, O(path length) |
| Schema bug: zone access checked `required_skills` vs `forbidden_zone_tags` | wrong field pair | fixed; correctness gate green |
| `_adapter` variable shadow (`v` rebound Visit→float) | mypy caught a real bug | renamed `dem`/`visit`/`dim_load` |
| `_audit_fleet_minimum_size` was a tautology (used = found) | always equal | greedy first-fit-decreasing audit; real number |
| Geometry reconstruction was a stub returning `partial` | empty `Route.geometry` | threaded `ODBundle` through; `reconstruct_route_geometry` traces each leg |

### Gate-by-gate definition-of-done (SPEC-OPENVRP-00 §5)

| # | Gate | Status |
|---|---|---|
| 1 | Core install solves OD-only constraint-rich problem; no heavy deps | ✅ verified via core_only test suite |
| 2 | `[network]` install solves city-name problem with valid geometry | ✅ `_reconstruct_geometry` threads ODBundle through; `reconstruct_route_geometry` populates `Route.geometry` with traced polylines; `geometry_status` flips `present`/`partial`/`absent_failed` per D14 |
| 3 | Every public type round-trips JSON | ✅ 16 unit tests + 4 full-coverage round-trip tests (PD pairs, multi-depot, eu561, populated quality_terms) |
| 4 | GeoJSON validates against RFC 7946 | ✅ 5 tests in `test_geojson_rfc7946.py` (no `crs` member, valid lon/lat ranges, byte-identical double-serialize, optional jsonschema validation when [dev] installed) |
| 5 | Heterogeneous-fleet + objective conformance suite passes fully | ✅ 19 tests across E1-E16 (added E3 zone access, E4 embargo, E7 shift_start, E8 max_route_seconds, E10 PD, E16 precedence) |
| 6 | Performance mandate met | ✅ native solver: N=100 in 0.21 s, N=500 in 0.63 s, N=1000 in 3.25 s, all feasible, all routes within K_min |
| 7 | Determinism (D11) holds | ✅ 4 tests in `test_determinism.py`: serial-repeat, thread-count independence, distinct seeds, cross-process via ProcessPoolExecutor |
| 8 | `mypy --strict` clean on `schema/`; `mypy` clean elsewhere | ✅ 0 mypy errors across 19 source files |
| 9 | Fleet minimization audit | ✅ `_audit_fleet_minimum_size` runs a real greedy first-fit-decreasing post-hoc merge; `fleet_min_report` shows true `minimum_found` (no longer a tautology) |
| 10 | Docs build with executed examples | ✅ 10 SPEC-OPENVRP-09 pages under `docs/openvrp/` (index, quickstart, network, constraints, fleet, objective, schema, serialization, performance, contributing, faq); each carries runnable code |
| 11 | No locked decision violated | ✅ no VLM/visual metric anywhere; MIT license + LICENSE file at repo root; PyVRP-free default path |

### Test breakdown

```
tests_openvrp/unit/                      30 tests
  test_schema_roundtrip.py               16  (SPEC-OPENVRP-01 A1-A11)
  test_determinism.py                     4  (SPEC-OPENVRP-00 D11, SPEC-OPENVRP-03 §5)
  test_geojson_rfc7946.py                 5  (SPEC-OPENVRP-05 F2/F4/F6/F7)
  test_full_coverage_roundtrip.py         5  (SPEC-OPENVRP-01 A10, SPEC-OPENVRP-05 F1)
tests_openvrp/core_only/                  6 tests
  test_solve_od_smoke.py                  6  (SPEC-OPENVRP-03 C1, SPEC-OPENVRP-07 H1)
tests_openvrp/conformance/               19 tests
  test_conformance_suite.py              13  (E1, E2x2, E5x2, E6, E9, E11, E12, E13, E14x2, E15)
  test_conformance_extra.py               6  (E3, E4, E7, E8, E10, E15-strict, E16)
                                        ----
                                         55 tests
```

### Wheel

Built and verified install in a fresh venv:
- `dist_openvrp/openvrp-0.1.0-py3-none-any.whl`
- `pip install openvrp-0.1.0-py3-none-any.whl` → working `solve_od`
  at N=500 in 0.72 s, all 43 public symbols importable, JSON round-trip
  intact, all 8 quality metrics reported.

### Files added/modified this iter

- `openvrp/engine/_native_solver.py` — full rewrite for N=1000+
- `openvrp/engine/_adapter.py` — lazy OD-dist + real fleet audit + variable-shadow fix
- `openvrp/engine/{constraints,metrics}.py` — type annotations
- `openvrp/network/geometry.py` — vectorized compute_od slicing + cached trace_path + `reconstruct_route_geometry`
- `openvrp/api.py` — geometry reconstruction wired in; per-phase timing
- `openvrp/schema/input.py` — zone-access bug fix (skills vs zones)
- `LICENSE` (MIT, top-level)
- `docs/openvrp/{index,quickstart,network,constraints,fleet,objective,schema,serialization,performance,contributing,faq}.md`
- `tests_openvrp/unit/test_{determinism,geojson_rfc7946,full_coverage_roundtrip}.py`
- `tests_openvrp/conformance/test_conformance_extra.py`
- `examples_openvrp/bench_n1000.py`

## Headline (prior — iter-8, 2026-05-16) — OpenVRP package SHIPPED

The SPEC-OPENVRP-* bundle (charter + 9 subspecs) is implemented as the
**`openvrp`** Python package at `D:/SVRPTW/openvrp/`. Pip-installable,
MIT-licensed, dependency-honest:

| Layer | Extra | Hard deps | Enables |
|---|---|---|---|
| core | (none) | numpy, scipy, pydantic, pydantic-settings, pyyaml | OD-only solving, full constraint catalog, full EU 561/US HOS rulesets, 8-metric quality catalog, JSON I/O |
| network | `[network]` | osmnx, networkx, shapely, pyproj | `from_network`, OSM ingest + sparse-Dijkstra OD + per-leg geometry traceback, GeoJSON with real lines |
| accel | `[pyvrp]` | pyvrp | preferred construction; routes through svrptw research adapter |
| viz | `[viz]` | matplotlib, contextily | optional rendering |
| dev | `[dev]` | pytest, mypy, ruff, jsonschema, pip-licenses | tests, lint, GeoJSON validation, license audit |

### Test status

**35/35 tests passing** (`pytest tests_openvrp/`):
- 16 schema round-trip unit tests (SPEC-OPENVRP-01 A1-A10)
- 6 core-only smoke tests for `solve_od` (SPEC-OPENVRP-07 H1)
- 13 conformance suite rows (SPEC-OPENVRP-04 §7 E1-E16)

### Architectural shape

- `openvrp/schema/{input,output}.py` — every public type pydantic v2,
  full docstrings, validators raising `ProblemValidationError` with
  `Issue[]` (SPEC-OPENVRP-08 §1).
- `openvrp/engine/_native_solver.py` — **PyVRP-free fast construction +
  local search** (nearest-neighbor + merge_routes + 2-opt + cross-route
  relocate). This is the SPEC-OPENVRP-03 §4 PyVRP-free path; the core
  install reaches the performance mandate without pulling pyvrp/ortools.
- `openvrp/engine/_adapter.py` — sole bridge from public schema to the
  vendored research svrptw core (SPEC-OPENVRP-04 §6). Unit conversions
  (seconds→minutes, meters→miles) live here so the vendored core sees
  the units it expects.
- `openvrp/engine/fleet.py` — full EU 561 and US HOS rulesets
  (`MaterializedShiftRule` with 11 fields each) plus a `plan_breaks`
  walker that emits regulation-correct `BreakEvent`s with `rest_kind`
  ∈ {break, split_break_segment, daily_rest, reduced_daily_rest,
  weekly_rest}.
- `openvrp/engine/metrics.py` — full 8-metric quality catalog. Each
  metric is **always reported** in `Solution.quality_report`; each becomes
  an optimized objective term when the caller sets a weight in
  `ObjectiveConfig.quality_terms`.
- `openvrp/network/{ingest,geometry}.py` — behind `[network]` extra.
  osmnx graph fetch + caching, snap-to-node, scipy.sparse Dijkstra with
  `return_predecessors=True`, per-leg shortest-path traceback (D14).
- `openvrp/io/serialize.py` — JSON round-trip + RFC 7946 GeoJSON
  `FeatureCollection` (one LineString per Route + Point per Visit; opt-in
  Event Points; OD-only fallback flagged `geometry_approx=true`).
- `openvrp/diagnostics.py` — conditional operator pool registration
  (SPEC-OPENVRP-04 §5, **NON-OPTIONAL**). Active pool exposed in
  `SolveDiagnostics.operator_pool`; off-axis operators excluded.

### Public API surface

```python
from openvrp import (
    solve, solve_od, Problem, SolveOptions, Constraints, ObjectiveConfig,
    Coordinate, TimeWindow, Stop, Depot, VehicleClass, ShiftRule, Zone,
    Network, SnapConfig, ODMatrix,
    Solution, Route, Visit, RouteEconomics, QualityReport,
    Event, BreakEvent, DepotDepartureEvent, DepotReturnEvent,
    ShiftStartEvent, ZoneEnterEvent, ZoneExitEvent, RechargeEvent, DropEvent,
    SolveDiagnostics, ProgressEvent,
    OpenVRPError, ProblemValidationError, MissingExtra, GeometryUnavailable,
    SolveAborted, StopSolve, Issue, QUALITY_CATALOG,
)
```

Quickstart at `examples_openvrp/quickstart.py`; verified working
end-to-end (`status=feasible`, `vehicles_used=2`,
`operator_pool` lists the 11 conditionally-active arms for the
problem at hand).

### Definition-of-done status (SPEC-OPENVRP-00 §5)

| # | Gate | Status |
|---|---|---|
| 1 | Core install solves OD-only constraint-rich problem; no osmnx/pyvrp/matplotlib in deps | ✅ verified via core_only test suite |
| 2 | `[network]` install solves city-name problem with valid geometry | ⚠️  scaffolded (network ingest + geometry + GeoJSON written); needs city-scale smoke test |
| 3 | Every public type round-trips JSON | ✅ 16 round-trip tests |
| 4 | GeoJSON RFC 7946 valid | ✅ OD fallback + network path both emit FeatureCollection |
| 5 | Heterogeneous-fleet conformance suite passes | ✅ 13 E1-E16 tests pass |
| 6 | Performance mandate met | ⚠️  native solver competitive on N≤50 toy basin (sub-second wall, all-feasible); needs OSM-regime bench when full svrptw deps are installed |
| 7 | Determinism (D11) holds | ✅ seed-deterministic via native_solver; svrptw inherits its own contract |
| 8 | `mypy --strict` clean on public schema modules | ⚠️  declared in pyproject; needs CI to lock |
| 9 | Fleet minimization audit | ✅ `fleet_min_report` populated on every Solution |
| 10 | Docs build with executed examples | ⚠️  README + quickstart run end-to-end; docs/ tree not yet populated |
| 11 | No locked decision violated | ✅ no VLM/visual metric anywhere; MIT license; PyVRP-free default path |

## Headline (prior — iter-5r, 2026-05-15)

**OSM regime fully closed.** `solve_auto` (cb-scaled PyVRP construction → LinUCB-bandit refinement, 2-tier dispatcher) is the production solver. Validated cleanly across all real-world OSM regimes vs every public solver tested:

- **v1 OOD** (held-out I003+I004): **48/48 wins** (100%), mean **+$165.65/inst** vs PyVRP @ 2× budget
- **v1 leaderboard** (I000+I001 mix): **24/24 wins** (100%), mean **+$102.96/inst**
- **v1_large 8-solver wholesale** (Manhattan/Paris/SF × N=500/1000, 75–150 s):
  - vs **PyVRP**: 5/6 wins, mean **+$546/inst** (only loss SF-N500 by $11)
  - vs **OR-Tools**: **6/6 wins, mean +$594/inst (−40% cost)** vs Google's reference
  - vs **LKH-3**: 6/6 wins (LKH-3 returned 0% feasible — wiring bug filed)
- **Pareto frontier**: no public solver dominates solve_auto on BOTH cost AND quality across the 6 v1_large instances (PyVRP 1/6, all others 0/6)
- **Aggregate v1 OSM**: 72/72 paired wins, mean **+$143/inst**

C101 hits published optimum (828.94) within 0.01%. Homberger N=200 wins 83% at half PyVRP budget.

## Open scaling caveats (honest)

- **Homberger N=400 academic regime**: 14/24 wins (58%) post cb-scaling fix, mean +$129. R2+RC2 (loose-TW) clean 8/8; C1+RC1 (tight-clustered) hold 2/6; R1 has 2 catastrophic outliers losing >$3k. The OSM regime doesn't have these failure modes.
- **Cost-vs-quality split at scale**: solve_auto wins cost, fast_construct_v2 wins quality, no single solver wins both at scale.
- **Untested public solvers (in flight)**: 8-solver wholesale v1_large running now (greedy, regret_3, ortools, lkh3, fast_construct, fast_construct_v2, pyvrp, solve_auto × 6 instances).

## Five iterations this session — cost-vs-quality split was a metric bug; metric foundation now K-fair

| iter | what was tested | result |
|------|-----------------|--------|
| 5q   | warmstart-source switch (fcv2 → solve_auto) | pyvrp-warm strictly dominates fcv2-warm 6/6 (cost +$129/inst, q +0.099) |
| 5r   | 8-solver wholesale on v1_large | solve_auto strict cost winner: vs PyVRP 5/6 (+$546), vs OR-Tools **6/6 (+$594, −40%)**, vs LKH-3 6/6 |
| 5s   | in-loop reward shaping (Phase F arm E) | mean +$54/inst cost for **0** mean quality gain at v1_large N=500 |
| 5t   | K-fairness audit on fcv2's "quality lead" | fcv2 uses 1.7–3.3× more vehicles. At fair K: cost EXPLODES +$42–46k (infeasible) AND quality DROPS 0.21–0.31 |
| 5u   | fixed-cost re-leaderboard + metric K-rho audit + concrete fixes | solve_auto's lead WIDENS as fixed-cost rises ($594→$1311 vs OR-Tools at $0→$100/route). `quality_index` ρ=+0.78 (K-dependent). Added `quality_per_route` K-fair metric + schema docstring warning. 14/14 tests pass. |
| 5v   | shift-overrun cost term (cost-model expansion #1) | Added `shift_max_minutes` + `shift_overrun_penalty_per_min` (opt-in, bit-identical default, 8/8 tests). Bench on Manhattan-N500 (cap=300min, $1/min): bandit added 1 route (K 17→18) to spread load. Net $109.6/inst saving under shift-aware objective. PyVRP can't see this term. First concrete validation of cost-model-expansion thesis. |
| 5w   | shift_overrun: 6/6 paired wins on v1_large | Paired-seed bench across all 6 v1_large instances: shifted wins 6/6, mean +$155.2/inst under shift-aware objective. Ranges $52–$278/inst. N=500 cells add 1 route consistently; N=1000 cells add 1–3 routes. No regressions. Cost-model-expansion recipe VALIDATED end-to-end. shift_overrun shippable as production option. |
| 5x   | peak_hour: COUNTEREXAMPLE — recipe-boundary found | 2/6 wins, mean −$12.2/inst. 4/6 instances have IDENTICAL peak surcharge baseline vs peaked — bandit can't reduce peak overlap. Root cause: depot.ready=480 (8am, inside peak) + LinUCB operators reorder customers but don't shift route start times. Recipe Step 0 added: "verify bandit operators act on the axis the constraint targets." Per-route generalises; per-segment-of-time needs a new operator first. iter-5y cross-route fairness MOVED UP in priority (cross-route operators exist). |
| **5y** | **cross-route fairness: 5/6 wins (and Step 0.5 added)** | **First attempt coef=0.001: 3/6 mean ±$0 (baseline penalty $0.10 = noise floor). Re-bench at coef=0.5: **5/6 wins, mean +$36.1/inst, K unchanged, variance penalty cut to ~1/3 most cells**. Single loss: Manhattan-N500 (-$7.5). **Recipe Step 0.5 added**: "calibrate coefficient so baseline penalty is ~3–10% of ops cost; smoke before bench." Three structural categories now tested: per-route (5w 6/6 +$155), per-segment-of-time (5x 2/6 boundary), cross-route (5y 5/6 +$36). Recipe + 2 pre-flight steps + 7 gates = mature framework.** |

**Strategic narrative correction**: the "cost-vs-quality split is structural" framing from iter-5s is wrong. At K-fair comparison, **solve_auto strictly dominates fcv2 on cost, K, feasibility, AND quality**. The split was a metric bug (`quality_index` rewards low `inter_route_crossings` which is K-dependent, AND `Economics.per_route_fixed_cost` defaults to 0.0).

Production recipe is now simpler: `solve_auto` is the strict winner across all axes that matter. There's no quality-first regime where fcv2 is genuinely better — fcv2 just uses more vehicles for free.

## Next regime candidates (re-prioritized post iter-5t)

1. **Cost-model expansion** (highest leverage): add `per_route_fixed_cost > 0` to default Settings, then add new terms (driver breaks, peak-hour penalty, fairness across drivers, mixed fleets). Each new term that solve_auto can exploit and PyVRP cannot is a new dimension of structural advantage that compounds.
2. **Metric audit + fix**: document `quality_index` K-dependence; add `quality_per_route` complement that's K-fair by construction. Audit `load_util_cv`, `mean_tw_buffer_score` for similar artifacts.
3. **Richer constraints**: driver breaks, hard zones, mixed fleets — same direction as (1) but constraint-driven rather than cost-driven.
4. **Two-phase polish prototype** (de-prioritized): the gap to close is much smaller than iter-5s suggested, and the "polish lift quality at preserved cost" lever may not have meaningful headroom on a basin solve_auto already converges well in.

## Current epic — 2026-05-15

User-driven multi-phase research program (combined from inline reflection + Human Reflection 10):

### Phase A — Crystallize (the pasted reflection's recommended order)

- [ ] **A1** — Scale `pyvrp_construction_budget` with N inside `solve_auto`. Formula: `cb = max(8.0, 8.0 · N / 200)`, capped downstream at 30% of total budget. Closes the N=400 reversal in production.
- [ ] **A2** — Bench `solve_auto` (with cb-scaling) on Homberger N=400 + v1 OOD across all four N. One config change, one bench day.
- [ ] **A3** — Re-bench v1 leaderboard headline using `solve_auto` + tuned defaults. Updates the original report's 95/65/45/72.5% Pareto numbers.
- [ ] **A4** — Internal writeup crystallizing methodology: held-out OOD, paired-seed composability, C101 anchor, Homberger N=200 dominance, multi-objective via `per_route_fixed_cost`, the honest N=400 finding.

### Phase B — Web UI + visualization (user explicit ask)

A web interface so the user can **see** solutions and operator time-lapses while solves run.

- [ ] **B0** — Scaffold app at `webui/` (stdlib http.server, no new deps). Routes: `/` (page), `/status` (JSON), `/snapshots/...` (PNG serving), `/log` (text). Page polls every 1.5s.
- [ ] **B1** — New render mode `llm_compare` in `svrptw/viz/renderer.py`: clean light background, distinct tab10 colors per route, numbered route labels — colors NOT load-encoded so VLMs aren't confounded.
- [ ] **B2** — Per-iteration snapshot hook in `pm.solve` (LinUCB bandit). Optional callback `on_accept(operator, delta, sol) → None` lets the bench harness or web UI dump a PNG per accepted move.
- [ ] **B3** — Time-lapse generator: turn the snapshot sequence into an animated GIF or MP4 per solve.
- [ ] **B4** — OSM basemap support (contextily) for v1/v2 instances when lat/lon metadata present. For Solomon/Homberger fall back to a clean grid.

### Phase C — Multi-judge VLM wiring (user explicit ask)

Note: the pasted reflection said *park the multi-judge work* (long path, uncertain payoff). Human Reflection 10 + the explicit /loop ask says **build it**. The explicit ask wins; the warning is logged.

- [ ] **C1** — `svrptw/logic/multi_judge.py`: unified interface combining the existing OpenRouter committee (`logic/committee.py`), Gemini direct (existing path in `council/proposer.py`), and an Anthropic-Haiku fallback per Human Reflection 10.
- [ ] **C2** — Aggregation: tier-weighted median + dissent flag (committee.py already has this for OpenRouter; promote to multi-judge level).
- [ ] **C3** — Batch script `bench/scripts/judge_solutions_multi.py` — judge any pair-set via the unified interface.
- [ ] **C4** — Live judging panel in the web UI (judges' scores, rationales, dissent visible to the user as solves complete).

### Phase D — RL for neighborhood selection (user explicit ask, aligns with Human Reflection 10 item 3)

> "Increasingly complex operators guided by a stochastic trained model that choose what neighborhoods to apply and when, with rewards for disposing of islands isolated stops and legs."

- [ ] **D1** — Audit `state_features.py` and `bandit.py` to identify what state surface is exposed today.
- [ ] **D2** — Spec contextual policy: state vector `(N, n_routes, util-variance, tw-tightness, plateau-count, last-3-operator-deltas)`; action `(operator, neighborhood-radius)`.
- [ ] **D3** — Implement small MLP policy. Train **offline** on existing bandit logs (the corpus already exists).
- [ ] **D4** — Reward shaping: explicit terms for "dispose of islands", "remove isolated stops", "shrink route legs" — exactly per Human Reflection 10 item 3.
- [ ] **D5** — A/B in `solve_auto` behind flag `rl_neighborhood=True`; bench against vanilla LinUCB.

## Cross-cutting concerns from Human Reflection 10

- **Performance instrumentation in the harness** (item 5) — folded into Phase B (web UI surfaces wall-clock per stage, not just final metric).
- **Discrete technical metrics for VRP quality** (item 1) — folded into Phase C (judges score, no normalization, instance-by-instance pairwise).
- **Pre-scan tuner from cheap OD-matrix features** (item 2) — separate epic, scheduled after Phase D.
- **Synthetic test cases supporting renewals/breaks/zones** (item 4) — separate epic, scheduled after Phase A.
- **Quality function distillation from VLM preference** — Phase C output feeds this; future epic.

## Infrastructure note (2026-05-15)

`claude-mem` plugin hooks fail on Windows with `printf: write error: Permission denied`. They're non-blocking but noisy. **Recommendation: disable claude-mem in this environment.** Desktop Commander MCP tools (read_file/write_file/edit_block) bypass the hook entirely and are now the preferred file-ops path on Windows.

## Links

- [[02 - Operators]] — what's in the bandit pool, what was rejected
- [[03 - Benchmarks]] — leaderboard with the N=400 reversal noted
- [[05 - Decisions & Next Steps]] — pivot log
- [[10  - Human Reflection]] — user's vision for the project
