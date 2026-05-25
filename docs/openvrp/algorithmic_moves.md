# Algorithmic moves (safety-review surface)

Every move the openvrp solver makes — construction, search, finalize,
termination. Read this page when you want to know exactly what the
solver does to your data and where you can intercept.

## Where the work happens

```
solve(problem, options)
  │
  ├─ ingest
  │   ├─ load_network()           [network mode only — osmnx fetch, cache hit ⇒ no I/O]
  │   ├─ snap_to_nodes()          [scipy cKDTree shortlist → haversine refine]
  │   ├─ compute_od()             [scipy.sparse.dijkstra w/ predecessors]
  │   └─ triangle_inequality_sample()  [200 random triples; warn-only per D16]
  │
  ├─ construct + refine (delegated to native or pyvrp solver)
  │   │
  │   ├─ native (default, no extras):
  │   │   ├─ _build_knn()         [O(N² log N) once; k=20 candidate lists]
  │   │   ├─ construct()           [parallel NN over kNN with running clock/load]
  │   │   ├─ merge_routes()        [greedy with infeas-pair cache + deadline check]
  │   │   ├─ two_opt_route()       [within-route, max_passes=50, deadline-cooperative]
  │   │   ├─ relocate_cross()      [cross-route via kNN, first-improvement on cost]
  │   │   ├─ untangle()            [NEW: 2-opt-star on physical crossings, ±0.5% cost]
  │   │   └─ polish_load_balance() [NEW: cost-neutral move heavy→light route]
  │   │
  │   └─ pyvrp (when [pyvrp] extra installed):
  │       └─ svrptw.solve_auto()   [PyVRP HGS construction → LinUCB bandit refine]
  │
  └─ finalize (ALWAYS runs, even on cancellation)
      ├─ from_svrptw_solution()   [convert svrptw-shaped Solution → public Solution]
      │   ├─ build Visits with arrival/lateness/load
      │   ├─ plan_breaks()        [EU 561 / US HOS state machine — emits BreakEvents]
      │   ├─ compute_all_metrics() [8-metric quality catalog, always reported]
      │   ├─ compose_objective()  [weighted breakdown per ObjectiveConfig]
      │   └─ _audit_fleet_minimum_size()  [greedy first-fit-decreasing bin-pack]
      │
      └─ _reconstruct_geometry()  [network mode — trace each leg via predecessors]
```

## Where you can intercept

Every progress event fires a `ProgressEvent` to your `on_progress`
callback. Raising `StopSolve` from inside the callback cancels the
search **cleanly with best-so-far**:

| Phase | Best-so-far on cancel | Returned status |
|---|---|---|
| `ingest` | none (no work done yet) | `partial` |
| `construct` | none (svrptw call not started) | `partial` |
| `refine` | **constructed sv_sol gets finalized** | `feasible` or `partial` |
| `finalize` | finalize already running; cancel is best-effort | `feasible` or `partial` |

Any other exception raised from `on_progress` is wrapped as
`SolveAborted` and propagated (per SPEC-OPENVRP-08 §1) — never a bare
`Exception` to the caller.

## Deadline cooperation

Every loop that runs >1 ms in the native solver checks `time.monotonic()`
at the top of every pass:

- `construct()` — top of the `while np.any(unserved)` loop
- `merge_routes()` — at the top of `while changed` AND inside the inner
  `for i in range(n)` AND every `_DEADLINE_CHECK_EVERY=64` iterations
  of the inner-inner pair-trial loop
- `two_opt_route()` — at the top of every pass, AND inside the outer
  `for i in range(n-1)` loop. Bounded by `max_passes=50` to defend
  against pathological basins
- `relocate_cross()` — at the top of every pass, at `for ri`, AND at
  `for pos`. Bounded by `max_passes=30`

Hard caps on pass counts mean even a sloppy deadline never lets the
search burn unbounded budget on a single phase.

## What you get out

The `Solution` object always has these populated, regardless of
cancellation:

- `status` ∈ `{feasible, infeasible, partial}`
- `objective_value` == `economics_total.total` within 1e-6
- `vehicles_used` and `vehicles_minimum_found` (D7 audit)
- `quality_report.solution_level` — all 8 catalog metrics
- `diagnostics.operator_pool` — which conditional operators are active
- `diagnostics.fleet_min_report` — `{used, minimum_found, objective_delta_if_minimized}`
- `diagnostics.feasibility_blockers` — if infeasible OR cancelled, names the rule that failed (or the cancellation reason)
- `diagnostics.timings` — `{ingest, construct_refine, finalize}` in seconds
- `diagnostics.snap_report` — per-stop `{stop_id, node_id, snap_meters}` (network mode)
- `diagnostics.triangle_violations` — D16 sampled-check findings

## Hard guarantees (tested in `tests_openvrp/safety/`)

22 tests in `test_termination_safety.py` verify:

1. `budget_seconds=0.001` returns a valid Solution (no crash).
2. `deadline=time.monotonic()-1.0` (already past) returns a valid Solution.
3. `budget_seconds=0.05` solve completes within 2 s wall (finalize + overhead bound).
4. Budget grid `[0.05, 0.1, 0.5, 1.0, 5.0]` — every budget returns valid.
5. `StopSolve` raised at `ingest` / `construct` / `refine` / `finalize` — all return cleanly.
6. `StopSolve` at `refine` returns **best-so-far with routes** (not empty).
7. `RuntimeError` from `on_progress` wrapped as `SolveAborted` per spec.
8. `KeyboardInterrupt` from `on_progress` propagates cleanly (no bad state).
9. Terminated Solution has `fleet_min_report` populated.
10. Terminated Solution has full 8-metric `quality_report`.
11. Terminated Solution round-trips through JSON losslessly.
12. 20 solves in a row don't leak threads.
13. 20 cancelled solves don't leak threads.
14. Same seed + same cancellation point ⇒ identical intermediate Solution.
15. `ProgressEvent.elapsed_seconds` monotonically non-decreasing.

## What this is NOT

- This page is the surface a security/safety reviewer needs to see
  every algorithmic move the solver can make. It is not a tutorial;
  see `quickstart.md` for that.
- It does not document the bandit's internal arm rules when running
  through the `[pyvrp]` extra — those are research-internal (svrptw)
  and the public contract here is "the constructed Solution is
  finalized through `from_svrptw_solution`."
- It does not document the EU 561 / US HOS state machine's regulation
  details — see `constraints.md` and `performance.md`'s honest
  implementation-depth section for that.
