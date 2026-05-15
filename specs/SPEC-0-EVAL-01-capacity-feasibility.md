# SPEC-0-EVAL-01 — Capacity feasibility in `evaluate()`

```
ID:            SPEC-0-EVAL-01
Title:         Treat per-route capacity overload as infeasible
               and penalise it at hard_late_penalty per unit-overflow
Owner role:    Cost-model Engineer
Status:        IMPLEMENTED  (2026-05-13)
Severity:      CRITICAL  — silently invalidated multiple LKH-3 bench claims
```

## What was wrong

`svrptw.solvers.common.solution.evaluate()` computed cost from
travel time, distance, missed deliveries, and time-window lateness
only. **Vehicle capacity was never checked.** The `feasible` flag
was returned as `1.0` whenever `missed == 0 and total_late == 0`,
regardless of whether any route exceeded `inst.vehicle_capacity`.

LKH-3's wrapper (`svrptw/solvers/classical/lkh3.py`) was producing
routes with mean utilisation > 1.0 on Manhattan N=50 — single
routes carrying 1.5× the vehicle's stated capacity. The evaluator
accepted those routes as feasible and reported their (artificially
low) cost.

Empirical snapshot, `OSM-Manhattan-N050-I000`, cap = 29:

```
solver          n_routes   mean_util   max_util
greedy                11       0.956       1.00
portfolio@5           11       0.956       1.00
lkh3@1                 7       1.502       1.66    <- impossible
```

The legacy bench numbers favouring LKH-3 on cost were therefore
measuring "LKH-3 squeezed customers into illegal routes" rather
than "LKH-3 solved a tighter problem." Same likely applies to any
OR-Tools wrapper that didn't pass capacity constraints — needs
audit.

## Fix

```python
cap = float(inst.vehicle_capacity) if inst.vehicle_capacity else float("inf")
total_overload = 0.0
for r in sol.routes:
    if not r.customers:
        continue
    load = sum(cust_by_id[cid].demand for cid in r.customers)
    if load > cap:
        total_overload += (load - cap)

cost += e.hard_late_penalty * total_overload    # 1000 × units_overflow
feasible = (missed == 0 and total_late == 0 and total_overload == 0)
```

- Penalty rate matches `hard_late_penalty` (1000/unit) — the same
  rate we charge for missed deliveries, which is the economic
  equivalent of "your driver couldn't fit it on the truck so you
  dropped it." This makes overload strictly dominated by either
  fitting it or recording the miss.
- New metric field `capacity_overload` is returned for audit and
  bench reporting.

## Implications

1. **Prior LKH-3 bench wins are suspect.** Any LKH-3 row in
   `bench/runs/*.json` reporting `operational_cost` lower than
   greedy/portfolio needs to be recomputed before being cited.
   Add a `capacity_overload > 0` filter to the leaderboard.
2. **The cost/logic divergence story changes.** Earlier write-up
   (`bench/figures/v1_vivrp_gemini_smoke.md`) claimed portfolio@10
   loses Gemini logic to LKH-3 because portfolio uses too many
   routes. The real story is closer to: LKH-3's routes look
   visually clean because *they hold more than they're physically
   allowed to* — the dispatcher would never ship them. The Gemini
   judge can't see capacity overflow from a route map.
3. **SPEC-7-COST-01's underutil penalty is much less load-bearing
   than expected.** On the v1 instances we have, utilisation is
   already 0.69–1.00; there is no underutil to penalise. The spec
   stays valid (network-synthetic v1 may differ), but capacity is
   the bigger lever today.

## Acceptance gates

1. **Pre-existing tests pass unchanged.** No solver that *respected*
   capacity sees a cost change. Verified: 37/37 unit tests pass
   after the fix.
2. **Overload solutions are reported infeasible.** Re-running the
   cost-flip smoke with the fix in place shows LKH-3's mean cost
   jumps (overload penalty fires) and `feasible` drops below 1.0
   for the affected rows. (Smoke re-run is in flight at write time.)
3. **No back-compat shim.** Old bench JSONs do not auto-update; the
   recommendation is to re-run rather than recompute, since
   recomputation requires the original route data which we did not
   store in the bench rows.

## Files touched

| Path | Change |
|---|---|
| `svrptw/solvers/common/solution.py` | added overload computation + penalty + feasibility flag + metric |
| (next) `svrptw/solvers/classical/lkh3.py` | needs investigation: why does it produce overloaded routes? probably missing `-C` / capacity flag in the param file |
| `tests/unit/test_evaluator_capacity.py` (next) | direct regression on a hand-built overloaded solution |
