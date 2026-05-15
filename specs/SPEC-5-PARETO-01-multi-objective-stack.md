# SPEC-5-PARETO-01 — Multi-objective evaluation stack

```
ID:            SPEC-5-PARETO-01
Title:         NSGA-III front + WFG hypervolume + IGD+ + lexicographic ranking
Owner role:    Bench Engineer
Status:        FROZEN
Inputs:        bench JSON with rows carrying multiple objectives
Outputs:       svrptw.bench.pareto (front extraction, HV, IGD+, R2, lex-rank)
```

## Why

Per Track-3 research (May 2026): a weighted-sum collapse of 5 objectives
(cost, missed, time-violation, cluster-coherence, VLM score) leaks the
ranker's preferences and hides the actual frontier. The standard MOO
stack — non-dominated sorting + hypervolume + IGD+ — is the defensible
way to report which solver "wins."

## Behaviour

```python
# Extract the Pareto-non-dominated subset from a bench row set.
front = nondominated(rows, objectives=["operational_cost", "missed_deliveries",
                                       "wall_clock_seconds", "vivrp_overall"])

# Hypervolume of the front w.r.t. a nadir reference.
hv = hypervolume(front, nadir=auto_nadir(rows), maximize=[False, False, False, True])

# IGD+ vs a reference front (e.g. a long Gurobi run on small instances).
igdp = igd_plus(front, reference=long_run_front)

# Cheap proxy when |front| is huge.
r2 = r2_indicator(front, nadir=auto_nadir(rows))

# Single-decision ranking: lex on hard constraints, then HV-contribution.
ranked = lex_then_hv(front, hard_zero=["missed_deliveries", "tw_late_minutes"])
```

## Implementation

- `pymoo` 0.6+ provides `NonDominatedSorting`, `HV`, `IGDPlus`, `R2` —
  use them. Our wrapper is glue + objective-direction handling
  (minimize cost, maximize VLM score).
- Pareto front extraction is O(N log N) for 2-D, O(N² · D) for D ≥ 3 —
  pymoo's implementation is fine up to ~10k rows.
- `auto_nadir` computes per-objective `max + 10%` from the row set.

## Acceptance

- `python -m svrptw.bench.pareto bench/runs/v1_partial.json --out
  bench/figures/v1_partial_pareto.md` writes a markdown table with
  per-N HV, IGD+ (if reference available), front-size, and the
  lex-HV-ranked top-3 solvers.
- The HV indicator is sensitive: a strictly-dominated row added to the
  set leaves HV unchanged; a non-dominated row strictly increases it.
- Returns the empty front gracefully on bench JSONs with one solver.
- 3 unit tests in `tests/unit/test_pareto.py`.

## Non-goals

- Reference-front generation (the Gurobi long-run is a separate spec).
- Visualisation beyond the existing `pareto_plot.py` (which already
  draws the 2-D cost-vs-time projection).
- Mid-search Pareto archive (this would feed Track-2's bandit reward —
  separate spec).

## Dependencies

- `pymoo` >= 0.6 (already a candidate dependency).
- SPEC-5-VIVRP-01 (the ViVRP scores that become objectives).
