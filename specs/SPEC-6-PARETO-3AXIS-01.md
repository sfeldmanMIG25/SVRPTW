# SPEC-6-PARETO-3AXIS-01 — Three-axis (cost, time, logic) Pareto reporting

```
ID:            SPEC-6-PARETO-3AXIS-01
Title:         Extend bench/pareto.py to a (cost, time, logic) frontier;
               headline = fraction of instances we dominate PyVRP on all 3
Owner role:    Bench Engineer
Status:        FROZEN
Inputs:        bench rows w/ operational_cost, wall_clock_seconds, logic_score
Outputs:       svrptw.bench.pareto3 + bench/figures/v1_pareto_3axis.{md,png}
Depends on:    SPEC-5-PARETO-01 (existing 2-axis), SPEC-6-LOGIC-01 (logic score)
```

## Why

Our existing `svrptw.bench.pareto` reports (cost, time) and a VLM
hypervolume on a 4-objective stack — useful for internal triage but
not a defensible headline. The dispatcher-acceptance reframe demands
a single number we can put on the front page:

> **Fraction of v1 instances where our best solver Pareto-dominates
> PyVRP under (operational_cost, wall_clock_seconds, logic_score).**

This is honest about three things at once:
- cost (the academic axis we will lose at by a small margin),
- time (where we win, because PyVRP@30 is 30 s and we are sub-second),
- logic (where we either win or have no signal — both fine).

## Behaviour

```python
from svrptw.bench.pareto3 import dominance_report

report = dominance_report(
    rows=load_bench("bench/baselines/v1.json"),
    challenger="portfolio@10",
    baseline="pyvrp@30",
    axes=("operational_cost", "wall_clock_seconds", "logic_score"),
    minimise=(True, True, False),
)

# report.headline:  float in [0, 1]  e.g. 0.62
# report.per_n:     dict[N -> fraction]
# report.detail:    DataFrame  one row per instance with the 3 deltas
# report.ties:      int    instances where neither strictly dominates
```

Dominance rule (strict Pareto):

> Challenger dominates baseline on instance `I` iff challenger is
> better-or-equal on every axis and strictly better on at least one,
> using a 1 % numerical tolerance per axis to avoid floating-point ties.

Logic-score axis special handling:

- If `logic_ensemble.authoritative == False` for *either* solution on
  instance `I`, the logic axis is dropped for that instance and
  dominance is computed on the remaining two axes. The detail row
  flags this as `logic_axis_dropped=True`.
- This is what "carry uncertainty" means at the reporting layer.

## Outputs

- `bench/figures/v1_pareto_3axis.md` — table per N-bin:

  ```
  | N-bin  | challenger        | baseline | dominates | ties | logic-dropped |
  | 20–60  | portfolio@10      | pyvrp@30 |    0.71   | 0.18 |     0.04      |
  | 80–120 | portfolio@10      | pyvrp@30 |    0.55   | 0.25 |     0.10      |
  | 160–220| pomo_v1+portfolio | pyvrp@30 |    0.40   | 0.30 |     0.12      |
  ```

- `bench/figures/v1_pareto_3axis.png` — 3-D scatter (or three 2-D
  projections, decided at impl time) coloured by dominance class.

- A one-line headline written to `bench/figures/HEADLINE.md`:

  > _Our solver Pareto-dominates PyVRP under (cost, time, logic) on
  > **62 %** of v1 instances; +18 % wins on time alone, –3 % losses
  > on cost alone._

## Acceptance gates

1. `dominance_report()` reproduces the existing 2-axis (cost, time)
   numbers when `axes=("operational_cost", "wall_clock_seconds")`.
2. When `logic_score` column is entirely missing, the function falls
   back to 2-axis without crashing; logged WARN.
3. Headline number is bit-stable across re-runs (deterministic given
   bench JSON + ensemble checkpoint).
4. Unit test confirms tolerance behaviour at the 1 % boundary.

## Non-goals

- Not changing the existing `svrptw.bench.pareto` API. We add
  `pareto3.py` next to it; the old one stays for back-compat and for
  the 5-objective VLM front.
- Not adding logic to the bench's solver-selection logic. This spec
  is only about *reporting*. The bandit consumes the same score via
  SPEC-6-BANDIT-PLATEAU-01, not via this module.

## Files this spec creates

| Path | Role |
|---|---|
| `svrptw/bench/pareto3.py` | `dominance_report`, 3-axis front extractor |
| `svrptw/bench/headline.py` | one-line headline writer |
| `tests/unit/test_pareto3_dominance.py` | tolerance, axis-drop, 2-axis fallback |
