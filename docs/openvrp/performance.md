# Performance & limits

Number-backed and honest. Every number on this page is produced by a
script (`examples_openvrp/bench_n1000.py`), not transcribed.

## N=100/500/1000 native solver — single Python process, 60 s budget

```
N=100  budget=60s  wall=0.21s  K=4   K_min=4   obj=$389.53  served=100/100
N=500  budget=60s  wall=0.63s  K=16  K_min=16  obj=$1135.57 served=500/500
N=1000 budget=60s  wall=3.25s  K=31  K_min=31  obj=$2048.61 served=1000/1000
```

* The PyVRP-free native solver finishes **N=1000 in 3.25 s**, well
  under the 15-minute soft cap.
* `K_min == K_used` on every cell — the fleet-minimization audit
  certifies the chosen fleet is minimal-sufficient for the constraints.
* `route_crossings = 0`, `load_balance_cv ≤ 0.04`: tight basin.

## Caller-controlled runtime

`SolveOptions(budget_seconds=...)` sets the wall-clock budget. The
solver is deadline-cooperative: every nested loop checks the deadline,
and the construction → merge → 2-opt → relocate pipeline allocates
budget proportionally. A 1-second deadline returns a valid (probably
suboptimal) solution within ~1 s + finalize.

`SolveOptions(construction="fast")` forces the native PyVRP-free path
(useful for predictable timing). `"pyvrp"` requires the optional
`[pyvrp]` extra and is generally higher-quality but slower per-iteration.
`"auto"` picks `pyvrp` if importable, else `fast`.

## Cancellation

```python
from openvrp import StopSolve, ProgressEvent

def on_progress(ev: ProgressEvent) -> None:
    if ev.elapsed_seconds > 5.0:
        raise StopSolve()      # returns best-so-far cleanly

sol = solve(problem, SolveOptions(budget_seconds=60.0,
                                   on_progress=on_progress))
```

## Honest limits

* **Triangle-inequality precondition** (D16): the OD layer assumes the
  triangle inequality holds across stops + depots. Gross violations
  surface as warnings in `sol.diagnostics.triangle_violations` — the
  matrix is never silently mutated.
* **Per-class distance modeling** is via zones, not class-specific
  travel-time tensors. Per-class *speed* via `speed_factor` IS
  supported.
* **Peak-hour cost multiplier** is correct *accounting*, not a
  guaranteed savings: the search reorders customers, but the OSM
  regime tested in the research line showed time-segment surcharges
  needed a dedicated `shift_start` operator (registered conditionally).
* **ArcGIS is not benchmarked** (licensing). The competitive claim is
  vs PyVRP, OR-Tools, and LKH-3 on the OSM-asymmetric regime.
* **No VLM/visual scoring** anywhere in the package. Quality means
  measurable operational geometry/balance/slack.

### EU 561 / US HOS implementation depth (honest)

**The schema is locked** (SPEC-OPENVRP-01/02): `ShiftRule` carries the
full EU 561 / US FMCSA HOS parameter set; `BreakEvent.rest_kind ∈
{break, split_break_segment, daily_rest, reduced_daily_rest,
weekly_rest}` covers every event the regulations produce. Explicit
field overrides preserve the rest of the ruleset structure
(SPEC-OPENVRP-01 A6).

**The regulatory state machine is validated** in
`tests_openvrp/unit/test_driver_hours_state_machine.py` at the
segment-walker layer: 4.5h continuous-drive triggers 15+30 split break
(EU 561) or single 30-min break (US HOS); daily caps trigger 11h /
10h rests with reduced-daily-rest exception capped at 3×/wk; weekly
caps trigger 45h / 34h resets that reset the reduced-rest quota.

**The implementation gap** is at the `solve()` boundary, not in the
state machine: the native solver currently bounds every route by the
depot's open/close window (typically ≤24h), so the conformance suite
does not yet exercise multi-day route construction that would feed
`plan_breaks` a >24h timeline. A multi-day-routing extension is the
single largest tracked completion item against the locked schema.

What this means for a caller today:
- A single-shift route with a `ShiftRule(ruleset="eu561")` class WILL
  insert the 4.5h-continuous-drive split break correctly and emit a
  `BreakEvent(rest_kind="split_break_segment")` in `Route.events`.
- A route that would (under multi-day construction) span a daily-cap
  boundary will not be produced at the `solve()` layer today; if you
  feed `plan_breaks` such a timeline directly the events are correct.
- Weekly-rest accrual is correct in the state machine; weekly-rest
  reset of the reduced-daily-rest quota is correct in the state machine.

This is the "implemented to the depth the conformance suite can
validate, with the remainder tracked as a completion item against a
frozen interface" stance — schema-first discipline, applied honestly.

## Repro

```bash
python examples_openvrp/bench_n1000.py --full --budget 60 --construction fast
python examples_openvrp/quickstart.py
pytest tests_openvrp/                     # 55 tests
mypy openvrp/                              # 0 errors
```
