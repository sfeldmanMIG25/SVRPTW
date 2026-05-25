# SPEC-OPENVRP-08 — Validation, Errors & Diagnostics

**Status:** FROZEN
**Owner role:** Spec Author
**Depends on:** SPEC-OPENVRP-01, 02, 03
**Modules:** `openvrp/errors.py`, `openvrp/diagnostics.py`, validation in
`Problem.from_*`

Contract: malformed problems raise structured exceptions at build time; merely *hard*
problems never raise — they return a `Solution` with `status="infeasible"` and
actionable diagnostics.

---

## 1. Exception hierarchy

```
OpenVRPError                       (base; everything subclasses this)
├── ProblemValidationError         (Problem.from_*; carries list[Issue])
├── MissingExtra                   (feature needs an uninstalled extra; has install hint)
├── GeometryUnavailable            (to_geojson on OD-only without points=)
├── SolveAborted                   (on_progress raised something other than StopSolve)
└── StopSolve                      (sentinel the caller raises to cancel; caught internally)
```
No bare `Exception`/`ValueError`/`ImportError` escapes the public API.
`Issue = {severity: "error"|"warning", code: str, message: str, location: str}`
(e.g. `location="stops[id=ACME-12].demand"`). Build succeeds with warning-only
issues; they surface in `Solution.diagnostics.input_warnings`.

## 2. Build-time validation

**Errors (block build):** mode/contents mismatch; demand key outside the fleet
capacity-dimension union; dangling `home_depot_id`/`pickup_of`/`delivery_of`;
non-square/non-finite/negative OD or `index_of` not covering all ids;
`TimeWindow.latest<=earliest`; a stop unservable by every class with
`allow_drops=False`; duplicate ids; an `ObjectiveConfig.quality_terms` key not in the
catalog (error lists the catalog); a `ShiftRule.ruleset` value outside the enum.

**Warnings (build proceeds):** unservable stop with `allow_drops=True`; snap distance
> `max_snap_meters` with `strict=False`; unbounded `count` (engages fleet
minimization — informational, not a problem); auto-closed zone polygon; a
`budget_seconds` low enough that full-stack solves are unreliable at this N (the
research seed-variance budget rule — warn with the recommended minimum); a sampled
triangle-inequality violation in the OD (SPEC-OPENVRP-06 §1).

## 3. Diagnostics (always attached to `Solution`)

```
SolveDiagnostics:
  input_warnings: list[Issue]
  construction_used: "fast" | "pyvrp"
  operator_pool: list[str]                 # the conditionally-registered active arms
  operator_contributions: dict[str,float]  # per-arm objective delta
  search_iterations: int
  budget_seconds: float
  budget_hit: bool                          # deadline truncated the search
  feasibility_blockers: list[str]           # status=infeasible: which precedence rule
                                            # (SPEC-OPENVRP-04 §4) failed + which stops
  fleet_min_report: { used: int, minimum_found: int,
                      objective_delta_if_minimized: float }   # D7 transparency
  quality_report_echo: dict[str,float]      # all catalog metrics (also on Solution)
  snap_report: list[{stop_id,node_id,snap_meters}]
  triangle_violations: list[str]            # sampled D16 check findings (may be empty)
  geometry_status: same as Solution
  timings: dict[str,float]                  # phase -> seconds
```
`feasibility_blockers` is the single most important field: an infeasible problem must
explain *why* in caller terms (e.g. "capacity: stop ACME-12 weight=900 > max class
capacity 500"; "skills: 3 stops require 'hazmat', no class provides it"; "driver-hour:
route exceeds EU561 daily driving even with maximal break insertion"). This is the
difference between a usable library and a black box.

`fleet_min_report` makes SPEC-OPENVRP-00 D7 auditable: it states the minimal
sufficient fleet, the fleet actually used, and — if they differ — the weighted-
objective gain that justified retaining extra vehicles.

## 4. Acceptance criteria

- I1: every §2 rule has a table-driven test asserting the specific subclass + a
  populated `Issue` with correct `location`.
- I2: warning-only problems build; warnings appear in `input_warnings`.
- I3: a well-formed hard-infeasible problem ⇒ `status="infeasible"` +
  `feasibility_blockers` naming the precedence rule and offending ids; **no raise**.
- I4: `MissingExtra` messages contain the exact `pip install` command.
- I5: `operator_pool` reflects conditional registration (cross-checks
  SPEC-OPENVRP-04 E14).
- I6: `budget_hit` true iff the deadline truncated refine; `timings` sums to
  `wall_seconds` ± epsilon.
- I7: `problem_fingerprint` stable for identical input, changes on any field change.
- I8: `fleet_min_report` present and internally consistent on every solve;
  cross-checks SPEC-OPENVRP-02 invariants for D7.
- I9: `triangle_violations` populated on a violating fixture, empty on a clean one;
  solve neither fails nor mutates the matrix (D16).

## 5. Non-goals
- No telemetry/remote reporting (charter: none, ever).
- No auto-repair of invalid problems (report; the caller fixes).
