# SPEC-OPENVRP-02 — Public Output Schema

**Status:** FROZEN
**Owner role:** Spec Author
**Depends on:** SPEC-OPENVRP-00, 01
**Module:** `openvrp/schema/output.py` — `mypy --strict`.

The returned `Solution` is the product: inspectable without solver internals,
losslessly serializable, carrying the logical plan, the cost decomposition, the
opted-in quality-metric values, and (network path) the real along-network geometry.

---

## 1. Visit & events

**`Visit`** — `stop_id`, `kind: Literal["depot","customer","pickup","delivery"]`,
`arrival_seconds`, `start_service_seconds`, `departure_seconds`, `wait_seconds`,
`lateness_seconds`, `load_after: dict[str,float]`, `sequence_index: int`.

**`Event`** (discriminated union on `kind`, time-ordered, interleaved with visits) —
base `kind: str`, `at_seconds: float`, `after_visit_index: int | None`. Concrete:
`DepotDepartureEvent`, `DepotReturnEvent`,
`BreakEvent(duration_seconds, rule, segment_index|None)` — covers EU 561 / US HOS
breaks **and** split-break segments and daily/weekly rests (distinguished by `rule`
and a `rest_kind` field: `break|daily_rest|reduced_daily_rest|weekly_rest`),
`ShiftStartEvent(offset_seconds)`, `ZoneEnterEvent(zone_tag)`,
`ZoneExitEvent(zone_tag)`, `RechargeEvent(at_depot_id)`,
`DropEvent(stop_id, reason)`.

The event timeline is the legibility feature: every active constraint becomes
visible. A full EU 561 solution shows continuous-drive breaks, split-break segments,
daily rest, and (for multi-day) weekly rest as distinct events with the regulation
parameter that forced each.

---

## 2. Geometry, economics, quality

**`LegGeometry`** (network path) — `from_stop_id`, `to_stop_id`,
`polyline: list[Coordinate]` (the actual traversed edges), `length_meters`,
`duration_seconds`. Endpoint- and length-consistent with the route per
SPEC-OPENVRP-00 D14.

**`RouteEconomics`** — `total`, `fixed_cost`, `time_cost`, `distance_cost`,
`peak_hour_surcharge`, `break_penalty`, `shift_overrun_penalty`, `embargo_penalty`,
`soft_tw_penalty`, `drop_penalty`, `vehicle_count_cost`,
`quality_penalty: dict[str,float]` (per opted-in quality term),
`other_terms: dict[str,float]`. Invariant: enumerated + Σ(quality_penalty) +
Σ(other_terms) == total within 1e-6.

**`QualityReport`** — every catalog metric (SPEC-OPENVRP-04 §3) computed for the
solution and **always reported** regardless of whether it was weighted into the
objective, so the caller can inspect operational quality even when optimizing pure
cost: `dict[str, float]` at solution level and per route. Weighted ones additionally
appear in `RouteEconomics.quality_penalty`. No visual/VLM metric exists in this report
(SPEC-OPENVRP-00 D9).

---

## 3. Route & solution

**`Route`** — `vehicle_class_id`, `vehicle_ordinal`, `home_depot_id`,
`visits: list[Visit]`, `events: list[Event]`, `geometry: list[LegGeometry]`,
`economics: RouteEconomics`, `quality: dict[str,float]`, `distance_meters`,
`duration_seconds`, `load_peak: dict[str,float]`, `feasible: bool`,
`violations: list[str]`.

**`Solution`**
```
status: Literal["feasible","infeasible","partial"]
                              # heuristic solver: "feasible" = best found, hard-feasible;
                              # no optimality proof is ever claimed
routes: list[Route]
dropped: list[DroppedStop]    # {stop_id, reason, priority}
objective_value: float        # under the caller's ObjectiveConfig
economics_total: RouteEconomics
quality_report: QualityReport
vehicles_used: int
vehicles_minimum_found: int   # smallest fleet the search proved sufficient (D7 surface)
vehicles_available: int | None  # None if any class unbounded
wall_seconds: float
seed: int
construction_used: Literal["fast","pyvrp"]
geometry_status: Literal["present","absent_od_mode","absent_failed","partial"]
geometry_failures: list[str]
solver_version: str
problem_fingerprint: str
diagnostics: SolveDiagnostics   # SPEC-OPENVRP-08
```

Invariants:
- `objective_value == economics_total.total` within 1e-6, evaluated under the
  caller's `ObjectiveConfig`.
- `status="feasible"` ⇒ all hard constraints satisfied on every route (precedence
  order SPEC-OPENVRP-04 §4).
- `vehicles_used == |{(class_id, ordinal)}|` and `vehicles_used >=
  Constraints.min_routes` when set.
- `vehicles_minimum_found <= vehicles_used`; if they differ the search retained a
  larger fleet only because it strictly improved the weighted objective (D7 +
  D8) — and the diagnostics record the cost difference.
- `geometry_status="present"` ⇒ every route's concatenated polyline is continuous and
  its summed leg length equals `Route.distance_meters` within 1e-3 (D14).
- `quality_report` is populated for all catalog metrics whether or not weighted.

---

## 4. Acceptance criteria

- B1: OD-only solve ⇒ `geometry_status="absent_od_mode"`, all economics/quality
  invariants hold, `quality_report` fully populated.
- B2: network solve ⇒ `geometry_status="present"`, D14 continuity & length
  invariants hold per route.
- B3: an EU 561 instance long enough to force a split break and a daily rest shows
  the correct `BreakEvent`s with `rest_kind` set and durations matching the ruleset.
- B4: a soft-TW violation ⇒ `status="feasible"`, offending visit `lateness_seconds>0`,
  `economics.soft_tw_penalty>0`.
- B5: a problem solvable with fewer vehicles than the naive construction uses ⇒
  `vehicles_minimum_found < naive` and `vehicles_used == vehicles_minimum_found`
  unless a weighted quality term justified more (then diagnostics explain the
  trade with numbers).
- B6: setting `objective.quality_terms={"route_crossings":1.0}` changes
  `objective_value` and the chosen routes versus pure cost on the same instance, and
  `quality_report["route_crossings"]` is lower in the weighted run.
- B7: every public type round-trips through JSON; the `Event` union deserializes to
  the correct concrete subtype including split-break/rest variants.
- B8: `economics_total` equals Σ routes + drop penalties within 1e-6 under the
  caller's objective.

## 5. Non-goals
- No rendering (geometry is data; drawing is the caller's or the `[viz]` extra's job).
- No optimality certificates (heuristic solver; `status` never claims "optimal").
- No visual/VLM fields (SPEC-OPENVRP-00 D9).
