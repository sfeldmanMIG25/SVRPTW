# OpenVRP — Specification Bundle

Ten frozen specs defining **OpenVRP**: a destination, not a plan. They state what the
package must be and the invariants that define "correct." Sprinting, sequencing,
effort, and timeline are deliberately absent — the implementing team owns those. Where
a spec says "how," that *how* is a contract (an invariant or an algorithmic
guarantee), never a schedule.

## Read order

1. **SPEC-OPENVRP-00 — Charter & Decisions Register.** Product statement, the 17
   locked decisions, scope boundary, dependency graph, package-level
   definition-of-done. Everything else is downstream of this.
2. **SPEC-OPENVRP-01 — Input Schema.** `Problem`, `Network`, `VehicleClass`,
   `ShiftRule` (full EU 561 / US HOS), `Zone`, `ObjectiveConfig`, `Constraints`,
   `SolveOptions`.
3. **SPEC-OPENVRP-02 — Output Schema.** `Solution`, `Route`, `Visit`, the `Event`
   union (incl. split breaks and weekly rests), `RouteEconomics`, `QualityReport`,
   the fleet-minimization surface.
4. **SPEC-OPENVRP-04 — Fleet, Constraints & Objective Control.** The value
   proposition: heterogeneous fleet, full driver-hour rulesets, the operational-
   quality metric catalog, caller-controlled objective, active fleet minimization,
   the non-optional conditional-operator-registration guard, the conformance suite.
5. **SPEC-OPENVRP-05 — Serialization.** Lossless JSON + first-class GeoJSON.
6. **SPEC-OPENVRP-06 — Network & Geometry.** Ingestion, the new along-network
   geometry capability, and the triangle-inequality precondition.
7. **SPEC-OPENVRP-03 — solve() API.** Single entry point, cancellable execution
   model, determinism, and the performance mandate over `solve_auto`.
8. **SPEC-OPENVRP-08 — Validation, Errors & Diagnostics.** Malformed input raises;
   hard-infeasible explains itself; fleet-minimization is auditable.
9. **SPEC-OPENVRP-07 — Packaging.** Core vs `[network]` vs `[pyvrp]` vs `[viz]`,
   MIT, layout.
10. **SPEC-OPENVRP-09 — Documentation.** Ten pages, every example executed in CI.

Correctness order (not work order), from SPEC-OPENVRP-00 §4:
01 → 02 → 04 → 05 → 06 → 03 → 08 → 07 → 09.

## Decisions, as answered

- **Name:** OpenVRP (`import openvrp`).
- **License:** MIT. Vendored research core relicensed MIT (ADR) before release.
- **Driver-hour rules:** full EU 561 and US HOS rulesets — weekly rest, split
  breaks, reduced-daily-rest exceptions — not a simplified single drive-cap.
- **Fleet:** unbounded fleets permitted; the solver *actively minimizes vehicles
  used* and reports the minimal sufficient count (`vehicles_minimum_found`,
  `fleet_min_report`). The optional `vehicle_count_weight` lets the caller intensify
  that pressure deliberately.
- **Snapping:** fully caller-configurable; documented defaults `mode="node"`,
  `max_snap_meters=250`, `strict=False`.
- **Performance:** the shipped solver must be **faster and not worse on cost than
  research `solve_auto`** at matched wall on the OSM regime, and retain the advantage
  over PyVRP/OR-Tools/LKH-3. Parity is not the bar; this is a standing gate
  (SPEC-OPENVRP-03 §4).
- **VLM/visual quality:** excluded entirely — no such metric exists anywhere in the
  package or its objective.
- **Operational quality:** a catalog of composite operational metrics (route
  geometry, balance, time-window slack, K-fair quality) is always computed and
  reported, and any subset is caller-toggleable into the optimized objective from the
  top level via `ObjectiveConfig.quality_terms`.
- **Shortest route / triangle inequality:** reconstructed legs are shortest paths
  under the optimized weights; the triangle inequality is a stated, documented
  precondition (callers cannot place nodes on arbitrary interior junctions in a way
  that breaks it), with a non-failing diagnostics check for gross violations.

## What these specs deliberately exclude

VLM/visual scoring (entirely); time-varying OD tensor (reserved schema extension
point only); GTFS/multimodal; stochastic/dynamic requests; real-time
re-optimization; UI; hosted network datasets; telemetry. These are not "later" —
they are outside this destination. Pursuing any is a new charter.

## Relationship to the research codebase

The research search core (adaptive operator bandit, evaluator, construction variants)
is **vendored** and reached only through `engine/_adapter.py`. Two hard-won research
findings are promoted from "fixes" to **contractual correctness requirements**:
conditional operator registration (SPEC-OPENVRP-04 §5, guarded) and graph-load-
before-timer (SPEC-OPENVRP-06 §2, regression-tested). The one genuinely new piece of
solver code is along-network geometry reconstruction via retained Dijkstra
predecessors (SPEC-OPENVRP-06 §4); everything else is contract, packaging, and
documentation around proven internals — plus meeting the standing mandate to be
faster and better than `solve_auto`.
