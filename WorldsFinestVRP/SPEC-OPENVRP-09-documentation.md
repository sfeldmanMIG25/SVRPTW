# SPEC-OPENVRP-09 — Documentation Deliverables

**Status:** FROZEN
**Owner role:** Documenter
**Depends on:** all prior specs
**Artifacts:** `docs/` (mkdocs-material or sphinx — Orchestrator picks the tool)

Documentation is a primary product goal. OpenVRP competes by letting a developer *not*
learn graph-search heuristics; the docs are where that promise holds or fails. Every
code block on every page executes in CI against the freshly built package; a
non-executing example fails the build.

---

## 1. Required pages

1. **Quickstart.** Solve a small OD constraint-rich problem in a handful of lines,
   print the route and its event timeline. Core install only. The reader sees a real
   result before any concept is explained.

2. **Network tutorial.** City name → snapped stops → solved routes with real
   along-network geometry → `to_geojson` → load in QGIS. The "open-source, library,
   handles the network too" story, concretely.

3. **Constraints guide.** One runnable section per mechanism in the
   SPEC-OPENVRP-04 §2/§3 catalog. Each shows the config, the solve, and the resulting
   **event timeline** from `Route.events`. The EU 561 section must display a
   continuous-drive break, a split break, a daily rest, and a weekly rest as distinct
   events; the multi-depot section the home-depot binding; the skills section a
   served-vs-dropped contrast. This is the value proposition; it gets the most
   editorial care.

4. **Heterogeneous fleet deep-dive.** A worked mixed-fleet example (small van + large
   truck + refrigerated unit, differing cost/skills/zones/shift rules) that reads the
   `RouteEconomics` decomposition to explain *why* the solver chose its mix and *why*
   it chose its fleet size. Directly addresses the cases commercial tools require
   tuning for.

5. **Objective control.** How to drive optimization from the top level: the default
   pure-cost objective; intensifying fleet minimization with `vehicle_count_weight`;
   folding composite operational-quality metrics in via `quality_terms`; reading the
   always-on `quality_report` even when optimizing pure cost. Explicitly states that
   there is no visual/VLM scoring and that quality here means measurable operational
   geometry/balance/slack.

6. **Schema reference.** Auto-generated from the pydantic docstrings
   (SPEC-OPENVRP-01/02). Never hand-authored (drift). Every field, type, default,
   invariant.

7. **Output & serialization.** `Solution` structure, JSON round-trip, GeoJSON layer
   semantics, loading into QGIS/Leaflet/kepler, the OD-only `GeometryUnavailable`
   behavior and the `points=` approximation flag.

8. **Performance & limits (the credibility page).** Number-backed and honest:
   the measured advantage in timing and cost over PyVRP / OR-Tools / LKH-3 at matched
   wall on the OSM-asymmetric regime, and over the research `solve_auto` baseline
   (the performance mandate); the PyVRP-free path's standing versus the accelerated
   path; the academic-cluster regime being weaker than OSM; peak-hour being correct
   *accounting* not guaranteed savings; per-class distance modeled via zones not
   speed; the **triangle-inequality precondition** (SPEC-OPENVRP-00 D16) stated
   plainly with its justification; why ArcGIS is not benchmarked (licensing) and what
   the competitive claim is instead. Honest limits are a trust signal for an OSS OR
   tool; every number here is produced by a script, never transcribed.

9. **Adding a constraint (contributor guide).** The research meta-recipe as an
   on-ramp: opt-in schema field → gated objective/evaluator term → unit tests →
   conformance row → conditional-operator registration if the axis is search-decided.

10. **FAQ / migration.** "I already have an OD" → `solve_od`. "I use OR-Tools" →
    mapping. "Determinism?" → the seed contract. "Is it really free?" → MIT. "Can I
    change what 'best' means?" → objective control. "Does it minimize trucks?" → yes,
    always; intensify with the weight.

## 2. Cross-cutting requirements

- Every page's code executes in CI against the built wheel.
- Constraints-guide and fleet-deep-dive outputs (event timelines, economics/quality
  tables) are snapshot-tested; drift fails the build.
- `docs/index.md` states the product thesis in three sentences (from
  SPEC-OPENVRP-00 §1) so a reader knows in ten seconds whether the tool is for them.
- README mirrors Quickstart + thesis + limits summary + the install matrix and links
  into the full docs.

## 3. Acceptance criteria

- J1: docs build strictly (warnings are errors); a broken link or non-executing block
  fails it.
- J2: all ten pages exist and their examples run against the fresh wheel in CI.
- J3: the schema reference is generated, not authored (asserted by a marker check).
- J4: the limits page's numbers are produced by `docs/scripts/bench_for_docs.py`,
  including the `solve_auto` performance-mandate comparison, run on a schedule, not
  per-commit, and not transcribed.
- J5: constraints-guide snapshots match current solver output.
- J6: a contributor can follow "Adding a constraint" on a toy term and pass the
  conformance gate (dogfooded once per release).

## 4. Non-goals
- No video series.
- Hosting decisions belong to the Orchestrator; this spec owns content and CI
  execution only.
