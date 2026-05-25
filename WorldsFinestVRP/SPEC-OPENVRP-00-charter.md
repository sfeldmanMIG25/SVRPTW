# SPEC-OPENVRP-00 — Charter & Decisions Register

**Status:** FROZEN
**Owner role:** Orchestrator

This bundle specifies a destination, not a plan. It states what OpenVRP must be and
the invariants that define "correct," and leaves sequencing, sprinting, and effort
estimation entirely to the implementing team. Where a spec says "how," it is because
the *how* is part of the contract (an invariant, an algorithmic guarantee), not a
schedule.

---

## 1. Product statement

**OpenVRP** is a pip-installable, dependency-honest Python library that solves rich
Vehicle Routing Problems with Time Windows and heterogeneous fleets. The caller
supplies a problem (an OD matrix, or a network plus points) and a constraint
configuration; OpenVRP returns structured route objects containing the logical plan
(stop sequence, event timeline, cost decomposition) and, when a network was supplied,
the actual along-network geometry.

**Target user.** An open-source developer or analyst who must handle real-world
routing constraints — non-homogeneous fleets, driver-hour law, zone access, shift
rules, pickup-delivery — without buying a commercial solver and without building a
metaheuristic search engine. They want to spend their effort on their customers' data,
not on graph-search heuristics.

**Competitive position.** Esri/ArcGIS users stay on Esri for its UI and curated
network data; OpenVRP does not compete there. OpenVRP competes by being free, open,
a library (not an application), and by covering constraints that otherwise require a
paid solver. ArcGIS cannot be benchmarked (licensed); the competitive claim is
therefore **timing and solution cost against open solvers (PyVRP, OR-Tools, LKH-3) at
matched wall-clock, plus constraint coverage they lack**.

**Performance mandate.** The shipped solver must be **faster and produce lower-cost
solutions than the research `solve_auto` baseline** at matched wall-clock on the
established OSM-asymmetric evaluation regime. Parity is not the bar; improvement is.
The research codebase's measured advantage (e.g. beating OR-Tools at −43% cost) is the
floor, not the ceiling.

---

## 2. Decisions register (locked)

| # | Decision | Value | Rationale |
|---|---|---|---|
| D1 | Package name | `OpenVRP` (import `openvrp`) | chosen |
| D2 | License | MIT | no copyright interest; simplest permissive; PyVRP (MIT) is a clean optional dep |
| D3 | Network inputs | OSM (osmnx), user graph (GeoPackage/Shapefile/GraphML), pre-built OD matrix | the three real entry points |
| D4 | Heterogeneous fleet attributes | capacity (multi-dim), speed, fixed+variable cost, skills, class-zone access, per-class shift+break rules | the core value proposition |
| D5 | Driver-hour rules | full EU 561 and US HOS rulesets (weekly rest, split breaks, reduced-daily-rest exceptions where the regulation defines them) — not simplified | target users do real compliance |
| D6 | Time-dependent travel | flat configurable peak-hour cost multiplier; a time-varying OD tensor is a reserved schema extension point, not a present feature | engine scope |
| D7 | Fleet sizing | unbounded fleet permitted; the solver **actively minimizes the number of vehicles used** as part of the objective and reports the minimal achievable fleet | stated requirement |
| D8 | Objective control | the objective function is **caller-configurable from the top level**; operational cost is the default, and composite operational-quality metrics are caller-toggleable terms | stated requirement |
| D9 | Quality assessment | composite operational-quality metrics (route geometry, balance, time-window slack, etc.) are computed and exposed for the caller to weight into the objective. **No VLM/visual scoring anywhere in the package or its objective.** | stated requirement |
| D10 | Execution model | synchronous, cancellable (deadline/cancel token), optional progress callback | usable in UI/notebook without freezing |
| D11 | Determinism | given a seed: identical stop sequence; timings exactly derived from it | testable, allows parallel search |
| D12 | Python / typing | Python ≥ 3.10, pydantic v2, full type hints; `mypy --strict` on the public schema modules | modern syntax; strict where it matters |
| D13 | PyVRP dependency | optional accelerator (`[pyvrp]` extra); the default construction path is PyVRP-free and must reach the §1 performance mandate without it | clean dependency tree |
| D14 | Geometry/cost invariant | the cost structure the solver optimizes and the geometry it reports derive from the **same graph**; a Euclidean fallback is never silently substituted | correctness; selling point |
| D15 | Snapping | caller-configurable snapping (mode, max distance, strictness); documented defaults `mode="node"`, `max_snap_meters=250`, `strict=False` | stated requirement |
| D16 | Metric precondition | the shortest-path / OD layer assumes the **triangle inequality holds**. Justification: callers cannot place nodes on arbitrary network junctions in a way that would break shortest-path consistency. This is a stated, documented precondition, not a silent assumption | stated requirement |
| D17 | Serialization | lossless JSON; GeoJSON as a first-class output | adoption lever |

Changing a locked decision is a charter amendment recorded here and propagated along
the dependency graph (§4).

---

## 3. Scope boundary

**In scope (the destination):** CVRPTW; heterogeneous fleet per D4; multi-depot;
full EU 561 and US HOS driver-hour rulesets; shift windows and overrun; embargo/hard
zones (global and per-class, time-windowed); pickup-delivery pairs with precedence;
skills/compatibility; labor floor and active fleet minimization; EV range with
recharge-as-depot; configurable peak-hour cost multiplier; OD-only and
network-backed solving with real along-network geometry; caller-configurable
objective with toggleable composite operational-quality terms; JSON + GeoJSON output;
synchronous cancellable API; the §1 performance mandate over `solve_auto`.

**Explicitly excluded:** any VLM/visual-quality scoring (D9); time-varying OD tensor
(reserved extension point only, D6); GTFS/multimodal routing; stochastic or dynamic
request handling; real-time re-optimization; UI; hosted/managed network datasets;
telemetry. These are not "later" — they are not part of this destination. If pursued,
they are a new charter.

---

## 4. Spec dependency graph

```
SPEC-OPENVRP-00  (charter — this)
  ├── SPEC-OPENVRP-01  Input schema
  │     └── SPEC-OPENVRP-02  Output schema
  │           └── SPEC-OPENVRP-05  Serialization (JSON/GeoJSON)
  ├── SPEC-OPENVRP-03  solve() API & execution model        (needs 01,02,06)
  ├── SPEC-OPENVRP-04  Fleet, constraints & objective control (needs 01)
  ├── SPEC-OPENVRP-06  Network ingestion & geometry          (needs 00:D14,D16; 01)
  ├── SPEC-OPENVRP-07  Packaging & dependency layering        (needs 00:D2,D12,D13)
  ├── SPEC-OPENVRP-08  Validation, errors & diagnostics       (needs 01,02,03)
  └── SPEC-OPENVRP-09  Documentation deliverables             (needs all)
```

This graph constrains *correctness order* (a spec cannot be satisfied before its
dependencies are), not work order. The team decides how to parallelize.

---

## 5. Definition of done (package-level invariants)

1. Core install (no extras) solves an OD-only constraint-rich problem in a few lines;
   the dependency tree contains no osmnx / PyVRP / plotting libraries.
2. `[network]` install solves a city-name problem and returns routes with valid
   along-network geometry.
3. Every public type round-trips: `T == T.model_validate_json(T.model_dump_json())`.
4. GeoJSON output validates against RFC 7946 and loads in standard GIS tooling.
5. The heterogeneous-fleet + objective conformance suite (SPEC-OPENVRP-04) passes
   fully.
6. **Performance mandate met:** at matched wall-clock on the OSM-asymmetric regime,
   the shipped solver is strictly faster AND not worse on cost than research
   `solve_auto`, and retains the established advantage over PyVRP/OR-Tools/LKH-3.
7. Determinism (D11) holds across serial and parallel execution.
8. `mypy --strict` clean on the public schema modules; `mypy` clean elsewhere.
9. The fleet is minimized: for every conformance instance, no returned solution uses
   more vehicles than a solution of equal-or-lower objective the suite can exhibit.
10. Documentation builds with every code example executed; the limits page is
    number-backed by a script, not transcription.
11. No locked decision violated.
