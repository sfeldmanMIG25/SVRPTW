# SPEC-OPENVRP-06 — Network Ingestion & Geometry Reconstruction

**Status:** FROZEN
**Owner role:** OR Engineer
**Depends on:** SPEC-OPENVRP-00 (D14, D15, D16), 01
**Modules:** `openvrp/network/{ingest,geometry}.py` — behind the `[network]` extra;
the core install never imports these.

This makes "give it a network, get back the real route" a tested capability, and is
the one genuinely new piece of solver code relative to the research core (which
computed OD but did not retain per-leg edge paths).

---

## 1. Triangle-inequality precondition (SPEC-OPENVRP-00 D16)

The shortest-path / OD layer assumes the triangle inequality holds across the OD set.
Justification, stated to callers: customers cannot place demand nodes on arbitrary
interior network junctions in a way that would make a "direct" OD entry shorter than a
composed path; snapping to nodes/edges plus shortest-path OD preserves the inequality
for the stop+depot set. This is a **documented precondition**. The ingestion layer
computes a cheap sampled check and, if it finds gross violations (a composed path
materially shorter than a direct entry beyond a tolerance), records a
**diagnostics warning** naming the offending triple — it does not silently "fix" the
matrix and does not fail the solve. Shortest-route quality depends on this holding;
the limits documentation states it plainly.

## 2. Ingestion

- `osm_place` / `osm_bbox`: osmnx drive graph, cached to `cache_dir` keyed by
  (place/bbox, osmnx version, network type). Second run is a cache hit (no network
  I/O). **Graph fetch/load happens before any solve-budget timer starts** — a cold
  graph load must never consume construction budget (carry the research regression
  test for this).
- `graph_file`: `.graphml` native, or `.gpkg`/`.shp` edges (+optional nodes) into a
  networkx DiGraph; CRS required or inferred; reproject to a metric CRS for lengths,
  keep EPSG:4326 for output coordinates.
- Snapping per `SnapConfig` (SPEC-OPENVRP-00 D15): `mode∈{node,edge}`,
  `max_snap_meters` (default 250), `strict`. Beyond `max_snap_meters` ⇒ error if
  `strict` else warning; per-stop snap distance recorded in diagnostics. All snapping
  parameters are caller-overridable.

## 3. OD construction

- Sparse CSR adjacency, edge weight = travel time (length / edge speed; per-class
  speed applied later as a time scale, SPEC-OPENVRP-04 §1).
- `scipy.sparse.csgraph.dijkstra(..., return_predecessors=True)` from each unique
  snapped source. **Retaining predecessors is mandatory** — it makes geometry
  reconstruction a path traceback (O(path length)), not a re-search.
- Emit `time_seconds` and `distance_meters` (n,n) over the stop+depot set plus the
  predecessor structures, cached with the graph. Performance must match or beat the
  research pipeline's OD throughput (the established sparse-Dijkstra speedup is reused,
  not reimplemented).

## 4. Geometry reconstruction (the new capability — shortest route is important)

For every ordered stop pair (a→b) appearing in a solved route:
- Trace predecessors b→a into an ordered node-id path; map to `(lon,lat)` →
  `LegGeometry.polyline`.
- `length_meters` = sum of traversed edge `length` (the graph's own lengths, not a
  geodesic of a simplified line); `duration_seconds` = the OD time the solver
  optimized for that pair (consistency, SPEC-OPENVRP-00 D14).
- The reconstructed leg is the shortest path under the same weights the OD used —
  shortest-route fidelity is therefore guaranteed by construction, contingent on the
  D16 precondition.

D14 enforcement (tested): polyline endpoints == snapped node coords; Σ leg length ==
`Route.distance_meters` ±1e-3; Σ leg duration == Σ OD times the evaluator used. A
broken predecessor chain (disconnected component) ⇒ that leg omitted,
`geometry_status` becomes `absent_failed` or `partial`, the pair listed in
`geometry_failures`; **never a silent straight-line substitution**.

## 5. Acceptance criteria

- G1: osmnx ingest caches; a second call is a cache hit (assert no network call).
- G2: a `.graphml` ingest → solve → geometry with D14 invariants holding.
- G3: cold graph load occurs before the solve timer (regression-tested).
- G4: a far stop with `strict=True` ⇒ validation error naming stop + snap distance;
  with `strict=False` ⇒ warning, solve proceeds; both honor caller-set
  `max_snap_meters`.
- G5: a deliberately disconnected graph ⇒ `geometry_status="partial"` with the right
  failing pair listed, not a straight line.
- G6: geometry reconstruction for a large solved instance is path-traceback time, not
  a re-search (bounded well under the solve budget).
- G7: OD and geometry derive from the same graph object (perturb the graph; both
  change consistently — D14).
- G8: the triangle-inequality sampled check emits a diagnostics warning on a
  hand-built violating fixture and is silent on a clean one; the solve never fails or
  mutates the matrix because of it (D16).

## 6. Non-goals
- No turn-restriction modeling beyond what osmnx provides.
- No map-matching of external traces; no k-shortest-paths (one shortest path per leg,
  matching the optimized cost model).
