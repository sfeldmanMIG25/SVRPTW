# SPEC-OPENVRP-05 — Serialization (JSON & GeoJSON)

**Status:** FROZEN
**Owner role:** Spec Author
**Depends on:** SPEC-OPENVRP-01, 02
**Module:** `openvrp/io/serialize.py`

GeoJSON output is a first-class deliverable (SPEC-OPENVRP-00 D17): it drops directly
into QGIS, Leaflet, kepler.gl, deck.gl, and Esri products with no conversion, and is
the strongest single adoption lever.

---

## 1. JSON — lossless, canonical

- `Problem` / `Solution` via pydantic v2 `model_dump_json` / `model_validate_json`.
- `ODMatrix` arrays serialize as nested lists plus recorded `dtype`/`shape`; a custom
  (de)serializer restores `NDArray[float]` exactly for float64.
- The discriminated `Event` union serializes with its `kind` and restores to the
  correct concrete subtype, including split-break / rest variants.
- `solution.to_json(path)` / `Solution.from_json(path)` convenience wrappers.
- Round-trip identity is a definition-of-done invariant, tested with a fixture that
  exercises every optional field (populated `quality_terms`, an `eu561` shift, PD
  pairs, multi-depot, zones).

## 2. GeoJSON — network path, geometry + key properties

`solution.to_geojson(path|None, *, include=...) -> dict` ⇒ a `FeatureCollection`,
EPSG:4326, no `crs` member (RFC 7946):

- One `LineString` `Feature` per `Route` = its `LegGeometry` polylines concatenated
  in visit order; `properties` carry the full `RouteEconomics`, the per-route
  `quality` dict, `distance_meters`, `duration_seconds`, vehicle identity,
  `feasible`, `violations`.
- One `Point` `Feature` per customer `Visit`: `stop_id`, `arrival_seconds`,
  `lateness_seconds`, `load_after`, `sequence_index`, owning vehicle.
- Optional event features (`include={"events": True}`, default off): `Point`s for
  `BreakEvent` (with `rest_kind`), `RechargeEvent`, `ZoneEnter/Exit`, carrying the
  event payload — so a map can show *where the driver legally rested*.
- One `Point` `Feature` per `DroppedStop`, `properties.dropped=true` + reason.

Determinism: feature order stable (routes by `(class_id, ordinal)`, visits by
`sequence_index`); two serializations of one `Solution` are byte-identical.

OD-only path: `to_geojson` raises `GeometryUnavailable` (message points to supplying a
`Network`), OR — if `points={stop_id: Coordinate}` is passed — emits `Point` features
and straight `LineString`s flagged `properties.geometry_approx=true`. A Euclidean
line is **never** emitted as if it were a real network path (SPEC-OPENVRP-00 D14).

## 3. Acceptance criteria

- F1: JSON round-trip identity for the full-coverage `Problem` and `Solution`
  fixtures.
- F2: emitted GeoJSON validates against RFC 7946 in CI (schema validator).
- F3: a network route's `LineString` is continuous (each leg's last coord == next
  leg's first within snap tolerance) and its length == `Route.distance_meters`
  ± 1e-3 (D14).
- F4: the GeoJSON loads via `fiona`/`shapely` in CI with routes as polylines and
  stops/events as attributed points.
- F5: `to_geojson` on an OD-only solution raises `GeometryUnavailable` unless
  `points=` is given, in which case approximate lines are explicitly flagged.
- F6: feature ordering deterministic; double-serialize byte-identical.
- F7: event-feature inclusion toggles cleanly without altering route/stop features.

## 4. Non-goals
- No shapefile/GeoPackage writer (GeoJSON covers the adoption need).
- No vector tiles, no geometry simplification (raw fidelity).
