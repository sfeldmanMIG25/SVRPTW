# Output & serialization

## JSON — lossless

```python
body = sol.model_dump_json()
sol2 = Solution.model_validate_json(body)
assert sol == sol2

# Convenience wrappers:
sol.to_json("solution.json")
sol3 = Solution.from_json("solution.json")
```

Every public type round-trips through JSON. `ODMatrix` serializes as
nested lists; the discriminated `Event` union restores to the correct
concrete subtype including split-break / rest variants.

## GeoJSON — first-class output (RFC 7946)

```python
fc = sol.to_geojson("routes.geojson", include={"events": True})
```

* One `LineString` `Feature` per `Route` (concatenated leg polylines).
* One `Point` `Feature` per customer `Visit`.
* Optional `Point` features for `BreakEvent`, `ZoneEnter/Exit`,
  `RechargeEvent` when `include={"events": True}`.
* One `Point` `Feature` per `DroppedStop` (when applicable).

Determinism: feature order is stable (routes by
`(class_id, ordinal)`, visits by `sequence_index`); two
serializations of one `Solution` are byte-identical.

## OD-only `GeometryUnavailable` (D14)

On an OD-only solve, `to_geojson()` raises `GeometryUnavailable`
unless you supply `points={stop_id: Coordinate}`:

```python
from openvrp.errors import GeometryUnavailable

try:
    sol.to_geojson()
except GeometryUnavailable:
    fc = sol.to_geojson(points={s.id: Coordinate(lon=..., lat=...)
                                for s in stops})
# Approximate-line features carry properties.geometry_approx=true
```

A Euclidean line is **never** emitted as if it were a real network
path. This is the D14 invariant: the cost the solver optimizes and the
geometry it reports come from the same graph.
