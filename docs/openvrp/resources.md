# Generalized resource replenishment (multi-depot, multi-resource)

Real fleets carry more than just a single capacity: water, spare
parts, fuel, refrigerant, oxygen tanks, batteries. Each resource has
its own onboard tank and may need to be refilled at depots that stock
it. Different depots stock different subsets — the regional hub has
spare parts, the field depot only has fuel.

OpenVRP models this as a **first-class part of the schema** since
iter-12. The contract is locked; the algorithm support is the
honestly-tracked next step.

## Schema (locked)

### `Depot.resources`

```python
from openvrp import Depot, Coordinate

regional_hub = Depot(
    id="hub_a",
    coordinate=Coordinate(lon=-74.01, lat=40.72),
    resources={
        "fuel_liters": 5000.0,       # 5000 L stocked at this depot
        "spare_parts": None,         # unlimited
        # "refrigerant" absent       — this depot does NOT stock it
    },
)

fuel_only_depot = Depot(
    id="hub_b",
    coordinate=Coordinate(lon=-73.95, lat=40.80),
    resources={"fuel_liters": 1200.0},
)
```

* Value of `None` means **unlimited stock** for that resource.
* A `float` is a **hard daily/session cap** the solver respects.
* A **missing key** means the depot does not stock that resource at all.
* Default empty dict ⇒ no replenishment offered (back-compatible with
  pre-resource workloads).

### `VehicleClass.consumes` + `onboard_capacity`

```python
from openvrp import VehicleClass

service_truck = VehicleClass(
    id="service",
    count=None,
    capacity={"weight": 1000.0},
    home_depot_id="hub_a",
    consumes={
        "fuel_liters": 0.08,        # 0.08 L per meter (= 8 L / 100 m)
        "spare_parts": 0.0,         # carried, never depleted (load-only)
    },
    onboard_capacity={
        "fuel_liters": 60.0,        # 60 L tank
        "spare_parts": 50.0,        # 50 parts onboard
    },
)
```

* `consumes[res]` = resource depletion rate per meter of route distance.
* `onboard_capacity[res]` = vehicle's tank/storage for that resource.
* A vehicle starts every route with a full tank and **must replenish
  at a depot stocking that resource** before the tank hits zero.
* Both default to empty dict ⇒ no replenishment needed.

### `ReplenishEvent` (output)

```python
from openvrp import ReplenishEvent

# Emitted by the solver when a route visits a depot to refill:
ReplenishEvent(
    at_seconds=12_600.0,
    after_visit_index=4,
    at_depot_id="hub_b",
    duration_seconds=600.0,
    resources_replenished={"fuel_liters": 45.0},
)
```

Discriminated-union arm `kind="replenish"`, distinct from the legacy
`RechargeEvent(kind="recharge")` which is retained for EV-range
back-compat.

## Validators (active today)

The `Problem` constructor warns or errors when:

| Code | Severity | What it catches |
|---|---|---|
| `class.resource_unstocked` | warning | A class consumes a resource no depot stocks anywhere. |
| `class.onboard_zero` | error | A class consumes a resource but its `onboard_capacity` for it is 0 — the vehicle would deplete immediately. |

## Conditional operator pool

`SolveDiagnostics.operator_pool` includes the
`replenish_insert` operator iff:
- any `VehicleClass.consumes` is non-empty, OR
- any `Depot.resources` is non-empty.

You can verify this on every solve:
```python
sol = solve(problem, options)
assert "replenish_insert" in sol.diagnostics.operator_pool
```

## Implementation depth (honest)

* **Schema** — locked. Every public type round-trips JSON; the
  discriminated `Event` union carries `ReplenishEvent` distinctly
  from the legacy `RechargeEvent`. 13 unit tests in
  `tests_openvrp/unit/test_resources_schema.py`.
* **Validators** — active. The two listed codes above fire on every
  malformed problem.
* **Conditional operator registration** — active. The diagnostics
  field is populated correctly.
* **Native solver `replenish_insert` algorithm** — **NOT yet
  implemented**. The operator name appears in the pool to surface the
  axis to downstream tooling, but the native solver does not currently
  insert depot waypoints mid-route to refill resources. This is the
  single largest tracked completion item for this feature, paired
  with the multi-day routing gap previously disclosed.
* **PyVRP integration via the `[pyvrp]` extra** — PyVRP 0.13.3
  (released 2026-02-16) does not natively support generalized
  multi-resource replenishment; the adapter passes the scalar
  capacity dimension through and ignores `consumes` /
  `onboard_capacity` / `resources`. Users requiring the full feature
  today should either (a) wait for the native solver completion item
  to land, or (b) encode resource-depletion routing as additional
  capacity dimensions plus depot-as-customer (a documented but
  manual workaround).

## What this means for a caller today

```python
# This contract is solid:
problem = Problem.from_matrix(od=od, depots=[hub_a, hub_b],
                              stops=stops, fleet=[service_truck])
sol = solve(problem)

# All of these work:
assert "replenish_insert" in sol.diagnostics.operator_pool
assert sol.depots[0].resources == ...   # round-trips
assert hasattr(ReplenishEvent, "resources_replenished")

# But this does NOT yet hold:
# (sol.routes will not currently emit ReplenishEvent — the native
# solver doesn't yet construct routes with mid-route depot waypoints.)
```

This is the schema-first discipline applied honestly to a new feature:
contract locked first so multi-day routing AND resource replenishment
can be built against a stable shape in any order.

## Related literature

- Schneider, M., Stenger, A., & Goeke, D. (2014). *The Electric
  Vehicle-Routing Problem with Time Windows and Recharging Stations.*
  Transportation Science 48. The single-resource case (EVRP) the
  generalized model strictly extends.
- Hiermann, G., Hartl, R. F., Hjorring, C. A., & Schneider, M. (2019).
  *Routing a mix of conventional, plug-in hybrid, and electric vehicles.*
  Mixed-fleet single-resource extensions.
- The multi-resource generalization (replenishment at depots stocking
  arbitrary resource subsets) does not have a single canonical
  reference; it generalizes EVRP and the satellite-facility VRP. The
  schema here captures it as a first-class shape.
