# Constraints guide

One runnable section per mechanism in the SPEC-OPENVRP-04 §2/§3
catalog. Each shows the config, the solve, and the resulting event
timeline from `Route.events`.

## Heterogeneous fleet — capacity + skills + cost

```python
from openvrp import VehicleClass, Stop, Depot, TimeWindow

fleet = [
    VehicleClass(id="van", capacity={"weight": 50.0, "volume": 20.0},
                 home_depot_id="W", cost_per_second=0.005,
                 cost_per_meter=0.001, fixed_cost=20.0),
    VehicleClass(id="refrigerated_truck", capacity={"weight": 200.0,
                                                     "volume": 80.0},
                 home_depot_id="W", cost_per_second=0.012,
                 cost_per_meter=0.0015, fixed_cost=80.0,
                 provides_skills=["cold_chain", "hazmat"]),
]
# A stop needing cold-chain can ONLY be served by the truck:
cold_stop = Stop(id="grocery", coordinate=..., demand={"weight": 100.0,
                                                       "volume": 40.0},
                 required_skills=["cold_chain"])
```

## EU 561 driver-hour rules

```python
from openvrp import VehicleClass, ShiftRule

fleet = [VehicleClass(
    id="long_haul",
    capacity={"weight": 500.0},
    home_depot_id="depot",
    shift=ShiftRule(ruleset="eu561"),     # 4.5h continuous drive then
                                          # 45-min break (or 15+30 split),
                                          # 9h daily drive, 11h daily rest,
                                          # 45h weekly rest.
)]
```

The returned `Route.events` carries each break as a `BreakEvent` with
`rest_kind` ∈ `{break, split_break_segment, daily_rest,
reduced_daily_rest, weekly_rest}` and a `rule` field naming the
regulation parameter that forced it.

### What's tested vs schema-locked-only

* **Schema** (locked, JSON round-trippable): the full `ShiftRule`
  parameter set and every `BreakEvent.rest_kind` variant.
* **State machine** (validated at the segment-walker layer in
  `tests_openvrp/unit/test_driver_hours_state_machine.py`):
  continuous-drive split-break, daily-cap with reduced-daily-rest
  3×/wk exception, weekly-cap reset of the reduced quota.
* **Conformance suite at `solve()`** (today): single-shift routes
  exercise the continuous-drive break correctly. Multi-day routes
  (where daily-rest and weekly-rest accrual become observable at the
  `solve()` boundary) are **not** produced by the current native
  solver because every route is bounded by `Depot.time_window` or
  the default 24h depot window. This is a tracked completion item
  against the locked schema, not a schema gap. See the limits page.

## US HOS rules

```python
fleet = [VehicleClass(
    id="us_property_carrier",
    capacity={"weight": 200.0},
    home_depot_id="hub",
    shift=ShiftRule(ruleset="us_hos"),    # 11h drive in 14h on-duty,
                                          # 30-min break after 8h,
                                          # 10h off-duty reset,
                                          # 60/70h in 7/8d, 34h restart.
)]
```

## Multi-depot

```python
depots = [Depot(id="hub_a", coordinate=Coordinate(lon=-74.0, lat=40.7)),
          Depot(id="hub_b", coordinate=Coordinate(lon=-73.9, lat=40.8))]
fleet = [VehicleClass(id="van_a", capacity={"w": 50.0}, home_depot_id="hub_a"),
         VehicleClass(id="van_b", capacity={"w": 50.0}, home_depot_id="hub_b")]
```

Each route starts and ends at its class's home depot; the diagnostics
expose `depot_shift` in `operator_pool` when ≥2 depots are present.

## Pickup-delivery pairs

```python
stops = [
    Stop(id="warehouse_A", ..., pickup_of="customer_X"),    # picks up
    Stop(id="customer_X", ..., delivery_of="warehouse_A"),  # delivers it
]
```

The solver keeps the pickup and delivery on the same route with pickup
strictly before delivery; the bandit's `pd_swap` operator is registered
conditionally when any PD pair exists.

## Embargo zones (time-windowed)

```python
from openvrp import Zone, TimeWindow

zones = [Zone(tag="downtown_closed", polygon=[Coordinate(...)],
              kind="embargo",
              active_window=TimeWindow(earliest=0, latest=7 * 3600))]
constraints = Constraints(embargo_soft=True, embargo_penalty=100.0)
```

`embargo_soft=False` makes the zone hard (route may not enter during
the active window). `True` prices each violation at `embargo_penalty`.

## EV range

```python
fleet = [VehicleClass(id="ev_van", capacity={"w": 50.0},
                      home_depot_id="depot",
                      max_route_meters=120_000.0)]  # 120km range
```

Routes that exceed the cap are flagged infeasible; the
`recharge_insert` operator is registered conditionally when any class
has `max_route_meters` set.
