# SPEC-0-INST-03 — Heterogeneous fleet, breaks, and renewable charging

```
ID:            SPEC-0-INST-03
Title:         Extend the Instance schema with hetero fleet, breaks, chargers
Owner role:    OR Engineer
Status:        FROZEN (schema + generator); evaluator support is staged
Inputs:        Existing Instance schema
Outputs:       svrptw.io.instance + svrptw.instances_gen.synthetic
```

## Why

Real fleets are heterogeneous (vans, box trucks, tractors), drivers
have statutory rest, and electric / renewable fleets must hit chargers
mid-shift.  Our existing schema collapses fleet to (`num_vehicles`,
`vehicle_capacity`) and ignores all three.  Adding them makes the
benchmark substantially harder and more discriminating.

## New fields (all optional, backward-compatible)

```python
@dataclass
class VehicleSpec:
    vclass: str = "van"                # "van" | "box" | "tractor" | "ev"
    capacity: int = 50
    fuel_capacity: float | None = None # range in miles (EV) or None (ICE = unlimited)
    recharge_rate: float | None = None # miles-per-minute at a charger
    allow_zones: list[str] = ()        # empty = all zones OK

@dataclass
class BreakRegime:
    name: str = "none"                 # "none" | "EU561" | "FMCSA"
    drive_limit_min: int = 270         # max continuous driving before a break
    break_duration_min: int = 45       # mandatory rest length
    rest_node_ids: list[int] = ()      # set of valid rest locations

@dataclass
class Charger:
    node_id: int
    rate: float = 5.0                  # miles recharged per minute
    types: tuple[str, ...] = ("ev",)   # which vehicle classes can use it

# Instance gets three new optional fields:
class Instance:
    vehicles: list[VehicleSpec] | None = None    # if None → homogeneous fleet
    breaks:   BreakRegime | None = None
    chargers: list[Charger] | None = None
```

## Generator additions (svrptw.instances_gen.synthetic)

Three flags wire up the new complexity:

- `--hetero-fleet`: emit `vehicles=[VehicleSpec...]` with mixed van/box/tractor in a fixed ratio.  Capacity drawn from the class's natural range.
- `--breaks {EU561|FMCSA|none}`: emit `breaks=BreakRegime(...)`; default `none`.  Rest areas are drawn from a random subset of customer nodes (5-15% by default) — synthetic, not OSM.
- `--chargers <N>`: emit `chargers=[Charger...]` with N stations at random nodes; EVs make up 30% of the hetero fleet when `--chargers` is active.

## Evaluator staging

The canonical evaluator (`svrptw.solvers.common.solution.evaluate`)
**does not yet enforce breaks or chargers**.  It will treat them as
informational fields in v3 and add cost/feasibility checks in a
follow-up spec (`SPEC-3-EVAL-RICH-01`).  Solvers that don't know about
these fields keep working unchanged.

Acceptance for **this** spec is strictly the schema + generator +
loader; the harder evaluator + solver compliance is a separate spec.

## Acceptance

- `Instance` loads with or without the new fields (back-compat with v1).
- `python -m svrptw.instances_gen.synthetic --N 200 --hetero-fleet --breaks EU561 --chargers 10` emits a parseable Instance with all three fields populated.
- Existing v1 instances still load and produce identical solver results.
- `tests/unit/test_instance_v3.py` verifies round-trip on a synthetic with all three new fields.

## Non-goals

- Cost/feasibility evaluation of breaks and chargers (separate spec).
- Solver awareness of hetero fleet (separate spec; falls under SPEC-3 ops).
- Real charging-graph data (we use synthetic stations).
