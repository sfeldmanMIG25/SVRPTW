# SPEC-OPENVRP-01 — Public Input Schema

**Status:** FROZEN
**Owner role:** Spec Author
**Depends on:** SPEC-OPENVRP-00
**Module:** `openvrp/schema/input.py` — under `mypy --strict`, no `Any`.

Defines every type a caller constructs. pydantic v2. This is the stable public
contract; internal solver structures are derived and not exposed.

Conventions (locked): coordinates are `(lon, lat)` EPSG:4326. OD time in **seconds**,
distance in **meters**. Money in caller-defined cost units. Demand dimensions are
caller-named and unitless. The OD/shortest-path layer assumes the triangle inequality
holds (SPEC-OPENVRP-00 D16) — documented as a precondition, not validated away.

---

## 1. Geometry & timing primitives

**`Coordinate`** — `lon: float [-180,180]`, `lat: float [-90,90]`.

**`TimeWindow`** — `earliest: float >=0`, `latest: float > earliest`,
`hard: bool = True`. `hard=False` ⇒ lateness priced via
`Constraints.soft_tw_penalty_per_sec`.

---

## 2. Demand points & depots

**`Stop`**
```
id: str                              # caller-unique, stable, non-empty
coordinate: Coordinate | None        # required on network path
node_index: int | None               # required on OD path
demand: dict[str, float] = {}        # keys ⊆ fleet capacity-dimension union
service_seconds: float = 0.0
time_windows: list[TimeWindow] = []  # empty=unconstrained; multiple=multi-window
required_skills: list[str] = []
pickup_of: str | None = None         # this stop delivers what stop <id> picked up
delivery_of: str | None = None       # mutually exclusive with pickup_of
allowed_zone_tags: list[str] | None = None   # None=unrestricted
forbidden_zone_tags: list[str] = []
priority: int = 0                    # higher ⇒ costlier to drop
```
Invariants: exactly one of `coordinate`/`node_index` per problem mode; PD references
resolve; demand keys legal; ids unique.

**`Depot`** — `id`, `coordinate|node_index`, `time_window: TimeWindow | None`.
A problem carries ≥1 depot; each vehicle class is bound to a home depot.

---

## 3. Heterogeneous fleet (SPEC-OPENVRP-00 D4, D7)

**`VehicleClass`**
```
id: str
count: int | None = None             # None or -1 ⇒ unbounded (solver minimizes use, D7)
capacity: dict[str, float]           # multi-dimensional; union of keys defines legal demand dims
speed_factor: float = 1.0            # leg time = base_OD_time / speed_factor
fixed_cost: float = 0.0              # cost to use one vehicle of this class at all
cost_per_second: float = 0.0
cost_per_meter: float = 0.0
home_depot_id: str
provides_skills: list[str] = []
allowed_zone_tags: list[str] | None = None   # None=unrestricted; else whitelist
forbidden_zone_tags: list[str] = []
shift: ShiftRule | None = None
max_route_seconds: float | None = None
max_route_meters: float | None = None         # EV range proxy
peak_hour_cost_multiplier: float = 1.0        # applied during Constraints.peak_windows
```
Invariants: `home_depot_id` resolves; capacity-key union defines the legal demand-key
set; a stop requiring skill `s` is serviceable only by classes with `s ∈
provides_skills`; unbounded `count` is permitted and the solver drives the used count
to the minimum consistent with the objective (D7).

**`ShiftRule`** (full rulesets — SPEC-OPENVRP-00 D5)
```
ruleset: Literal["none","eu561","us_hos","custom"] = "none"
shift_start_window: TimeWindow | None = None
# custom / override knobs (used as-is when ruleset="custom",
# and as overrides on top of a named ruleset's defaults otherwise):
max_drive_seconds_before_break: float | None = None
break_seconds: float | None = None
max_continuous_drive_seconds: float | None = None
daily_drive_cap_seconds: float | None = None
daily_onduty_cap_seconds: float | None = None
weekly_drive_cap_seconds: float | None = None
min_daily_rest_seconds: float | None = None
reduced_daily_rest_seconds: float | None = None
max_reduced_daily_rests_between_weekly_rests: int | None = None
split_break_segments: list[float] | None = None      # e.g. EU561 15+30 split
min_weekly_rest_seconds: float | None = None
```
Semantics:
- `ruleset="eu561"` instantiates the **full** EU 561/2006 logic: 4.5 h continuous
  driving then ≥45 min break (or a 15+30 split), 9 h daily driving (10 h twice/week),
  11 h daily rest (reducible to 9 h max 3×/week between weekly rests), 56 h weekly
  driving, 45 h weekly rest (reducible with compensation). Explicit fields override
  individual parameters; the *rule structure* is the regulation's, not a single
  drive-cap simplification.
- `ruleset="us_hos"` instantiates the **full** US FMCSA HOS property-carrier logic:
  11 h driving within a 14 h on-duty window, 30 min break after 8 h driving, 10 h
  off-duty reset, 60/70 h in 7/8 days, 34 h restart.
- `ruleset="custom"` uses only the explicit fields.
- Breaks/rests are inserted by the solver as timeline events
  (SPEC-OPENVRP-02), placed at the latest feasible position; stops are never silently
  relocated to satisfy a rule.

---

## 4. Zones, network, OD

**`Zone`** — `tag: str`, `polygon: list[Coordinate]` (network) or
`node_indices: list[int]` (OD), `kind: Literal["embargo","access"]="embargo"`,
`active_window: TimeWindow | None`.

**`Network`** — `source: Literal["osm_place","osm_bbox","graph_file","none"]`;
`osm_place|osm_bbox|graph_path` set per source; `graph_crs`, `cache_dir`;
`snapping: SnapConfig`.

**`SnapConfig`** (SPEC-OPENVRP-00 D15) —
`mode: Literal["node","edge"]="node"`, `max_snap_meters: float = 250.0`,
`strict: bool = False`. `strict=True` ⇒ a stop beyond `max_snap_meters` is a
validation error; `False` ⇒ a warning.

**`ODMatrix`** — `time_seconds: NDArray[float] (n,n)`,
`distance_meters: NDArray[float] | None`, `index_of: dict[str,int]`,
`asymmetric: bool = True`. Square, finite, non-negative, zero diagonal,
`index_of` covers every stop and depot exactly once. The triangle inequality is
assumed (D16); the schema does not enforce it but the precondition is documented and
a diagnostics check (SPEC-OPENVRP-08) reports gross violations as a warning.

---

## 5. Objective control (SPEC-OPENVRP-00 D8, D9)

**`ObjectiveConfig`** — the caller's top-level lever over what "best" means.
```
operational_cost_weight: float = 1.0     # the default driver of optimization
vehicle_count_weight: float = 0.0        # >0 adds explicit pressure to shrink the fleet
                                         # (fleet minimization is always active via D7;
                                         #  this term lets the caller intensify it)
quality_terms: dict[str, float] = {}     # name -> weight; composite operational-quality
                                         # metrics the caller opts into (see catalog below)
soft_tw_penalty_per_sec: float = 0.0
```
**Composite operational-quality metric catalog** (names usable as `quality_terms`
keys; all are geometric/operational, **no visual/VLM metric exists**):
`route_crossings`, `mean_detour_ratio`, `load_balance_cv`, `load_balance_gini`,
`time_window_slack`, `intra_route_compactness`, `cross_route_overlap`,
`quality_per_route` (the K-fair composite). Each is defined in SPEC-OPENVRP-04 §3
with its sign convention (lower-is-better normalized to a penalty). Default empty ⇒
pure operational cost + active fleet minimization. Setting a weight makes that metric
part of the objective the search optimizes, not merely a reported number.

**`Constraints`** — global toggles & penalties:
```
drop_penalty: float = 1000.0          # scaled by (priority+1); only if allow_drops
allow_drops: bool = False
embargo_soft: bool = False
embargo_penalty: float = 0.0
min_routes: int | None = None         # labor floor (lower bound on vehicles used)
per_route_fixed_cost: float = 0.0
peak_windows: list[TimeWindow] = []
objective: ObjectiveConfig = ObjectiveConfig()
```

**`SolveOptions`** — `budget_seconds: float = 30.0`, `seed: int = 0`,
`construction: Literal["auto","fast","pyvrp"]="auto"`, `threads: int = 4`,
`deadline: float | None`, `on_progress: Callable[[ProgressEvent],None] | None`,
`return_geometry: bool = True`, `verbose: bool = False`.

---

## 6. The problem object

**`Problem`** — `mode: Literal["network","od"]` (validated against contents),
`network: Network`, `od: ODMatrix | None`, `depots: list[Depot]`,
`stops: list[Stop]`, `fleet: list[VehicleClass]`, `zones: list[Zone] = []`,
`constraints: Constraints = Constraints()`, `t0_epoch: float | None`.
Constructors: `Problem.from_network(...)`, `Problem.from_matrix(...)`. Both fully
validate (SPEC-OPENVRP-08) and raise `ProblemValidationError` on malformed input;
neither solves.

---

## 7. Acceptance criteria

- A1: every model pydantic v2, importable from `openvrp`, fully typed,
  `mypy --strict` clean.
- A2: mode/contents mismatches rejected with a stop-and-field-specific error.
- A3: capacity-dimension union enforced against every `Stop.demand` key.
- A4: dangling PD / `home_depot_id` references rejected naming both ends.
- A5: a stop unservable by every class ⇒ error if `allow_drops=False`, droppable
  warning otherwise.
- A6: `ruleset="eu561"` and `="us_hos"` instantiate the full parameter set; explicit
  fields override individual parameters without discarding the rest of the ruleset
  structure (table-driven test, one row per regulation parameter).
- A7: `count=None`/`-1` accepted and flagged as unbounded (engages D7 minimization).
- A8: `SnapConfig` defaults are as in D15 and every field is caller-overridable.
- A9: `ObjectiveConfig.quality_terms` accepts only catalog names; an unknown name is
  a validation error listing the catalog.
- A10: full-object round-trip through JSON for a fixture exercising every optional
  field, including a populated `quality_terms` and an `eu561` shift.
- A11: every public field has a docstring (the schema reference is generated from
  them, SPEC-OPENVRP-09).

## 8. Non-goals
- No solving or network I/O here (`Network` is a description; ingestion is
  SPEC-OPENVRP-06).
- No visual/VLM field anywhere (SPEC-OPENVRP-00 D9).
- No back-compat with research `Settings`/`Economics` names (internal adapter only,
  SPEC-OPENVRP-04 §6).
