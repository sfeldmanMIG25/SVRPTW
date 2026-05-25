"""SPEC-OPENVRP-01 — Public input schema (FROZEN).

Every type a caller constructs to describe a problem. pydantic v2.

Conventions (locked, see SPEC-OPENVRP-00):
- Coordinates are ``(lon, lat)`` EPSG:4326.
- OD time in **seconds**, distance in **meters**.
- Money in caller-defined cost units.
- Demand dimensions are caller-named and unitless.
- The OD/shortest-path layer assumes the triangle inequality holds (D16)
  — documented as a precondition, not validated away.
"""
from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ============================================================
# 1. Geometry & timing primitives
# ============================================================


class Coordinate(BaseModel):
    """A geographic point in EPSG:4326. ``lon`` first per GeoJSON/RFC 7946."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    lon: float = Field(ge=-180.0, le=180.0, description="Longitude in degrees, -180..180.")
    lat: float = Field(ge=-90.0, le=90.0, description="Latitude in degrees, -90..90.")

    def as_tuple(self) -> tuple[float, float]:
        """Return ``(lon, lat)`` — GeoJSON axis order."""
        return (self.lon, self.lat)


class TimeWindow(BaseModel):
    """A half-open time window in seconds from problem ``t0``.

    Setting ``hard=False`` makes the window soft: lateness past
    ``latest`` is priced via ``Constraints.soft_tw_penalty_per_sec``
    rather than blocking feasibility.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    earliest: float = Field(ge=0.0, description="Earliest start, seconds.")
    latest: float = Field(gt=0.0, description="Latest start, seconds. Must be > earliest.")
    hard: bool = Field(default=True, description="If False, lateness is priced not blocked.")

    @model_validator(mode="after")
    def _check_order(self) -> TimeWindow:
        if self.latest <= self.earliest:
            raise ValueError(
                f"TimeWindow.latest ({self.latest}) must be > earliest ({self.earliest})"
            )
        return self


# ============================================================
# 2. Demand points & depots
# ============================================================


class Stop(BaseModel):
    """A demand point. Exactly one of ``coordinate``/``node_index`` is set
    according to the problem's mode (network vs OD).

    ``demand`` keys are caller-named capacity dimensions; the schema
    validates them against the union of all ``VehicleClass.capacity`` keys.
    """

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, description="Caller-unique, stable, non-empty stop id.")
    coordinate: Coordinate | None = Field(default=None, description="Required on network mode.")
    node_index: int | None = Field(default=None, ge=0, description="Required on OD mode.")
    demand: dict[str, float] = Field(default_factory=dict,
        description="Per-dimension demand. Keys must be subset of fleet capacity dims.")
    service_seconds: float = Field(default=0.0, ge=0.0, description="On-site service duration, s.")
    time_windows: list[TimeWindow] = Field(default_factory=list,
        description="Empty list = unconstrained; multi-element = multi-window.")
    required_skills: list[str] = Field(default_factory=list,
        description="Skills a serving vehicle must provide.")
    pickup_of: str | None = Field(default=None,
        description="If set, this stop is the pickup half of a PD pair (the matching delivery has delivery_of=this.id).")
    delivery_of: str | None = Field(default=None,
        description="If set, this stop delivers what stop <id> picked up. Mutually exclusive with pickup_of.")
    allowed_zone_tags: list[str] | None = Field(default=None,
        description="None = unrestricted. Otherwise the stop may only be served by classes/zones tagged.")
    forbidden_zone_tags: list[str] = Field(default_factory=list,
        description="Stop is unservable from any class entering a zone with these tags.")
    priority: int = Field(default=0, description="Higher = costlier to drop (drop_penalty scales by priority+1).")

    @model_validator(mode="after")
    def _check_pd_exclusive(self) -> Stop:
        if self.pickup_of is not None and self.delivery_of is not None:
            raise ValueError(
                f"Stop {self.id!r}: pickup_of and delivery_of are mutually exclusive."
            )
        if self.pickup_of == self.id or self.delivery_of == self.id:
            raise ValueError(f"Stop {self.id!r}: cannot reference itself in pickup_of/delivery_of.")
        return self


class Depot(BaseModel):
    """A depot. Vehicle classes are bound to a home depot via ``home_depot_id``.

    The optional ``time_window`` constrains when a vehicle can depart/return.
    Problems must carry at least one depot.
    """

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, description="Caller-unique, stable depot id.")
    coordinate: Coordinate | None = Field(default=None, description="Required on network mode.")
    node_index: int | None = Field(default=None, ge=0, description="Required on OD mode.")
    time_window: TimeWindow | None = Field(default=None, description="Depot open/close window.")


# ============================================================
# 3. Heterogeneous fleet (SPEC-OPENVRP-00 D4, D7)
# ============================================================


class ShiftRule(BaseModel):
    """Driver-hour rule set. ``ruleset="eu561"`` and ``"us_hos"`` instantiate
    the *full* regulatory structure (multi-day rest, split breaks, daily caps).
    Explicit fields override individual parameters without collapsing the
    rest of the ruleset.
    """

    model_config = ConfigDict(extra="forbid")
    ruleset: Literal["none", "eu561", "us_hos", "custom"] = Field(
        default="none", description="Named regulatory ruleset.")
    shift_start_window: TimeWindow | None = Field(default=None,
        description="When the driver may begin the shift.")
    # Override knobs (used as-is for ruleset="custom"; as overrides otherwise)
    max_drive_seconds_before_break: float | None = Field(default=None, ge=0.0,
        description="Continuous-drive cap before a break is mandatory.")
    break_seconds: float | None = Field(default=None, ge=0.0,
        description="Minimum break duration after continuous-drive cap.")
    max_continuous_drive_seconds: float | None = Field(default=None, ge=0.0,
        description="Hard cap on continuous driving without a 30+ min interruption.")
    daily_drive_cap_seconds: float | None = Field(default=None, ge=0.0,
        description="Total drive seconds per 24h day.")
    daily_onduty_cap_seconds: float | None = Field(default=None, ge=0.0,
        description="Total on-duty seconds (drive + service + wait) per 24h day.")
    weekly_drive_cap_seconds: float | None = Field(default=None, ge=0.0,
        description="Total drive seconds per 7-day week.")
    min_daily_rest_seconds: float | None = Field(default=None, ge=0.0,
        description="Minimum daily rest between shifts.")
    reduced_daily_rest_seconds: float | None = Field(default=None, ge=0.0,
        description="Reduced daily rest (EU 561: 9h, up to 3x per week).")
    max_reduced_daily_rests_between_weekly_rests: int | None = Field(default=None, ge=0,
        description="EU 561: limit on reduced rests per week.")
    split_break_segments: list[float] | None = Field(default=None,
        description="EU 561 split-break: e.g. [15*60, 30*60] (15 min then 30 min).")
    min_weekly_rest_seconds: float | None = Field(default=None, ge=0.0,
        description="Minimum weekly rest.")


class VehicleClass(BaseModel):
    """A template for physical vehicles. ``count=None`` (or ``-1``) makes the
    fleet unbounded; the solver actively minimizes the count used (D7).
    """

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, description="Caller-unique class id.")
    count: int | None = Field(default=None,
        description="None or -1 ⇒ unbounded; solver minimizes used count.")
    capacity: dict[str, float] = Field(default_factory=dict,
        description="Multi-dimensional capacity. Keys form the legal demand-dim union.")
    speed_factor: float = Field(default=1.0, gt=0.0,
        description="Leg time = base_OD_time / speed_factor.")
    fixed_cost: float = Field(default=0.0, ge=0.0,
        description="Cost to use one vehicle of this class at all.")
    cost_per_second: float = Field(default=0.0, ge=0.0,
        description="Marginal cost per second of route duration.")
    cost_per_meter: float = Field(default=0.0, ge=0.0,
        description="Marginal cost per meter of route distance.")
    home_depot_id: str = Field(min_length=1, description="Which depot this class starts/ends at.")
    provides_skills: list[str] = Field(default_factory=list,
        description="Skills the class can satisfy.")
    allowed_zone_tags: list[str] | None = Field(default=None,
        description="None = unrestricted. Otherwise whitelist of zone tags this class may enter.")
    forbidden_zone_tags: list[str] = Field(default_factory=list,
        description="Zone tags this class may not enter.")
    shift: ShiftRule | None = Field(default=None,
        description="Driver-hour rules for this class.")
    max_route_seconds: float | None = Field(default=None, gt=0.0,
        description="Hard cap on per-route depot-to-depot duration.")
    max_route_meters: float | None = Field(default=None, gt=0.0,
        description="EV range proxy; total distance cap.")
    peak_hour_cost_multiplier: float = Field(default=1.0, ge=0.0,
        description="Cost multiplier during Constraints.peak_windows.")

    @field_validator("count", mode="after")
    @classmethod
    def _normalize_count(cls, v: int | None) -> int | None:
        # -1 is the documented sentinel for unbounded; normalize to None internally.
        if v is not None and v < 0:
            return None
        return v


# ============================================================
# 4. Zones, network, OD
# ============================================================


class Zone(BaseModel):
    """A zone for access/embargo. Either ``polygon`` (network mode) or
    ``node_indices`` (OD mode) is set."""

    model_config = ConfigDict(extra="forbid")
    tag: str = Field(min_length=1, description="Caller-defined tag.")
    polygon: list[Coordinate] = Field(default_factory=list,
        description="Polygon vertices (network mode). Empty if OD mode.")
    node_indices: list[int] = Field(default_factory=list,
        description="Node indices inside zone (OD mode). Empty if network mode.")
    kind: Literal["embargo", "access"] = Field(default="embargo",
        description="embargo = forbidden during active_window; access = whitelist required.")
    active_window: TimeWindow | None = Field(default=None,
        description="When the zone restriction is active. None = always.")


class SnapConfig(BaseModel):
    """Caller-controlled snapping policy. Documented defaults per SPEC-OPENVRP-00 D15."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["node", "edge"] = Field(default="node",
        description="Snap each stop to nearest network node or edge midpoint.")
    max_snap_meters: float = Field(default=250.0, gt=0.0,
        description="Beyond this snap distance, strict=True raises and strict=False warns.")
    strict: bool = Field(default=False,
        description="If True, distance>max_snap_meters is a validation error.")


class Network(BaseModel):
    """Description of how to obtain the road network. Ingestion is a
    separate concern under ``openvrp[network]``."""

    model_config = ConfigDict(extra="forbid")
    source: Literal["osm_place", "osm_bbox", "graph_file", "none"] = Field(
        default="none", description="Where to load the network from.")
    osm_place: str | None = Field(default=None,
        description="Place name for osmnx, e.g. 'Manhattan, New York, USA'.")
    osm_bbox: tuple[float, float, float, float] | None = Field(default=None,
        description="(min_lon, min_lat, max_lon, max_lat) for osmnx.")
    graph_path: str | None = Field(default=None,
        description="Path to a .graphml / .gpkg / .shp file.")
    graph_crs: str | None = Field(default=None,
        description="EPSG code for the input graph; needed for non-OSM files.")
    cache_dir: str | None = Field(default=None,
        description="Where to cache fetched graphs. Default: ~/.cache/openvrp.")
    snapping: SnapConfig = Field(default_factory=SnapConfig)


class ODMatrix(BaseModel):
    """Pre-built origin-destination matrix. Square, finite, non-negative,
    zero diagonal. ``index_of`` covers every stop and depot id exactly once.

    Stored internally as nested lists so the model round-trips through
    JSON; ``as_numpy`` reconstructs ``np.ndarray`` views.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    time_seconds: list[list[float]] = Field(
        description="(n,n) travel time in seconds.")
    distance_meters: list[list[float]] | None = Field(default=None,
        description="(n,n) travel distance in meters. None ⇒ defaults to time*1.0 (warned).")
    index_of: dict[str, int] = Field(
        description="Map: stop_or_depot_id -> matrix index.")
    asymmetric: bool = Field(default=True,
        description="Documentation hint; not enforced (use both halves).")

    @model_validator(mode="after")
    def _validate_matrix(self) -> ODMatrix:
        n = len(self.time_seconds)
        if n == 0:
            raise ValueError("ODMatrix.time_seconds is empty.")
        for i, row in enumerate(self.time_seconds):
            if len(row) != n:
                raise ValueError(
                    f"ODMatrix.time_seconds row {i} has length {len(row)} != n={n}")
        if self.distance_meters is not None:
            if len(self.distance_meters) != n:
                raise ValueError("ODMatrix.distance_meters has different row count.")
            for i, row in enumerate(self.distance_meters):
                if len(row) != n:
                    raise ValueError(
                        f"ODMatrix.distance_meters row {i} has length {len(row)} != n={n}")
        # index_of must cover [0,n) exactly
        idxs = set(self.index_of.values())
        if len(idxs) != n or min(idxs) != 0 or max(idxs) != n - 1:
            raise ValueError(
                f"ODMatrix.index_of must map ids to exactly [0..{n - 1}], got {sorted(idxs)[:5]}...")
        return self

    def as_numpy(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return ``(time, distance)`` numpy arrays (float64)."""
        t = np.asarray(self.time_seconds, dtype=np.float64)
        d = (np.asarray(self.distance_meters, dtype=np.float64)
             if self.distance_meters is not None else None)
        return t, d

    @classmethod
    def from_numpy(cls, time_seconds: np.ndarray, index_of: dict[str, int],
                   *, distance_meters: np.ndarray | None = None,
                   asymmetric: bool = True) -> ODMatrix:
        """Build from numpy arrays."""
        return cls(
            time_seconds=time_seconds.tolist(),
            distance_meters=distance_meters.tolist() if distance_meters is not None else None,
            index_of=dict(index_of),
            asymmetric=asymmetric,
        )


# ============================================================
# 5. Objective control (SPEC-OPENVRP-00 D8, D9)
# ============================================================


#: Names usable as ``ObjectiveConfig.quality_terms`` keys.
#: All are operational/geometric; **no visual/VLM metric exists** (D9).
#: Defined in SPEC-OPENVRP-04 §3; sign conventions in
#: ``openvrp.engine.metrics``.
QUALITY_CATALOG: tuple[str, ...] = (
    "route_crossings",
    "mean_detour_ratio",
    "load_balance_cv",
    "load_balance_gini",
    "time_window_slack",
    "intra_route_compactness",
    "cross_route_overlap",
    "quality_per_route",
)


class ObjectiveConfig(BaseModel):
    """The caller's top-level lever over what "best" means.

    Default: pure operational cost + active fleet minimization (D7).
    Setting any ``quality_terms[k]`` makes metric ``k`` part of the
    objective the search optimizes, not just a reported number.
    """

    model_config = ConfigDict(extra="forbid")
    operational_cost_weight: float = Field(default=1.0, ge=0.0,
        description="Weight on operational cost (the default driver).")
    vehicle_count_weight: float = Field(default=0.0, ge=0.0,
        description=">0 intensifies fleet shrinkage. Fleet minimization is always active.")
    quality_terms: dict[str, float] = Field(default_factory=dict,
        description="catalog_key -> weight. Catalog: see openvrp.schema.QUALITY_CATALOG.")
    soft_tw_penalty_per_sec: float = Field(default=0.0, ge=0.0,
        description="Per-second penalty applied to soft-TW lateness.")

    @field_validator("quality_terms", mode="after")
    @classmethod
    def _check_catalog(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k in v if k not in QUALITY_CATALOG]
        if bad:
            raise ValueError(
                f"ObjectiveConfig.quality_terms has unknown key(s): {bad}. "
                f"Catalog: {QUALITY_CATALOG}"
            )
        return v


class Constraints(BaseModel):
    """Global toggles and penalties applied across the problem."""

    model_config = ConfigDict(extra="forbid")
    drop_penalty: float = Field(default=1000.0, ge=0.0,
        description="Cost to leave a stop unserved (only if allow_drops). Scaled by (priority+1).")
    allow_drops: bool = Field(default=False,
        description="If True, the solver may drop stops at drop_penalty cost.")
    embargo_soft: bool = Field(default=False,
        description="If True, embargo violations are priced not forbidden.")
    embargo_penalty: float = Field(default=0.0, ge=0.0,
        description="Per-violation penalty when embargo_soft=True.")
    min_routes: int | None = Field(default=None, ge=0,
        description="Labor floor: lower bound on vehicles used (None = no floor).")
    per_route_fixed_cost: float = Field(default=0.0, ge=0.0,
        description="Global fixed cost added per active route.")
    peak_windows: list[TimeWindow] = Field(default_factory=list,
        description="Windows during which class.peak_hour_cost_multiplier applies.")
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig,
        description="The caller-controlled objective.")


# ============================================================
# 6. Solve options & Problem
# ============================================================


class SolveOptions(BaseModel):
    """Per-solve knobs."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    budget_seconds: float = Field(default=30.0, gt=0.0,
        description="Wall-clock budget. Combined with deadline; soonest wins.")
    seed: int = Field(default=0, description="RNG seed; determinism contract (D11).")
    construction: Literal["auto", "fast", "pyvrp"] = Field(default="auto",
        description="auto=pyvrp if importable else fast; pyvrp=requires extra; fast=PyVRP-free.")
    threads: int = Field(default=4, ge=1, description="Internal worker thread count.")
    deadline: float | None = Field(default=None,
        description="Absolute time.monotonic() deadline; sooner of (deadline, now+budget) wins.")
    on_progress: Callable[[Any], None] | None = Field(default=None,
        description="Callback invoked per phase and >=1/s during refine. Raise StopSolve to cancel.")
    return_geometry: bool = Field(default=True,
        description="If False, skip the geometry-reconstruction step even on network mode.")
    verbose: bool = Field(default=False, description="Print phase headers to stderr.")


class Problem(BaseModel):
    """A complete problem instance. ``mode`` is validated against contents.

    Construct via ``Problem.from_matrix`` (OD mode) or
    ``Problem.from_network`` (network mode); both fully validate per
    SPEC-OPENVRP-08 and raise ``ProblemValidationError`` on malformed input.
    Neither solves — call ``openvrp.solve(problem, ...)``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    mode: Literal["network", "od"]
    network: Network = Field(default_factory=lambda: Network(source="none"))
    od: ODMatrix | None = Field(default=None,
        description="Required on mode='od'; None on mode='network' (computed at ingest).")
    depots: list[Depot]
    stops: list[Stop]
    fleet: list[VehicleClass]
    zones: list[Zone] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    t0_epoch: float | None = Field(default=None,
        description="Epoch second corresponding to relative time 0. Optional metadata.")

    # ---- Validators ----

    @model_validator(mode="after")
    def _validate_problem(self) -> Problem:
        from openvrp.errors import ProblemValidationError, ValidationReport
        rep = ValidationReport()

        # 1. Depots non-empty
        if not self.depots:
            rep.add_error("problem.no_depots", "Problem has no depots.", "depots")
        # 2. Stops non-empty
        if not self.stops:
            rep.add_error("problem.no_stops", "Problem has no stops.", "stops")
        # 3. Fleet non-empty
        if not self.fleet:
            rep.add_error("problem.no_fleet", "Problem has no vehicle classes.", "fleet")

        # 4. Unique ids per category
        seen_dep: set[str] = set()
        for i, d in enumerate(self.depots):
            if d.id in seen_dep:
                rep.add_error("depot.duplicate_id",
                              f"Duplicate depot id {d.id!r}", f"depots[{i}]")
            seen_dep.add(d.id)
        seen_stop: set[str] = set()
        for i, s in enumerate(self.stops):
            if s.id in seen_stop:
                rep.add_error("stop.duplicate_id",
                              f"Duplicate stop id {s.id!r}", f"stops[{i}]")
            seen_stop.add(s.id)
        seen_cls: set[str] = set()
        for i, c in enumerate(self.fleet):
            if c.id in seen_cls:
                rep.add_error("class.duplicate_id",
                              f"Duplicate class id {c.id!r}", f"fleet[{i}]")
            seen_cls.add(c.id)

        # 5. Mode/contents consistency
        if self.mode == "od":
            if self.od is None:
                rep.add_error("problem.od_missing",
                              "mode='od' requires an ODMatrix in `od`.", "od")
            for i, s in enumerate(self.stops):
                if s.node_index is None:
                    rep.add_error("stop.node_index_required",
                                  f"mode='od' requires every stop to have node_index. Stop {s.id!r}.",
                                  f"stops[{i}=={s.id!r}].node_index")
            for i, d in enumerate(self.depots):
                if d.node_index is None:
                    rep.add_error("depot.node_index_required",
                                  f"mode='od' requires every depot to have node_index. Depot {d.id!r}.",
                                  f"depots[{i}=={d.id!r}].node_index")
            if self.od is not None:
                # index_of must contain every stop+depot id
                missing = []
                for s in self.stops:
                    if s.id not in self.od.index_of:
                        missing.append(s.id)
                for d in self.depots:
                    if d.id not in self.od.index_of:
                        missing.append(d.id)
                if missing:
                    rep.add_error("od.index_of_missing",
                                  f"ODMatrix.index_of is missing ids: {missing[:5]}{'...' if len(missing) > 5 else ''}",
                                  "od.index_of")
        else:  # network
            for i, s in enumerate(self.stops):
                if s.coordinate is None:
                    rep.add_error("stop.coord_required",
                                  f"mode='network' requires every stop to have coordinate. Stop {s.id!r}.",
                                  f"stops[{i}=={s.id!r}].coordinate")
            for i, d in enumerate(self.depots):
                if d.coordinate is None:
                    rep.add_error("depot.coord_required",
                                  f"mode='network' requires every depot to have coordinate. Depot {d.id!r}.",
                                  f"depots[{i}=={d.id!r}].coordinate")

        # 6. Vehicle classes: home_depot_id resolves
        depot_ids = {d.id for d in self.depots}
        for i, c in enumerate(self.fleet):
            if c.home_depot_id not in depot_ids:
                rep.add_error("class.home_depot_unresolved",
                              f"VehicleClass {c.id!r}.home_depot_id={c.home_depot_id!r} not found.",
                              f"fleet[{i}=={c.id!r}].home_depot_id")
            if c.count is None:
                rep.add_warning("class.unbounded",
                                f"VehicleClass {c.id!r}.count=None — fleet is unbounded; solver minimizes use.",
                                f"fleet[{i}=={c.id!r}].count")

        # 7. Capacity-dim union enforced against every stop.demand key
        cap_keys: set[str] = set()
        for c in self.fleet:
            cap_keys.update(c.capacity.keys())
        for i, s in enumerate(self.stops):
            for k in s.demand:
                if k not in cap_keys:
                    rep.add_error("stop.demand_key_unknown",
                                  f"Stop {s.id!r}.demand key {k!r} not in fleet capacity-dim union {sorted(cap_keys)}.",
                                  f"stops[{i}=={s.id!r}].demand")

        # 8. PD pairs: every reference resolves and is mutually consistent
        stop_index: dict[str, Stop] = {s.id: s for s in self.stops}
        for i, s in enumerate(self.stops):
            if s.pickup_of is not None:
                other = stop_index.get(s.pickup_of)
                if other is None:
                    rep.add_error("stop.pd_unresolved",
                                  f"Stop {s.id!r}.pickup_of={s.pickup_of!r} not found.",
                                  f"stops[{i}=={s.id!r}].pickup_of")
                elif other.delivery_of != s.id and other.pickup_of != s.id:
                    # We expect: s = pickup that 'other' takes the delivery of,
                    # which is encoded as other.delivery_of == s.id. If we got
                    # here, the pair is dangling.
                    rep.add_warning("stop.pd_asymmetric",
                                    f"PD pair: {s.id!r}.pickup_of={s.pickup_of!r} but {s.pickup_of!r}.delivery_of != {s.id!r}.",
                                    f"stops[{i}=={s.id!r}].pickup_of")
            if s.delivery_of is not None:
                other = stop_index.get(s.delivery_of)
                if other is None:
                    rep.add_error("stop.pd_unresolved",
                                  f"Stop {s.id!r}.delivery_of={s.delivery_of!r} not found.",
                                  f"stops[{i}=={s.id!r}].delivery_of")

        # 9. Stop unservable by every class
        for i, s in enumerate(self.stops):
            servable = False
            for c in self.fleet:
                # skill match
                missing_skills = [sk for sk in s.required_skills if sk not in c.provides_skills]
                if missing_skills:
                    continue
                # capacity dimensions covered
                bad_demand = [k for k in s.demand
                              if k not in c.capacity or s.demand[k] > c.capacity[k]]
                if bad_demand:
                    continue
                # zone access (D4): if the stop has allowed_zone_tags, the
                # class must either be unrestricted (allowed_zone_tags=None)
                # or share at least one tag with the stop. Forbidden-zone
                # tags on the stop must NOT appear in the class's forbidden
                # set (the class is forbidden from those zones, so it can't
                # reach a stop located in them).
                if s.allowed_zone_tags is not None and c.allowed_zone_tags is not None:
                    if not set(s.allowed_zone_tags).intersection(c.allowed_zone_tags):
                        continue
                if any(t in c.forbidden_zone_tags for t in s.forbidden_zone_tags):
                    continue
                servable = True
                break
            if not servable:
                if self.constraints.allow_drops:
                    rep.add_warning("stop.unservable_droppable",
                                    f"Stop {s.id!r} cannot be served by any class; will be dropped.",
                                    f"stops[{i}=={s.id!r}]")
                else:
                    rep.add_error("stop.unservable",
                                  f"Stop {s.id!r} cannot be served by any class (skills/capacity/zone). Enable Constraints.allow_drops=True or fix the input.",
                                  f"stops[{i}=={s.id!r}]")

        # 10. ObjectiveConfig.quality_terms (catalog already enforced by field validator)

        # 11. Constraints.min_routes ≤ |fleet|/total count (when bounded)
        if self.constraints.min_routes is not None:
            bounded_total = sum(c.count for c in self.fleet if c.count is not None)
            has_unbounded = any(c.count is None for c in self.fleet)
            if not has_unbounded and self.constraints.min_routes > bounded_total:
                rep.add_error("constraints.min_routes_infeasible",
                              f"Constraints.min_routes={self.constraints.min_routes} > total bounded fleet={bounded_total}.",
                              "constraints.min_routes")

        # Raise on any error
        if rep.errors:
            raise ProblemValidationError(rep.issues)

        # Stash warnings; they're picked up by the solve diagnostics.
        # pydantic Model is frozen for `extra='forbid'` but the model_config
        # allows assignment-time fields — store in __pydantic_extra__.
        object.__setattr__(self, "_input_warnings", rep.warnings)
        return self

    # ---- Constructors ----

    @classmethod
    def from_matrix(cls, *, od: ODMatrix, depots: list[Depot], stops: list[Stop],
                    fleet: list[VehicleClass], zones: list[Zone] | None = None,
                    constraints: Constraints | None = None,
                    t0_epoch: float | None = None) -> Problem:
        """Build an OD-mode Problem and run full validation."""
        return cls(
            mode="od",
            network=Network(source="none"),
            od=od,
            depots=depots,
            stops=stops,
            fleet=fleet,
            zones=list(zones or []),
            constraints=constraints or Constraints(),
            t0_epoch=t0_epoch,
        )

    @classmethod
    def from_network(cls, *, network: Network, depots: list[Depot], stops: list[Stop],
                     fleet: list[VehicleClass], zones: list[Zone] | None = None,
                     constraints: Constraints | None = None,
                     t0_epoch: float | None = None) -> Problem:
        """Build a network-mode Problem and run full validation. The OD
        matrix is computed at ingest (SPEC-OPENVRP-06) once the network is
        loaded by ``openvrp.solve``."""
        return cls(
            mode="network",
            network=network,
            od=None,
            depots=depots,
            stops=stops,
            fleet=fleet,
            zones=list(zones or []),
            constraints=constraints or Constraints(),
            t0_epoch=t0_epoch,
        )

    # ---- Convenience accessors ----

    def input_warnings(self) -> list[Any]:
        """Return the validator-collected warnings (may be empty)."""
        return list(getattr(self, "_input_warnings", []))


__all__ = [
    "Coordinate", "TimeWindow", "Stop", "Depot", "VehicleClass", "ShiftRule",
    "Zone", "Network", "SnapConfig", "ODMatrix", "ObjectiveConfig",
    "Constraints", "SolveOptions", "Problem", "QUALITY_CATALOG",
]
