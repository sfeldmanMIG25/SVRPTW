"""SPEC-OPENVRP-02 — Public output schema (FROZEN).

The returned ``Solution`` is the product: inspectable without solver
internals, losslessly serializable, carrying the logical plan, the cost
decomposition, the opted-in quality-metric values, and (network path)
the real along-network geometry.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from openvrp.schema.input import Coordinate

# ============================================================
# 1. Visit & events
# ============================================================


class Visit(BaseModel):
    """A single stop arrival on a route."""

    model_config = ConfigDict(extra="forbid")
    stop_id: str = Field(description="Stop id (or depot id for depot kinds).")
    kind: Literal["depot", "customer", "pickup", "delivery"] = Field(
        description="Kind of visit.")
    arrival_seconds: float = Field(description="When the vehicle arrived.")
    start_service_seconds: float = Field(description="When service started (>= arrival; waits roll up here).")
    departure_seconds: float = Field(description="When the vehicle departed.")
    wait_seconds: float = Field(default=0.0, ge=0.0, description="Time spent waiting for TW open.")
    lateness_seconds: float = Field(default=0.0, ge=0.0, description="How late vs the latest TW (0 if on time).")
    load_after: dict[str, float] = Field(default_factory=dict,
        description="Per-dimension load *after* this stop.")
    sequence_index: int = Field(description="0-based position within the route's visit list.")


# ---- Event union (discriminated on `kind`) ----


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    at_seconds: float = Field(description="Event start time (relative).")
    after_visit_index: int | None = Field(default=None,
        description="Index into route.visits the event follows; None ⇒ before visit 0.")


class DepotDepartureEvent(_EventBase):
    kind: Literal["depot_departure"] = "depot_departure"
    depot_id: str


class DepotReturnEvent(_EventBase):
    kind: Literal["depot_return"] = "depot_return"
    depot_id: str


class BreakEvent(_EventBase):
    """Driver break / rest event. Covers EU 561 / US HOS breaks AND
    split-break segments and daily/weekly rests (distinguished by
    ``rule`` and ``rest_kind``).
    """
    kind: Literal["break"] = "break"
    duration_seconds: float = Field(ge=0.0)
    rule: str = Field(description="e.g. 'eu561:continuous_drive_break' or 'us_hos:30min'")
    rest_kind: Literal["break", "split_break_segment", "daily_rest",
                       "reduced_daily_rest", "weekly_rest"] = "break"
    segment_index: int | None = Field(default=None,
        description="For split-break segments only: which segment (0-indexed).")


class ShiftStartEvent(_EventBase):
    kind: Literal["shift_start"] = "shift_start"
    offset_seconds: float = Field(description="Applied shift-start offset within shift_start_window.")


class ZoneEnterEvent(_EventBase):
    kind: Literal["zone_enter"] = "zone_enter"
    zone_tag: str


class ZoneExitEvent(_EventBase):
    kind: Literal["zone_exit"] = "zone_exit"
    zone_tag: str


class RechargeEvent(_EventBase):
    """Legacy single-resource recharge event (EV range / energy_meters).

    Retained for backwards compatibility. New code should prefer
    :class:`ReplenishEvent` which carries a per-resource dict and
    aligns with the generalized multi-depot multi-resource model
    (see :attr:`Depot.resources` and :attr:`VehicleClass.consumes`).
    """
    kind: Literal["recharge"] = "recharge"
    at_depot_id: str
    duration_seconds: float = Field(default=0.0, ge=0.0)
    energy_meters_restored: float = Field(default=0.0, ge=0.0)


class ReplenishEvent(_EventBase):
    """Generalized multi-resource replenishment at a depot mid-route.

    Emitted by the solver when a route visits a depot to refill one or
    more onboard resources. The ``resources_replenished`` dict carries
    per-resource units actually added (bounded by the depot's
    :attr:`Depot.resources` stock at the time of visit and the
    vehicle's :attr:`VehicleClass.onboard_capacity`).
    """
    kind: Literal["replenish"] = "replenish"
    at_depot_id: str
    duration_seconds: float = Field(default=0.0, ge=0.0,
        description="Wall time the vehicle spent at the depot replenishing.")
    resources_replenished: dict[str, float] = Field(default_factory=dict,
        description="Per-resource units added during this stop. Keys "
                    "align with VehicleClass.consumes / Depot.resources.")


class DropEvent(_EventBase):
    kind: Literal["drop"] = "drop"
    stop_id: str
    reason: str


#: Discriminated union of every event type. Round-trips through JSON.
Event = Annotated[
    Union[
        DepotDepartureEvent,
        DepotReturnEvent,
        BreakEvent,
        ShiftStartEvent,
        ZoneEnterEvent,
        ZoneExitEvent,
        RechargeEvent,
        ReplenishEvent,
        DropEvent,
    ],
    Field(discriminator="kind"),
]


# ============================================================
# 2. Geometry, economics, quality
# ============================================================


class LegGeometry(BaseModel):
    """Real along-network geometry for one route leg (network mode only).

    ``polyline`` is the ordered list of network node coordinates
    traversed; endpoint- and length-consistent with the route per
    SPEC-OPENVRP-00 D14.
    """

    model_config = ConfigDict(extra="forbid")
    from_stop_id: str
    to_stop_id: str
    polyline: list[Coordinate]
    length_meters: float = Field(ge=0.0)
    duration_seconds: float = Field(ge=0.0)


class RouteEconomics(BaseModel):
    """Cost decomposition for a single route. Invariant: enumerated terms
    + Σ(quality_penalty) + Σ(other_terms) == ``total`` within 1e-6.
    """

    model_config = ConfigDict(extra="forbid")
    total: float = Field(description="Sum of all enumerated and dict terms.")
    fixed_cost: float = 0.0
    time_cost: float = 0.0
    distance_cost: float = 0.0
    peak_hour_surcharge: float = 0.0
    break_penalty: float = 0.0
    shift_overrun_penalty: float = 0.0
    embargo_penalty: float = 0.0
    soft_tw_penalty: float = 0.0
    drop_penalty: float = 0.0
    vehicle_count_cost: float = 0.0
    quality_penalty: dict[str, float] = Field(default_factory=dict,
        description="Per opted-in quality term name -> applied penalty.")
    other_terms: dict[str, float] = Field(default_factory=dict,
        description="Anything else (e.g. unrecognized solver-internal terms).")


class QualityReport(BaseModel):
    """All catalog quality metrics, computed for the solution and
    **always reported** regardless of objective weighting (SPEC-OPENVRP-04 §3).

    No visual/VLM metric exists in this report (SPEC-OPENVRP-00 D9).
    """

    model_config = ConfigDict(extra="forbid")
    solution_level: dict[str, float] = Field(default_factory=dict,
        description="Catalog metric -> solution-level value.")
    per_route: list[dict[str, float]] = Field(default_factory=list,
        description="Catalog metric -> per-route value (list aligned with Solution.routes).")


class DroppedStop(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stop_id: str
    reason: str
    priority: int = 0


# ============================================================
# 3. Diagnostics
# ============================================================


class SolveDiagnostics(BaseModel):
    """SPEC-OPENVRP-08 — always attached to Solution. The
    ``feasibility_blockers`` field is the single most important when
    status='infeasible': it names the precedence rule that failed.
    """

    model_config = ConfigDict(extra="forbid")
    input_warnings: list[dict[str, Any]] = Field(default_factory=list)
    construction_used: Literal["fast", "pyvrp"] = "fast"
    operator_pool: list[str] = Field(default_factory=list,
        description="Conditionally-registered active operator arms.")
    operator_contributions: dict[str, float] = Field(default_factory=dict,
        description="Per-arm cumulative objective delta over the search.")
    search_iterations: int = 0
    budget_seconds: float = 0.0
    budget_hit: bool = False
    feasibility_blockers: list[str] = Field(default_factory=list)
    fleet_min_report: dict[str, float] = Field(default_factory=dict,
        description="{'used','minimum_found','objective_delta_if_minimized'}.")
    quality_report_echo: dict[str, float] = Field(default_factory=dict)
    snap_report: list[dict[str, Any]] = Field(default_factory=list,
        description="Per-stop {stop_id,node_id,snap_meters} for network mode.")
    triangle_violations: list[str] = Field(default_factory=list,
        description="Sampled-check D16 findings (may be empty).")
    geometry_status: Literal["present", "absent_od_mode", "absent_failed", "partial"] = "absent_od_mode"
    timings: dict[str, float] = Field(default_factory=dict,
        description="Phase name -> seconds.")
    geometry_failures: list[str] = Field(default_factory=list)


class ProgressEvent(BaseModel):
    """Payload of the ``SolveOptions.on_progress`` callback."""

    model_config = ConfigDict(extra="forbid")
    fraction: float = Field(ge=0.0, le=1.0)
    best_objective: float | None = None
    vehicles_used: int | None = None
    vehicles_minimum_found: int | None = None
    elapsed_seconds: float = Field(ge=0.0)
    phase: Literal["ingest", "construct", "refine", "finalize"] = "refine"


# ============================================================
# 4. Route & Solution
# ============================================================


class Route(BaseModel):
    """A single vehicle's route. ``visits`` and ``events`` are time-ordered;
    events are interleaved logically via ``after_visit_index``.
    """

    model_config = ConfigDict(extra="forbid")
    vehicle_class_id: str
    vehicle_ordinal: int = Field(ge=0, description="0-based per-class instance index.")
    home_depot_id: str
    visits: list[Visit] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    geometry: list[LegGeometry] = Field(default_factory=list,
        description="Empty in OD mode; one LegGeometry per ordered visit pair in network mode.")
    economics: RouteEconomics
    quality: dict[str, float] = Field(default_factory=dict,
        description="Per-route values of every catalog quality metric.")
    distance_meters: float = Field(default=0.0, ge=0.0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    load_peak: dict[str, float] = Field(default_factory=dict,
        description="Per-dimension peak load along the route.")
    feasible: bool = True
    violations: list[str] = Field(default_factory=list)


class Solution(BaseModel):
    """A solved problem. Heuristic status only; never claims optimality.

    Invariants (enforced at finalize):
    - ``objective_value`` == ``economics_total.total`` within 1e-6 under the
      caller's ObjectiveConfig.
    - ``status='feasible'`` ⇒ all hard constraints satisfied on every route
      (precedence order per SPEC-OPENVRP-04 §4).
    - ``vehicles_used`` >= ``Constraints.min_routes`` when set.
    - ``vehicles_minimum_found`` <= ``vehicles_used``.
    - In network mode with ``geometry_status='present'``, the concatenated
      polyline is continuous and the summed leg length equals
      ``Route.distance_meters`` within 1e-3.
    """

    model_config = ConfigDict(extra="forbid")
    status: Literal["feasible", "infeasible", "partial"]
    routes: list[Route] = Field(default_factory=list)
    dropped: list[DroppedStop] = Field(default_factory=list)
    objective_value: float
    economics_total: RouteEconomics
    quality_report: QualityReport = Field(default_factory=QualityReport)
    vehicles_used: int = 0
    vehicles_minimum_found: int = 0
    vehicles_available: int | None = None
    wall_seconds: float = 0.0
    seed: int = 0
    construction_used: Literal["fast", "pyvrp"] = "fast"
    geometry_status: Literal["present", "absent_od_mode", "absent_failed", "partial"] = "absent_od_mode"
    geometry_failures: list[str] = Field(default_factory=list)
    solver_version: str = ""
    problem_fingerprint: str = ""
    diagnostics: SolveDiagnostics = Field(default_factory=SolveDiagnostics)

    # ---- Convenience I/O ----

    def to_json(self, path: str | None = None) -> str:
        """Lossless JSON. If ``path`` is None, return the string."""
        body = self.model_dump_json()
        if path is not None:
            from pathlib import Path
            Path(path).write_text(body, encoding="utf-8")
        return body

    @classmethod
    def from_json(cls, path: str) -> Solution:
        from pathlib import Path
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_geojson(self, path: str | None = None, *,
                   include: dict[str, bool] | None = None,
                   points: dict[str, Coordinate] | None = None) -> dict[str, Any]:
        """Emit RFC 7946 FeatureCollection. See SPEC-OPENVRP-05.

        ``include={'events': True}`` opts in event Points
        (BreakEvent, ZoneEnter/Exit, RechargeEvent).

        On an OD-only solve, raises ``GeometryUnavailable`` unless
        ``points={stop_id: Coordinate}`` is supplied — in which case the
        emitted lines are flagged with ``properties.geometry_approx=true``.
        """
        from openvrp.io.serialize import solution_to_geojson
        return solution_to_geojson(self, path=path, include=include or {}, points=points)


__all__ = [
    "Visit",
    "Event", "BreakEvent", "DepotDepartureEvent", "DepotReturnEvent",
    "ShiftStartEvent", "ZoneEnterEvent", "ZoneExitEvent", "RechargeEvent",
    "ReplenishEvent", "DropEvent",
    "LegGeometry", "RouteEconomics", "QualityReport", "DroppedStop",
    "SolveDiagnostics", "ProgressEvent", "Route", "Solution",
]
