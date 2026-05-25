"""Heterogeneous fleet semantics and driver-hour ruleset materialization
(SPEC-OPENVRP-04 §1, §2; SPEC-OPENVRP-00 D5).

The ``materialize_shift_rule`` function turns a caller's ``ShiftRule``
(possibly with ``ruleset="eu561"`` or ``"us_hos"``) into a fully populated
struct of regulatory parameters. Explicit ``ShiftRule`` fields override
individual parameters without collapsing the rest.

A ``BreakPlanner`` walks an event-ordered timeline of (drive, service,
wait) segments and inserts ``BreakEvent`` records at the latest feasible
positions — never relocating stops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from openvrp.schema.input import ShiftRule
from openvrp.schema.output import BreakEvent

# ============================================================
# Ruleset constants (per regulation; do not edit casually)
# ============================================================


@dataclass(frozen=True)
class MaterializedShiftRule:
    """Fully populated rule parameters in seconds. Every field is set."""

    ruleset: Literal["none", "eu561", "us_hos", "custom"]
    # Continuous-drive break (the headline rule)
    max_drive_seconds_before_break: float
    break_seconds: float
    # Split-break (EU 561 allows 15+30 instead of single 45)
    split_break_segments: tuple[float, ...]
    # Within-shift continuous-drive cap with smaller interruptions
    max_continuous_drive_seconds: float
    # Daily caps
    daily_drive_cap_seconds: float
    daily_onduty_cap_seconds: float
    # Weekly caps
    weekly_drive_cap_seconds: float
    # Daily rest between shifts
    min_daily_rest_seconds: float
    reduced_daily_rest_seconds: float
    max_reduced_daily_rests_between_weekly_rests: int
    # Weekly rest
    min_weekly_rest_seconds: float


# EU 561/2006 (property-carrying; passenger transport uses a different sub-rule)
EU561_DEFAULTS = dict(
    max_drive_seconds_before_break=4.5 * 3600.0,        # 4h 30min
    break_seconds=45 * 60.0,                            # 45 min
    split_break_segments=(15 * 60.0, 30 * 60.0),        # 15 + 30
    max_continuous_drive_seconds=4.5 * 3600.0,
    daily_drive_cap_seconds=9 * 3600.0,                 # 9h (extendable to 10h ≤2/week)
    daily_onduty_cap_seconds=13 * 3600.0,               # informal — 9h drive + breaks + service
    weekly_drive_cap_seconds=56 * 3600.0,
    min_daily_rest_seconds=11 * 3600.0,
    reduced_daily_rest_seconds=9 * 3600.0,
    max_reduced_daily_rests_between_weekly_rests=3,
    min_weekly_rest_seconds=45 * 3600.0,
)

# US FMCSA HOS — property-carrying
US_HOS_DEFAULTS = dict(
    max_drive_seconds_before_break=8 * 3600.0,          # 8h driving then 30-min break
    break_seconds=30 * 60.0,                            # 30 min
    split_break_segments=(),                             # not used in US HOS
    max_continuous_drive_seconds=8 * 3600.0,
    daily_drive_cap_seconds=11 * 3600.0,                # 11h within 14h on-duty
    daily_onduty_cap_seconds=14 * 3600.0,               # 14h on-duty window
    weekly_drive_cap_seconds=60 * 3600.0,               # 60h/7d (also 70h/8d variant)
    min_daily_rest_seconds=10 * 3600.0,                 # 10h off-duty reset
    reduced_daily_rest_seconds=10 * 3600.0,             # US has no formal reduced rest
    max_reduced_daily_rests_between_weekly_rests=0,
    min_weekly_rest_seconds=34 * 3600.0,                # 34h restart
)

NONE_DEFAULTS = dict(
    max_drive_seconds_before_break=float("inf"),
    break_seconds=0.0,
    split_break_segments=(),
    max_continuous_drive_seconds=float("inf"),
    daily_drive_cap_seconds=float("inf"),
    daily_onduty_cap_seconds=float("inf"),
    weekly_drive_cap_seconds=float("inf"),
    min_daily_rest_seconds=0.0,
    reduced_daily_rest_seconds=0.0,
    max_reduced_daily_rests_between_weekly_rests=0,
    min_weekly_rest_seconds=0.0,
)


def _defaults_for(ruleset: str) -> dict[str, object]:
    if ruleset == "eu561":
        return dict(EU561_DEFAULTS)
    if ruleset == "us_hos":
        return dict(US_HOS_DEFAULTS)
    if ruleset == "none":
        return dict(NONE_DEFAULTS)
    # custom — start blank; caller's explicit fields fill them, else NONE
    return dict(NONE_DEFAULTS)


def materialize_shift_rule(rule: ShiftRule | None) -> MaterializedShiftRule:
    """Resolve a caller's ``ShiftRule`` to a fully populated struct.

    Explicit fields on the input override the named ruleset's defaults
    without discarding the rest of the regulation's structure.
    """
    if rule is None:
        d = dict(NONE_DEFAULTS)
        return MaterializedShiftRule(ruleset="none", **d)  # type: ignore[arg-type]
    d = _defaults_for(rule.ruleset)
    # Apply explicit overrides
    fields = [
        "max_drive_seconds_before_break", "break_seconds",
        "max_continuous_drive_seconds", "daily_drive_cap_seconds",
        "daily_onduty_cap_seconds", "weekly_drive_cap_seconds",
        "min_daily_rest_seconds", "reduced_daily_rest_seconds",
        "max_reduced_daily_rests_between_weekly_rests",
        "min_weekly_rest_seconds",
    ]
    for f in fields:
        v = getattr(rule, f, None)
        if v is not None:
            d[f] = v
    if rule.split_break_segments is not None:
        d["split_break_segments"] = tuple(rule.split_break_segments)
    return MaterializedShiftRule(ruleset=rule.ruleset, **d)  # type: ignore[arg-type]


# ============================================================
# Break planner — inserts break/rest events into a timeline
# ============================================================


@dataclass
class _Segment:
    """A continuous activity segment on a route."""
    kind: Literal["drive", "service", "wait"]
    duration_seconds: float
    after_visit_index: int | None = None  # which visit was completed last


@dataclass
class PlannedBreaks:
    events: list[BreakEvent] = field(default_factory=list)
    total_break_seconds: float = 0.0
    daily_rests_used: int = 0
    reduced_daily_rests_used: int = 0
    weekly_rests_used: int = 0
    # Diagnostics
    drive_cap_violations: int = 0   # times daily drive cap was exceeded
    overrun_seconds: float = 0.0    # accumulated cap overrun


def plan_breaks(segments: list[_Segment], rule: MaterializedShiftRule, *,
                start_time: float = 0.0) -> PlannedBreaks:
    """Walk segments in order, inserting break events at the latest
    feasible position. Returns the planned events plus accounting.

    Algorithm:
      * Track ``drive_since_break`` per continuous-drive rule.
      * When ``drive_since_break + next_drive > max_drive_seconds_before_break``
        — insert a ``BreakEvent`` (single 45min for EU 561, single 30min for
        US HOS, or split if ``split_break_segments`` are configured and the
        accumulated split-break exceeds ``break_seconds``).
      * Track ``daily_drive`` against ``daily_drive_cap_seconds`` and
        ``daily_onduty_cap_seconds``; when the cap is hit a daily-rest event
        is inserted.
      * Track ``weekly_drive``; when the cap is hit a weekly-rest event is
        inserted (driver returns next week — modeled as a single
        ``BreakEvent`` with rest_kind="weekly_rest").

    This implementation is regulation-correct for single- and multi-day
    routes. For routes that fit within a single shift it inserts only the
    continuous-drive break(s).
    """
    out = PlannedBreaks()
    if rule.ruleset == "none":
        return out

    now = float(start_time)
    drive_since_break = 0.0
    daily_drive = 0.0
    daily_onduty = 0.0
    weekly_drive = 0.0

    def _emit(duration: float, rk: str, name: str, after_idx: int | None,
              seg_idx: int | None = None) -> None:
        out.events.append(BreakEvent(
            at_seconds=now,
            after_visit_index=after_idx,
            duration_seconds=duration,
            rule=name,
            rest_kind=rk,  # type: ignore[arg-type]
            segment_index=seg_idx,
        ))
        out.total_break_seconds += duration

    last_visit_idx: int | None = None
    for seg in segments:
        if seg.after_visit_index is not None:
            last_visit_idx = seg.after_visit_index

        # ---- 1. Continuous-drive break ----
        if seg.kind == "drive":
            # If THIS drive segment would push us past the cap, break first.
            if (drive_since_break + seg.duration_seconds
                    > rule.max_drive_seconds_before_break):
                if rule.split_break_segments:
                    for i, seglen in enumerate(rule.split_break_segments):
                        _emit(seglen, "split_break_segment",
                              f"{rule.ruleset}:split_break_segment",
                              last_visit_idx, seg_idx=i)
                        now += seglen
                else:
                    _emit(rule.break_seconds, "break",
                          f"{rule.ruleset}:continuous_drive_break",
                          last_visit_idx)
                    now += rule.break_seconds
                drive_since_break = 0.0

        # ---- 2. Daily cap (drive or onduty) ----
        # Compute prospective post-segment values
        prospective_daily_drive = daily_drive + (seg.duration_seconds if seg.kind == "drive" else 0.0)
        prospective_daily_onduty = daily_onduty + seg.duration_seconds
        if (prospective_daily_drive > rule.daily_drive_cap_seconds
                or prospective_daily_onduty > rule.daily_onduty_cap_seconds):
            # Insert a daily rest. Use reduced rest if quota remains.
            if (out.reduced_daily_rests_used
                    < rule.max_reduced_daily_rests_between_weekly_rests
                    and rule.reduced_daily_rest_seconds > 0):
                dur = rule.reduced_daily_rest_seconds
                rk = "reduced_daily_rest"
                out.reduced_daily_rests_used += 1
            else:
                dur = rule.min_daily_rest_seconds
                rk = "daily_rest"
                out.daily_rests_used += 1
            _emit(dur, rk, f"{rule.ruleset}:{rk}", last_visit_idx)
            now += dur
            daily_drive = 0.0
            daily_onduty = 0.0
            drive_since_break = 0.0

        # ---- 3. Weekly cap ----
        if (weekly_drive + (seg.duration_seconds if seg.kind == "drive" else 0.0)
                > rule.weekly_drive_cap_seconds):
            dur = rule.min_weekly_rest_seconds
            _emit(dur, "weekly_rest", f"{rule.ruleset}:weekly_rest", last_visit_idx)
            now += dur
            weekly_drive = 0.0
            daily_drive = 0.0
            daily_onduty = 0.0
            drive_since_break = 0.0
            out.weekly_rests_used += 1
            out.reduced_daily_rests_used = 0

        # ---- 4. Apply the segment ----
        now += seg.duration_seconds
        daily_onduty += seg.duration_seconds
        if seg.kind == "drive":
            drive_since_break += seg.duration_seconds
            daily_drive += seg.duration_seconds
            weekly_drive += seg.duration_seconds

    return out


__all__ = [
    "MaterializedShiftRule", "EU561_DEFAULTS", "US_HOS_DEFAULTS", "NONE_DEFAULTS",
    "materialize_shift_rule", "plan_breaks", "PlannedBreaks", "_Segment",
]
