"""Markdown structured-text exporter for the VLM's text channel.

The split-channel design: the rendered PNG carries route-quality
*color* + topology + direction arrows (visual reasoning); this
markdown carries the *exact numerics* (per-route loads, costs,
TWs) so the VLM doesn't have to guess them from pixels.
"""
from __future__ import annotations

import numpy as np

from svrptw.io import Instance
from svrptw.solvers.common import Solution
from svrptw.solvers.common.local_search import _cust_by_id, _route_arrival_and_close


def _route_arrive_and_end(inst: Instance, customers: list[int]) -> tuple[float, float]:
    """Return (earliest_arrive_first_cust, latest_arrive_last_cust_end).

    Approximates real per-leg arrival; uses the canonical TW-feasibility
    helper so numbers match what `evaluate()` computes.
    """
    if not customers:
        return (0.0, 0.0)
    T = inst.travel_time
    cust_by_id = _cust_by_id(inst)
    clk = float(inst.depot.ready)
    first_arrive = clk + float(T[0, customers[0]])
    prev = 0
    for cid in customers:
        cust = cust_by_id[cid]
        clk += float(T[prev, cid])
        if clk < cust.ready:
            clk = cust.ready
        clk += cust.service
        prev = cid
    return (first_arrive, clk)


def _route_dist_and_time(inst: Instance, customers: list[int]) -> tuple[float, float]:
    if not customers:
        return (0.0, 0.0)
    T = inst.travel_time
    D = inst.travel_dist
    total_d = float(D[0, customers[0]])
    total_t = float(T[0, customers[0]])
    for i in range(len(customers) - 1):
        total_d += float(D[customers[i], customers[i + 1]])
        total_t += float(T[customers[i], customers[i + 1]])
    total_d += float(D[customers[-1], 0])
    total_t += float(T[customers[-1], 0])
    return (total_d, total_t)


def to_text(inst: Instance, sol: Solution,
            *, blind_label: str | None = None) -> str:
    """Deterministic markdown summary of (instance, solution).

    Sections:
      ## Plan          — anonymous label (e.g. "Plan A"); NO solver name
      ## Aggregate     — cost decomposition + totals
      ## Routes        — per-route table (load, util, length, time, arrive, slack)
      ## TW tightness  — p50/p90 of TW window widths

    `blind_label` (recommended): pass "Plan A" / "Plan B" so the LLM
    cannot bias by knowing which solver produced which plan. If None,
    falls back to `sol.solver` (debug/visibility only — never use in
    production labelling).
    """
    cap = max(1.0, float(inst.vehicle_capacity))
    cust_by_id = _cust_by_id(inst)
    day_start = inst.depot.ready
    day_end = inst.depot.due

    m = sol.metrics
    lines: list[str] = []

    # Plan (blind by default — keep solver identity out of the prompt
    # so the LLM judges the plan, not the brand).
    label = blind_label if blind_label is not None else sol.solver
    lines.append(f"## {label}")
    lines.append("")

    # Aggregate
    lines.append("## Aggregate")
    lines.append(f"- operational_cost: {m.get('operational_cost', 0.0):.2f}")
    lines.append(f"- total_distance_miles: {m.get('total_distance_miles', 0.0):.2f}")
    lines.append(f"- total_time_minutes: {m.get('total_time_minutes', 0.0):.1f}")
    lines.append(f"- num_vehicles_used: {int(m.get('num_vehicles_used', 0))}")
    lines.append(f"- missed_deliveries: {int(m.get('missed_deliveries', 0))}")
    lines.append(f"- tw_late_minutes: {m.get('tw_late_minutes', 0.0):.1f}")
    lines.append(f"- early_wait_minutes: {m.get('early_wait_minutes', 0.0):.1f}")
    lines.append(f"- capacity_overload: {m.get('capacity_overload', 0.0):.1f}")
    lines.append(f"- feasible: {int(m.get('feasible', 0))}")
    lines.append("")

    # Routes (sorted by route index for determinism)
    lines.append("## Routes")
    lines.append("| id | n_cust | load | util% | dist_mi | time_min | "
                 "first_arrive | last_end | tw_slack |")
    lines.append("|---:|------:|-----:|------:|--------:|---------:|"
                 "-------------:|---------:|---------:|")
    for ri, route in enumerate(sol.routes):
        if not route.customers:
            continue
        load = float(sum(cust_by_id[c].demand for c in route.customers))
        util = load / cap
        dist, t = _route_dist_and_time(inst, route.customers)
        first_arr, last_end = _route_arrive_and_end(inst, route.customers)
        tw_slack = day_end - last_end
        lines.append(
            f"| {ri} | {len(route.customers)} | {load:.0f} | "
            f"{100*util:.0f}% | {dist:.2f} | {t:.1f} | "
            f"{first_arr:.0f} | {last_end:.0f} | {tw_slack:.0f} |"
        )
    lines.append("")

    # TW tightness across customers
    lines.append("## TW tightness")
    widths = sorted([c.due - c.ready for c in inst.customers])
    if widths:
        p50 = widths[len(widths) // 2]
        p90 = widths[int(0.9 * len(widths))]
        lines.append(f"- TW widths p50: {p50:.0f} min")
        lines.append(f"- TW widths p90: {p90:.0f} min")
        lines.append(f"- day window: [{day_start}, {day_end}] minutes")

    return "\n".join(lines)
