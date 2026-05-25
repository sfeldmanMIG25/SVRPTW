"""Render-and-push helper for the live web UI.

Bench scripts and per-iteration callbacks call ``push_solution()`` to
write a PNG into the static snapshot dir AND register it with the bus
in one shot. The web page picks it up on its next 1.5s poll.

Usage:
    from webui.snapshot import push_solution
    push_solution(inst, sol, label="warm@iter=37", stream="warm",
                  cost=sol.metrics["operational_cost"], mode="llm_compare")
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from webui import client as ui

if TYPE_CHECKING:
    from svrptw.io.instance import Instance
    from svrptw.solvers.common.solution import Solution

_SNAPSHOT_DIR = Path(__file__).resolve().parent / "static" / "snapshots"
_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(label: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    return "".join(c if c in keep else "_" for c in label)[:80]


def push_solution(
    inst: "Instance",
    sol: "Solution",
    *,
    label: str,
    stream: str = "default",
    mode: str = "llm_compare",
    cost: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Render `sol` to a PNG under static/snapshots/ and register it.

    Returns the absolute path written. Side-effect: pushes a Snapshot
    entry to the EventBus so the live page picks it up.

    `mode` selects the render style. ``"llm_compare"`` (default) is the
    LLM-comp safe layout; pass ``"fair_mode"`` or ``"human"`` to use the
    older renderer modes (see svrptw.viz.renderer).
    """
    from svrptw.viz.renderer import render_solution
    ts_ms = int(time.time() * 1000)
    fname = f"{_slugify(stream)}__{_slugify(label)}__{ts_ms}.png"
    out = _SNAPSHOT_DIR / fname
    fair_mode = (mode in ("fair_mode", "llm_compare"))
    render_solution(inst, sol, out, fair_mode=fair_mode)
    meta: dict[str, Any] = dict(extra or {})
    if cost is not None:
        meta["cost"] = float(cost)
    meta.setdefault("mode", mode)
    ui.push_snapshot(label=label, rel_path=fname, stream=stream, **meta)
    return out


def make_on_accept(
    inst: "Instance",
    *,
    stream: str,
    every_n_accepts: int = 5,
    min_improvement: float = 5.0,
    mode: str = "llm_compare",
) -> "Callable":
    """Build a rate-limited ``on_accept`` callback for the bandit loop.

    The bandit calls ``on_accept(op_name, improvement, ops_applied,
    new_sol)`` on every accepted move; rendering every one is wasteful
    (50-150ms per render at N<=500).

    This factory returns a callback that pushes a snapshot only when:
      - it's the Nth accept since the last push (every_n_accepts), AND
      - the improvement clears ``min_improvement`` dollars.

    The callback closes over ``inst`` so the bandit doesn't have to
    pass it in (callers know the instance at construction time).
    """
    state = {"last_push_at": -1, "n_accepts": 0}

    def cb(op_name: str, improvement: float, ops_applied: int,
           new_sol: object) -> None:
        state["n_accepts"] += 1
        # ALWAYS push the cost-over-iteration series point (cheap, no render).
        # The chart panel uses this to draw a sparkline of solver progress.
        try:
            cost_now = float(getattr(new_sol, "metrics", {}).get(
                "operational_cost", 0.0))
            ui.push_series(stream, x=int(ops_applied), y=cost_now)
        except Exception:
            pass
        # accept-count gate (rendering is the expensive part)
        if state["n_accepts"] % max(1, every_n_accepts) != 0:
            return
        # improvement gate
        if abs(improvement) < min_improvement:
            return
        try:
            push_solution(
                inst, new_sol,  # type: ignore[arg-type]
                label=f"{op_name}@i{ops_applied}",
                stream=stream,
                mode=mode,
                cost=getattr(new_sol, "metrics", {}).get("operational_cost"),
                extra={"operator": op_name, "improvement": float(improvement),
                       "ops_applied": int(ops_applied)},
            )
            state["last_push_at"] = ops_applied
        except Exception:
            pass

    return cb
