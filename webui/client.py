"""HTTP client for cross-process bus pushes.

Bench scripts and per-iteration solver hooks call these helpers.
If env var ``SVRPTW_WEBUI_URL`` is set (e.g. ``http://127.0.0.1:8765``),
events are POSTed to that running webui server. Otherwise events are
pushed to the in-process EventBus singleton (useful for tests + for
single-process embedded use).

This decouples senders from the transport. Senders never import the
bus directly; they use these helpers.

Usage::

    from webui import client as ui
    ui.push_stage("solving", "warm phase started", N=200)
    ui.push_snapshot(label="warm@iter=37", rel_path="warm__37.png",
                     stream="warm", cost=2031.5)
    ui.push_progress("cb_scaled_rebench", 5, 24, eta_seconds=180)
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Optional

_LOG = logging.getLogger("webui.client")
_TIMEOUT_S = 1.5  # short -- never block the solver


def _server_url() -> Optional[str]:
    return os.environ.get("SVRPTW_WEBUI_URL", "").rstrip("/") or None


def _post(etype: str, payload: dict[str, Any]) -> None:
    url = _server_url()
    if url is None:
        # In-process fallback.
        try:
            from webui.event_bus import bus
            b = bus()
            if etype == "stage":
                b.set_stage(str(payload.get("stage", "")),
                            str(payload.get("msg", "")),
                            **{k: v for k, v in payload.items()
                                if k not in ("stage", "msg")})
            elif etype == "snapshot":
                b.add_snapshot(label=str(payload["label"]),
                               rel_path=str(payload["rel_path"]),
                               stream=str(payload.get("stream", "default")),
                               **{k: v for k, v in payload.items()
                                  if k not in ("label", "rel_path", "stream")})
            elif etype == "progress":
                b.set_bench_progress(payload["name"],
                                     int(payload["completed"]),
                                     int(payload["total"]),
                                     eta_seconds=payload.get("eta_seconds"))
            elif etype == "judges":
                b.set_judge_panel(payload["pair_id"],
                                  list(payload.get("judges", [])),
                                  payload.get("consensus"))
            elif etype == "log":
                b.add_log(str(payload.get("line", "")))
            elif etype == "agent":
                b.set_agent_status(
                    name=str(payload["name"]),
                    status=str(payload.get("status", "")),
                    progress=payload.get("progress"),
                    eta_seconds=payload.get("eta_seconds"),
                    summary=str(payload.get("summary", "")),
                    **{k: v for k, v in payload.items()
                        if k not in ("name", "status", "progress",
                                     "eta_seconds", "summary")},
                )
            elif etype == "series":
                b.push_series_point(
                    stream=str(payload["stream"]),
                    x=float(payload["x"]), y=float(payload["y"]),
                )
        except Exception as e:
            _LOG.debug("in-process push failed: %s", e)
        return

    body = json.dumps({"type": etype, "payload": payload}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/event",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            resp.read()
    except urllib.error.URLError as e:
        _LOG.debug("HTTP push failed (%s): %s", etype, e)
    except Exception as e:
        _LOG.debug("HTTP push unexpected error (%s): %s", etype, e)


# ---- public helpers -----------------------------------------------

def push_stage(stage: str, msg: str = "", **extra: Any) -> None:
    payload = {"stage": stage, "msg": msg}
    payload.update(extra)
    _post("stage", payload)


def push_snapshot(label: str, rel_path: str, stream: str = "default",
                  **meta: Any) -> None:
    payload = {"label": label, "rel_path": rel_path, "stream": stream}
    payload.update(meta)
    _post("snapshot", payload)


def push_progress(name: str, completed: int, total: int,
                  eta_seconds: float | None = None) -> None:
    payload: dict[str, Any] = {
        "name": name, "completed": int(completed), "total": int(total),
    }
    if eta_seconds is not None:
        payload["eta_seconds"] = float(eta_seconds)
    _post("progress", payload)


def push_judges(pair_id: str, judges: list[dict[str, Any]],
                 consensus: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"pair_id": pair_id, "judges": list(judges)}
    if consensus is not None:
        payload["consensus"] = dict(consensus)
    _post("judges", payload)


def push_log(line: str) -> None:
    _post("log", {"line": line})


def is_remote() -> bool:
    """True iff SVRPTW_WEBUI_URL is set (events go HTTP, not in-process)."""
    return _server_url() is not None


def push_series(stream: str, x: float, y: float) -> None:
    """Append a (x, y) point to a named time series.

    The page renders an SVG sparkline per stream below the snapshots
    panel. Typical use: push (iteration_count, operational_cost) on
    every accepted bandit move.
    """
    _post("series", {"stream": stream, "x": float(x), "y": float(y)})


def push_agent(name: str, status: str, *,
               progress: float | None = None,
               eta_seconds: float | None = None,
               summary: str = "",
               **extra: Any) -> None:
    """Update an agent's lifecycle row.

    Status is free-form; common values: "queued", "running", "completed", "failed".
    Progress is an optional 0..1 fraction for the bar.
    """
    payload: dict[str, Any] = {"name": name, "status": status, "summary": summary}
    if progress is not None:
        payload["progress"] = float(progress)
    if eta_seconds is not None:
        payload["eta_seconds"] = float(eta_seconds)
    payload.update(extra)
    _post("agent", payload)
