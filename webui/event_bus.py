"""Thread-safe singleton state for the live web UI.

Solvers and bench scripts push events here; the HTTP layer reads
them via `bus().snapshot()` and serves to the browser.

Design choices:
- stdlib only (threading.RLock, collections.deque, dataclasses)
- bounded buffers (snapshot history per stream, log lines)
- monotonic timestamps for everything visible to the UI
- safe to import from ProcessPoolExecutor workers (each gets its own
  bus, so push events from the *parent* process for cross-worker view)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

_LOG_BUFFER_SIZE = 400
_SNAPSHOT_HISTORY_PER_STREAM = 40


@dataclass
class Snapshot:
    label: str          # e.g. "warm@iter=37" or "PyVRP@final"
    rel_path: str       # path under /snapshots/, served by the HTTP layer
    stream: str         # logical channel: "warm", "pyvrp", "compare-A", "compare-B"
    ts: float           # unix epoch
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchProgress:
    name: str
    completed: int
    total: int
    eta_seconds: Optional[float] = None
    started_at: Optional[float] = None


@dataclass
class JudgePanel:
    """Live multi-judge VLM scoring on a (solution_a, solution_b) pair."""
    pair_id: str
    judges: list[dict[str, Any]] = field(default_factory=list)
    consensus: Optional[dict[str, Any]] = None
    ts: float = 0.0


class EventBus:
    """Thread-safe singleton state for the live web UI."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stage: str = "idle"
        self._stage_msg: str = ""
        self._stage_extra: dict[str, Any] = {}
        self._snapshots: dict[str, deque[Snapshot]] = {}
        self._log: deque[str] = deque(maxlen=_LOG_BUFFER_SIZE)
        self._bench: dict[str, BenchProgress] = {}
        self._judges: dict[str, JudgePanel] = {}
        # SPEC-WEBUI-05 -- per-agent dashboard. The main agent or a
        # bench wrapper pushes lifecycle + progress events here when it
        # spawns sub-agents (or when it itself is being polled). The
        # HTTP route surfaces this as the "Agents" panel.
        self._agents: dict[str, dict[str, Any]] = {}
        # SPEC-WEBUI-07 -- per-stream time series (cost-over-iteration etc.).
        # Each stream is a deque of (x, y, ts) tuples; the page renders an
        # inline SVG sparkline per stream.
        self._series: dict[str, deque[tuple[float, float, float]]] = {}
        self._started_at = time.time()

    # ---- setters ---------------------------------------------------

    def set_stage(self, stage: str, msg: str = "", **extra: Any) -> None:
        with self._lock:
            self._stage = stage
            self._stage_msg = msg
            self._stage_extra = dict(extra)
            self._log.append(f"[{stage}] {msg}".rstrip())

    def add_snapshot(self, label: str, rel_path: str | Path,
                     stream: str = "default", **meta: Any) -> None:
        snap = Snapshot(label=label, rel_path=str(rel_path), stream=stream,
                        ts=time.time(), meta=dict(meta))
        with self._lock:
            buf = self._snapshots.setdefault(
                stream, deque(maxlen=_SNAPSHOT_HISTORY_PER_STREAM))
            buf.append(snap)

    def add_log(self, line: str) -> None:
        with self._lock:
            self._log.append(line.rstrip())

    def set_bench_progress(self, name: str, completed: int, total: int,
                            eta_seconds: Optional[float] = None) -> None:
        with self._lock:
            existing = self._bench.get(name)
            started = existing.started_at if existing else time.time()
            self._bench[name] = BenchProgress(
                name=name, completed=completed, total=total,
                eta_seconds=eta_seconds, started_at=started,
            )

    def set_judge_panel(self, pair_id: str,
                         judges: list[dict[str, Any]],
                         consensus: Optional[dict[str, Any]] = None) -> None:
        with self._lock:
            self._judges[pair_id] = JudgePanel(
                pair_id=pair_id,
                judges=list(judges),
                consensus=consensus,
                ts=time.time(),
            )

    def clear_stream(self, stream: str) -> None:
        """Drop all snapshots in a stream (e.g. starting a fresh solve)."""
        with self._lock:
            self._snapshots.pop(stream, None)

    def set_agent_status(self, name: str, status: str, *,
                          progress: Optional[float] = None,
                          eta_seconds: Optional[float] = None,
                          summary: str = "",
                          **extra: Any) -> None:
        """Update a single agent's lifecycle row.

        ``status`` is free-form (e.g. "running", "completed", "failed").
        ``progress`` is an optional 0..1 fraction for the bar.
        First-call sets ``started_at``; subsequent calls keep it.
        """
        now = time.time()
        with self._lock:
            cur = self._agents.get(name, {})
            started = float(cur.get("started_at", now))
            self._agents[name] = {
                "name": name,
                "status": str(status),
                "summary": str(summary),
                "progress": (None if progress is None else float(progress)),
                "eta_seconds": (None if eta_seconds is None else float(eta_seconds)),
                "started_at": started,
                "updated_at": now,
                **{k: v for k, v in extra.items()
                    if k not in ("started_at", "updated_at", "name", "status",
                                 "summary", "progress", "eta_seconds")},
            }

    def remove_agent(self, name: str) -> None:
        with self._lock:
            self._agents.pop(name, None)

    def push_series_point(self, stream: str, x: float, y: float) -> None:
        """Append (x, y, ts) to a per-stream time series.

        Bounded to the last 500 points per stream so old runs don't
        pile up indefinitely. Page renders sparklines from this.
        """
        with self._lock:
            buf = self._series.setdefault(stream, deque(maxlen=500))
            buf.append((float(x), float(y), time.time()))

    def clear_series(self, stream: str | None = None) -> None:
        with self._lock:
            if stream is None:
                self._series.clear()
            else:
                self._series.pop(stream, None)

    # ---- getter ----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stage": self._stage,
                "stage_msg": self._stage_msg,
                "stage_extra": dict(self._stage_extra),
                "snapshots": {
                    stream: [asdict(s) for s in list(buf)]
                    for stream, buf in self._snapshots.items()
                },
                "log": list(self._log),
                "bench": {n: asdict(p) for n, p in self._bench.items()},
                "judges": {pid: asdict(jp) for pid, jp in self._judges.items()},
                "agents": {n: dict(d) for n, d in self._agents.items()},
                "series": {
                    stream: [{"x": x, "y": y, "ts": ts}
                              for (x, y, ts) in list(buf)]
                    for stream, buf in self._series.items()
                },
                "uptime_s": time.time() - self._started_at,
            }


# ---- module singleton ------------------------------------------------

_BUS: EventBus = EventBus()


def bus() -> EventBus:
    return _BUS


def reset() -> None:
    """Test helper — drop the singleton between unit tests."""
    global _BUS
    _BUS = EventBus()
