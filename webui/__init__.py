"""Live web UI for solver progress + snapshots + judge panels.

Stdlib-only (http.server + ThreadingHTTPServer). No new deps.

Public surface:
  webui.event_bus   — thread-safe singleton state used by solvers/benches
  webui.app         — `python -m webui.app [--port 8765]` to start the server
  webui.snapshot    — render-helper used by bench scripts to push live frames
"""
from __future__ import annotations
