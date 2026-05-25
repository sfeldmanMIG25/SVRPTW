"""ProcessRegistry -- subprocess lifecycle owner for whitelisted bench scripts.

Owned by the webui server process. Spawns ``python -m <module> ...`` as
detached child processes (Windows: ``CREATE_NEW_PROCESS_GROUP |
DETACHED_PROCESS`` so we can send ``CTRL_BREAK_EVENT`` to the whole
group, which is required to propagate stop signals to
``ProcessPoolExecutor`` workers spawned by the bench).

Each running process has a background reader thread that pumps stdout
into ``bus().add_log(...)`` and posts agent lifecycle status (running /
completed / failed) to the bus.

stdlib only: subprocess, threading, signal, os, json, pathlib, time.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from webui.event_bus import bus

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Filled in by webui.app.serve() once the HTTP server is bound. Used to
# inject SVRPTW_WEBUI_URL into spawned bench subprocesses so they push
# events back to this same server.
_SERVER_URL: Optional[str] = None


def set_server_url(url: str) -> None:
    """Called by webui.app.serve() after binding -- exposes the server
    URL to the registry so spawned subprocesses can push events back."""
    global _SERVER_URL
    _SERVER_URL = url


def _server_url() -> Optional[str]:
    return _SERVER_URL


# Whitelist of allowed `--out` path prefixes. Any --out value that is not
# prefixed by one of these (relative to repo root) is rejected.
_ALLOWED_OUT_PREFIXES = ("bench/runs/", "bench/runs\\")


# Two whitelisted bench scripts. Anything else POSTed to /control/run is
# rejected. The args_allowed set defines the only flag names that may
# appear; args_choices restricts allowed values for specific flags.
WHITELIST: dict[str, dict[str, Any]] = {
    "cb_scaled_rebench": {
        "module": "bench.scripts.cb_scaled_rebench",
        "args_allowed": {"--workers", "--webui", "--include", "--out"},
        "args_choices": {
            "--include": {"v1_ood", "homberger_n400"},
        },
    },
    "v1_leaderboard_solve_auto": {
        "module": "bench.scripts.v1_leaderboard_solve_auto",
        "args_allowed": {"--workers", "--webui", "--N", "--reps",
                         "--cities", "--out"},
        "args_choices": {},
    },
}

# Reject any value with these characters -- defense in depth even though
# we always pass args via list and shell=False.
_FORBIDDEN_CHARS = re.compile(r"[;|&<>$`\n\r]")


def validate_args(script_name: str, args: list[str]) -> list[str]:
    """Walk the arg list and reject anything unsafe.

    Rules:
      - script_name must be in WHITELIST
      - every flag (token starting with ``--``) must be in args_allowed
      - if the flag has args_choices, each subsequent value (until the
        next flag) must be in the allowed set
      - no value may contain shell metacharacters
      - --out values must be relative paths under bench/runs/ (no .., no
        absolute paths, no drive letters)

    Returns the sanitized arg list (currently unchanged from input).
    Raises ValueError on any violation.
    """
    if script_name not in WHITELIST:
        raise ValueError(f"unknown script: {script_name!r}")
    spec = WHITELIST[script_name]
    allowed = spec["args_allowed"]
    choices = spec.get("args_choices", {}) or {}

    sanitized: list[str] = []
    i = 0
    current_flag: Optional[str] = None
    while i < len(args):
        tok = str(args[i])
        if _FORBIDDEN_CHARS.search(tok):
            raise ValueError(f"forbidden character in arg: {tok!r}")
        if tok.startswith("--"):
            if tok not in allowed:
                raise ValueError(
                    f"flag {tok!r} not allowed for {script_name!r}; "
                    f"allowed: {sorted(allowed)}"
                )
            current_flag = tok
            sanitized.append(tok)
        else:
            # Value for the previous flag.
            if current_flag is None:
                raise ValueError(f"orphan value (no preceding flag): {tok!r}")
            if current_flag in choices and tok not in choices[current_flag]:
                raise ValueError(
                    f"value {tok!r} not allowed for {current_flag}; "
                    f"choices: {sorted(choices[current_flag])}"
                )
            if current_flag == "--out":
                _validate_out_path(tok)
            sanitized.append(tok)
        i += 1
    return sanitized


def _validate_out_path(value: str) -> None:
    """Ensure --out is a relative path under bench/runs/."""
    v = value.replace("\\", "/")
    if v.startswith("/") or (len(v) >= 2 and v[1] == ":"):
        raise ValueError(f"--out must be a relative path under bench/runs/: {value!r}")
    if ".." in v.split("/"):
        raise ValueError(f"--out may not contain '..': {value!r}")
    if not v.startswith("bench/runs/"):
        raise ValueError(
            f"--out must be under bench/runs/, got {value!r}"
        )


class _ProcEntry:
    """Tracking record for one running subprocess."""
    __slots__ = ("name", "proc", "started_at", "args", "thread", "module",
                 "_final_emitted")

    def __init__(self, name: str, proc: subprocess.Popen, args: list[str],
                 module: str) -> None:
        self.name = name
        self.proc = proc
        self.args = list(args)
        self.module = module
        self.started_at = time.time()
        self.thread: Optional[threading.Thread] = None
        self._final_emitted = False


class ProcessRegistry:
    """Singleton owner of all spawned bench subprocesses.

    All public methods are safe to call from the HTTP handler thread.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._procs: dict[str, _ProcEntry] = {}

    # ---- query --------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for entry in self._procs.values():
                rc = entry.proc.poll()
                out.append({
                    "name": entry.name,
                    "pid": entry.proc.pid,
                    "started_at": entry.started_at,
                    "returncode": rc,
                    "args": list(entry.args),
                    "module": entry.module,
                })
            return out

    def is_running(self, name: str) -> bool:
        with self._lock:
            entry = self._procs.get(name)
            if entry is None:
                return False
            return entry.proc.poll() is None


    # ---- mutation -----------------------------------------------

    def run(self, name: str, args: list[str]) -> dict[str, Any]:
        """Spawn ``python -m <whitelist[name].module> <args>``.

        Validates args via ``validate_args``. Raises ValueError on bad
        args. Raises RuntimeError if a process with this name is
        already running.
        """
        with self._lock:
            existing = self._procs.get(name)
            if existing is not None and existing.proc.poll() is None:
                raise RuntimeError(f"already running: {name!r}")

            sanitized = validate_args(name, args)
            module = WHITELIST[name]["module"]
            cmd = [sys.executable, "-u", "-m", module, *sanitized]

            env = os.environ.copy()
            # Point the spawned bench at this webui server so its
            # ui.push_* helpers POST events to /event. We pull the URL
            # from the live HTTP server bound by serve(); fall back to
            # the inherited env var if the server hasn't registered yet.
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("PYTHONPATH", str(_REPO_ROOT))
            srv_url = _server_url()
            if srv_url and not env.get("SVRPTW_WEBUI_URL"):
                env["SVRPTW_WEBUI_URL"] = srv_url

            flags = 0
            if sys.platform.startswith("win"):
                flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                         | subprocess.DETACHED_PROCESS)

            proc = subprocess.Popen(
                cmd,
                cwd=str(_REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                creationflags=flags,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            entry = _ProcEntry(name, proc, sanitized, module)
            self._procs[name] = entry


            t = threading.Thread(
                target=self._reader_loop,
                args=(entry,),
                daemon=True,
                name=f"procreader-{name}",
            )
            entry.thread = t
            t.start()

            bus().set_agent_status(
                name=name, status="running",
                summary=f"pid={proc.pid} module={module} args={' '.join(sanitized)}",
            )
            bus().add_log(f"[procreg] started {name} pid={proc.pid}")

            return {
                "name": name,
                "pid": proc.pid,
                "started_at": entry.started_at,
            }

    def stop(self, name: str) -> dict[str, Any]:
        """Stop a running process. Idempotent."""
        with self._lock:
            entry = self._procs.get(name)
        if entry is None:
            return {"name": name, "returncode": None, "note": "not_running"}
        rc = entry.proc.poll()
        if rc is not None:
            # Already exited; clean record.
            return {"name": name, "returncode": rc, "note": "already_exited"}


        bus().add_log(f"[procreg] stopping {name} pid={entry.proc.pid}")

        # Step 1: graceful signal.
        if sys.platform.startswith("win"):
            try:
                os.kill(entry.proc.pid, signal.CTRL_BREAK_EVENT)
            except (OSError, ValueError) as e:
                bus().add_log(f"[procreg] CTRL_BREAK failed: {e}")
        else:
            try:
                entry.proc.terminate()
            except OSError as e:
                bus().add_log(f"[procreg] terminate failed: {e}")

        rc = self._wait(entry.proc, 5.0)

        # Step 2: terminate() (Windows) or escalate.
        if rc is None:
            try:
                entry.proc.terminate()
            except OSError:
                pass
            rc = self._wait(entry.proc, 2.0)

        # Step 3: hard kill.
        if rc is None:
            try:
                entry.proc.kill()
            except OSError:
                pass
            rc = self._wait(entry.proc, 2.0)


        # The reader thread will emit the final agent status when it
        # observes the process exit (poll != None). But ensure we also
        # post one even if the reader hasn't gotten there yet.
        if not entry._final_emitted:
            self._emit_final(entry, rc)
        return {"name": name, "returncode": rc}

    @staticmethod
    def _wait(proc: subprocess.Popen, timeout_s: float) -> Optional[int]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rc = proc.poll()
            if rc is not None:
                return rc
            time.sleep(0.1)
        return proc.poll()

    # ---- internals ----------------------------------------------

    def _emit_final(self, entry: _ProcEntry, rc: Optional[int]) -> None:
        if entry._final_emitted:
            return
        entry._final_emitted = True
        status = ("completed" if rc == 0
                  else ("failed" if rc is not None else "stopped"))
        bus().set_agent_status(
            name=entry.name,
            status=status,
            summary=f"pid={entry.proc.pid} returncode={rc}",
        )
        bus().add_log(f"[procreg] {entry.name} exit rc={rc}")


    def _reader_loop(self, entry: _ProcEntry) -> None:
        """Pump child stdout into bus().add_log; emit final status on exit."""
        proc = entry.proc
        try:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n").rstrip("\r")
                if line:
                    bus().add_log(f"[{entry.name}] {line}")
                if proc.poll() is not None:
                    # Drain remaining buffered lines, then break.
                    pass
        except Exception as e:  # pragma: no cover -- defensive
            bus().add_log(f"[procreg] reader error for {entry.name}: {e}")
        finally:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass
            rc = proc.poll()
            if rc is None:
                # Process still alive but stdout closed -- wait briefly.
                rc = self._wait(proc, 2.0)
            self._emit_final(entry, rc)


# ---- module singleton -----------------------------------------------

_REGISTRY = ProcessRegistry()


def registry() -> ProcessRegistry:
    return _REGISTRY


def reset_registry() -> None:
    """Test helper -- create a fresh registry. Does NOT stop running procs."""
    global _REGISTRY
    _REGISTRY = ProcessRegistry()
