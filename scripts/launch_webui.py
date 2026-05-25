"""launch_webui.py -- one-command launcher (cross-shell variant).

Behavior matches scripts/launch_webui.ps1:
  1. Detect repo root from this file's location
  2. If port already serving /status -> reuse, skip launch
  3. Else fork `python -m webui.app --port PORT` as a background process
  4. Poll /status until 200 (or 10s timeout)
  5. Open default browser
  6. Print the env export line for other shells

Stdlib only. Idempotent.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_VENV_PY = _REPO / ".venv" / "Scripts" / "python.exe"
_PIDFILE_DIR = _REPO / ".cache" / "webui"


def _python() -> str:
    if _VENV_PY.exists():
        return str(_VENV_PY)
    return shutil.which("python") or sys.executable


def is_ready(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/status", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def _pidfile(port: int) -> Path:
    return _PIDFILE_DIR / f"webui_{port}.pid"


def _existing_pid(port: int) -> int | None:
    p = _pidfile(port)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def _spawn(host: str, port: int) -> int:
    """Spawn `python -m webui.app --port ... --host ...` detached. Returns pid.

    Windows: uses CREATE_NO_WINDOW so no console flashes when the
    launcher is invoked from a GUI shortcut or from PowerShell. Combined
    with DETACHED_PROCESS the server has no parent shell, no visible
    window, survives the launcher exiting.
    """
    py = _python()
    # Use pythonw.exe (no-console launcher) on Windows when available --
    # belt-and-braces against any console flash from the python.exe path.
    if sys.platform.startswith("win"):
        pyw = Path(py).with_name("pythonw.exe")
        if pyw.exists():
            py = str(pyw)
    args = [py, "-m", "webui.app", "--port", str(port), "--host", host]
    flags = 0
    if sys.platform.startswith("win"):
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.DETACHED_PROCESS
                 | subprocess.CREATE_NO_WINDOW)
    proc = subprocess.Popen(
        args, cwd=str(_REPO),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
    )
    _PIDFILE_DIR.mkdir(parents=True, exist_ok=True)
    _pidfile(port).write_text(str(proc.pid))
    return proc.pid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    url = f"http://{args.host}:{args.port}"

    if is_ready(url):
        print(f"[launch_webui] server already up at {url}")
    else:
        existing_pid = _existing_pid(args.port)
        if existing_pid is not None:
            print(f"[launch_webui] stale pidfile (pid={existing_pid}); spawning fresh")
        pid = _spawn(args.host, args.port)
        print(f"[launch_webui] spawned pid={pid}")
        deadline = time.time() + args.timeout
        ready = False
        while time.time() < deadline:
            if is_ready(url):
                ready = True
                break
            time.sleep(0.25)
        if not ready:
            print(f"[launch_webui] FAILED: server did not respond within {args.timeout}s",
                  file=sys.stderr)
            return 1
        print(f"[launch_webui] ready at {url}")

    os.environ["SVRPTW_WEBUI_URL"] = url
    if not args.no_browser:
        webbrowser.open(url)
    print()
    print(f"    URL:   {url}")
    print(f"    Paste this into other shells to point benches at it:")
    if sys.platform.startswith("win"):
        print(f"        $env:SVRPTW_WEBUI_URL = \"{url}\"")
    else:
        print(f"        export SVRPTW_WEBUI_URL=\"{url}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
