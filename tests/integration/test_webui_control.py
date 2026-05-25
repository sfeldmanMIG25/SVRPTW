"""End-to-end integration test for the B0pp /control/* surface.

Spawns the webui server on a fresh port (8767), drives a small
cb_scaled_rebench run through it via /control/run, asserts the bench
pushes progress events to the bus, then /control/stop's it and
finally /control/clear-snapshots.

Skips if the OSM-*-N100-I003.json instances aren't on disk -- the
bench would build zero tasks and immediately exit.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_PORT = 8767
_BASE = f"http://127.0.0.1:{_PORT}"


def _instances_present() -> bool:
    base = _REPO / "instances" / "v1"
    return any(base.glob("OSM-*-N100-I003.json"))


def _http_get(path: str, timeout: float = 2.0) -> tuple[int, dict | str]:
    req = urllib.request.Request(_BASE + path)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8")
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body


def _http_post(path: str, body: dict | None, timeout: float = 5.0) -> tuple[int, dict | str]:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        _BASE + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode("utf-8")
            try:
                return r.status, json.loads(txt)
            except json.JSONDecodeError:
                return r.status, txt
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def _wait_until_ready(deadline_s: float = 12.0) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            code, _ = _http_get("/status", timeout=1.0)
            if code == 200:
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.2)
    return False


@pytest.fixture(scope="module")
def webui_server():
    """Spawn the webui server on a non-default port. Tear it down at end."""
    if not _instances_present():
        pytest.skip("instances missing -- need instances/v1/OSM-*-N100-I003.json")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    flags = 0
    if sys.platform.startswith("win"):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "webui.app", "--port", str(_PORT)],
        cwd=str(_REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=flags,
    )
    try:
        ok = _wait_until_ready(12.0)
        if not ok:
            try:
                out = proc.stdout.read(2000) if proc.stdout else b""
            except Exception:
                out = b""
            pytest.fail(f"webui never became ready on {_BASE}; stdout: {out!r}")
        yield proc
    finally:
        try:
            if sys.platform.startswith("win"):
                try:
                    os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                except OSError:
                    pass
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except Exception:
            pass


def test_control_run_stop_clear(webui_server):
    # 0. Sanity: /status should be reachable.
    code, body = _http_get("/status")
    assert code == 200
    assert isinstance(body, dict) and "stage" in body

    # 1. Spawn the small bench.
    code, payload = _http_post(
        "/control/run",
        {"script": "cb_scaled_rebench",
         "args": ["--workers", "2", "--webui", "--include", "v1_ood"]},
    )
    assert code == 200, f"unexpected: {code} {payload}"
    assert isinstance(payload, dict) and payload.get("ok") is True
    assert payload.get("name") == "cb_scaled_rebench"
    assert isinstance(payload.get("pid"), int) and payload["pid"] > 0

    # 2. Within 8s, the bench should have pushed at least one progress
    # event so /status.bench includes our key.
    deadline = time.time() + 8.0
    saw = False
    while time.time() < deadline:
        code, body = _http_get("/status")
        if (code == 200
                and isinstance(body, dict)
                and "cb_scaled_rebench" in (body.get("bench") or {})):
            saw = True
            break
        time.sleep(0.4)
    assert saw, "bench progress event never observed in /status.bench"

    # 3. /control/list should show it.
    code, body = _http_get("/control/list")
    assert code == 200 and isinstance(body, dict)
    procs = body.get("procs") or []
    assert any(p.get("name") == "cb_scaled_rebench" for p in procs)

    # 4. Stop the bench. Must complete within ~10s.
    code, payload = _http_post(
        "/control/stop", {"name": "cb_scaled_rebench"}, timeout=12.0
    )
    assert code == 200, f"unexpected: {code} {payload}"
    assert isinstance(payload, dict) and payload.get("ok") is True
    # returncode may be non-zero (we killed it) but must be set.
    assert "returncode" in payload

    # 5. Clear snapshots -- must succeed even if no PNGs exist.
    code, payload = _http_post("/control/clear-snapshots", None)
    assert code == 200, f"unexpected: {code} {payload}"
    assert isinstance(payload, dict) and payload.get("ok") is True
    assert isinstance(payload.get("removed"), int)


def test_control_run_validation_errors(webui_server):
    """Bad scripts / flags should be rejected with 400, never 500."""
    code, _ = _http_post("/control/run", {"script": "evil_script", "args": []})
    assert code == 400

    code, _ = _http_post(
        "/control/run",
        {"script": "cb_scaled_rebench", "args": ["--unknown-flag", "x"]},
    )
    assert code == 400

    code, _ = _http_post(
        "/control/run",
        {"script": "cb_scaled_rebench", "args": ["--include", "wat"]},
    )
    assert code == 400


def test_viewer_route(webui_server):
    code, body = _http_get("/viewer")
    assert code == 200
    assert isinstance(body, str)
    assert "snapshot viewer" in body.lower()


def test_build_report(webui_server):
    code, payload = _http_post("/control/build-report", None)
    assert code == 200
    assert isinstance(payload, dict)
    assert payload.get("ok") is True
    assert payload.get("report_url") == "/reports/progress"
    # And the report should now be reachable.
    code, _ = _http_get("/reports/progress")
    assert code == 200
