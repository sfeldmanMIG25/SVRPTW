# launch_webui.ps1 -- one-command launcher for the WorldsFinestVRP live UI.
#
# Behavior:
#   1. Detect repo root (assumes this script lives in <repo>/scripts/)
#   2. If port 8765 is already serving /status -> reuse, skip launch
#   3. Else start `python -m webui.app --port 8765` as a background Job
#   4. Poll /status until 200 (or 10s timeout)
#   5. Set $env:SVRPTW_WEBUI_URL in the *current* shell
#   6. Open http://127.0.0.1:8765/ in the default browser
#   7. Print the env export line so the user can paste it into other terminals
#
# Idempotent: re-running detects the existing server and skips the launch.

param(
    [int]$Port = 8765,
    [string]$BindHost = "127.0.0.1",
    [int]$TimeoutSeconds = 10,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Url = "http://${BindHost}:${Port}"
$StatusUrl = "$Url/status"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }


function Test-WebUIReady {
    try {
        $r = Invoke-WebRequest -Uri $StatusUrl -UseBasicParsing -TimeoutSec 1.5 -ErrorAction Stop
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Start-WebUI {
    # Use Python launcher's detached-spawn path -- Start-Job dies with the
    # parent shell, which doesn't survive `pwsh -File launch_webui.ps1`.
    # The Python launcher uses CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    # so the server actually outlives this script.
    $launcher = Join-Path $RepoRoot "scripts\launch_webui.py"
    if (Test-Path $launcher) {
        & $Python $launcher --port $Port --host $BindHost --no-browser | Out-Host
        return
    }
    # fallback: Start-Process detached if the python launcher is missing
    $args = @("-m", "webui.app", "--port", $Port, "--host", $BindHost)
    Start-Process -FilePath $Python -ArgumentList $args -WindowStyle Hidden `
        -WorkingDirectory $RepoRoot | Out-Null
    Write-Host "[launch_webui] started detached server process"
}


# 1. fast path: server already up?
if (Test-WebUIReady) {
    Write-Host "[launch_webui] server already up at $Url"
} else {
    Start-WebUI
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if (Test-WebUIReady) { $ready = $true; break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) {
        Write-Error "[launch_webui] server failed to come up at $Url within $TimeoutSeconds s"
        exit 1
    }
    Write-Host "[launch_webui] server is ready"
}

# 2. set env in current shell
$env:SVRPTW_WEBUI_URL = $Url

# 3. open browser (unless suppressed)
if (-not $NoBrowser) { Start-Process $Url }

# 4. print paste-ready export so user can wire other terminals
Write-Host ""
Write-Host "    URL:   $Url"
Write-Host "    Paste this into other PowerShell terminals to point benches at it:"
Write-Host "        `$env:SVRPTW_WEBUI_URL = `"$Url`""
Write-Host ""
Write-Host "    To stop:   Get-Process python | Where-Object {`$_.CommandLine -like '*webui.app*'} | Stop-Process"
