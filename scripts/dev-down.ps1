param(
    [switch]$StopDependencies
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $projectRoot ".codex-run"
$composeFile = Join-Path $projectRoot "docker-compose.yml"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Stop-RecordedProcess {
    param([string]$Name)

    $pidFile = Join-Path $runDir "$Name.pid"
    if (-not (Test-Path $pidFile)) {
        return
    }

    $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($pidValue) {
        $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Step "Stopping $Name, PID=$pidValue"
            Stop-Process -Id ([int]$pidValue) -Force
        }
    }

    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

function Stop-ProcessOnPort {
    param(
        [int]$Port,
        [string]$Name
    )

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        $proc = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Step "Stopping $Name on port $Port, PID=$($proc.Id)"
            Stop-Process -Id $proc.Id -Force
        }
    }
}

function Stop-WorkerFallback {
    $processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*-m app.worker*" }
    foreach ($process in $processes) {
        $proc = Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Step "Stopping worker fallback, PID=$($proc.Id)"
            Stop-Process -Id $proc.Id -Force
        }
    }
}

if (Test-Path $runDir) {
    Stop-RecordedProcess -Name "frontend"
    Stop-RecordedProcess -Name "backend"
    Stop-RecordedProcess -Name "worker"
}

Stop-ProcessOnPort -Port 5173 -Name "frontend"
Stop-ProcessOnPort -Port 8000 -Name "backend"
Stop-WorkerFallback

if ($StopDependencies) {
    Write-Step "Stopping Redis and Qdrant containers"
    & docker compose -f $composeFile stop redis qdrant
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to stop Redis/Qdrant containers."
    }
}

Write-Host "Development environment has been stopped."
