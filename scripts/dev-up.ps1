Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $projectRoot ".codex-run"
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"
$pythonExe = Join-Path $backendDir ".venv-win\Scripts\python.exe"
$viteCmd = Join-Path $frontendDir "node_modules\.bin\vite.cmd"
$composeFile = Join-Path $projectRoot "docker-compose.yml"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Read-EnvFile {
    param([string]$Path)

    $values = @{}
    if (-not (Test-Path $Path)) {
        return $values
    }

    foreach ($line in Get-Content -Path $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }

        $pair = $trimmed.Split("=", 2)
        if ($pair.Count -ne 2) {
            continue
        }

        $key = $pair[0].Trim()
        $value = $pair[1].Trim()
        if ($value.Length -ge 2) {
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        $values[$key] = $value
    }

    return $values
}

function Apply-EnvironmentFiles {
    param([string[]]$Paths)

    foreach ($path in $Paths) {
        $values = Read-EnvFile -Path $path
        foreach ($key in $values.Keys) {
            [Environment]::SetEnvironmentVariable($key, $values[$key], "Process")
        }
    }
}

function Get-ListeningProcessId {
    param([int]$Port)

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($connections) {
        return ($connections | Select-Object -First 1 -ExpandProperty OwningProcess)
    }
    return $null
}

function Test-HttpReady {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5 | Out-Null
            return $true
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

function Test-RedisReady {
    param(
        [string]$PythonPath,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        & $PythonPath -c "from redis import Redis; Redis.from_url('redis://localhost:6379/0').ping(); print('ok')" *> $null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Assert-Exists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path $Path)) {
        throw "$Label not found: $Path"
    }
}

function Start-LoggedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory
    )

    $outLog = Join-Path $runDir "$Name.out.log"
    $errLog = Join-Path $runDir "$Name.err.log"
    $pidFile = Join-Path $runDir "$Name.pid"

    if (Test-Path $outLog) {
        Remove-Item $outLog -Force
    }
    if (Test-Path $errLog) {
        Remove-Item $errLog -Force
    }

    $proc = Start-Process -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -WindowStyle Hidden `
        -PassThru

    Set-Content -Path $pidFile -Value $proc.Id -Encoding ASCII
    return $proc
}

function Get-RecordedProcess {
    param([string]$Name)

    $pidFile = Join-Path $runDir "$Name.pid"
    if (-not (Test-Path $pidFile)) {
        return $null
    }

    $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if (-not $pidValue) {
        return $null
    }

    $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
    if (-not $proc) {
        Remove-Item $pidFile -ErrorAction SilentlyContinue
        return $null
    }
    return $proc
}

function Get-WorkerProcess {
    $candidate = Get-RecordedProcess -Name "worker"
    if ($candidate) {
        return $candidate
    }

    $processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*-m app.worker*" -and $_.ExecutablePath -eq $pythonExe }

    if ($processes) {
        $match = $processes | Select-Object -First 1
        Set-Content -Path (Join-Path $runDir "worker.pid") -Value $match.ProcessId -Encoding ASCII
        return (Get-Process -Id $match.ProcessId -ErrorAction SilentlyContinue)
    }

    return $null
}

function Ensure-DockerDesktop {
    Write-Step "Checking Docker Desktop"
    try {
        & docker info *> $null
    } catch {
        throw "Docker Desktop is not running. Redis and Qdrant cannot be started."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop is not running. Redis and Qdrant cannot be started."
    }
}

function Ensure-ComposeService {
    param(
        [string]$ServiceName,
        [scriptblock]$HealthCheck,
        [string]$FailureMessage
    )

    if (& $HealthCheck) {
        Write-Step "$ServiceName is ready"
        return
    }

    Write-Step "Starting $ServiceName container"
    & docker compose -f $composeFile up -d $ServiceName
    if ($LASTEXITCODE -ne 0) {
        throw "$ServiceName container failed to start."
    }

    if (-not (& $HealthCheck)) {
        throw $FailureMessage
    }
}

function Assert-NoRunningComposeMode {
    $runningServices = & docker compose -f $composeFile ps --services --filter "status=running"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect docker compose service status."
    }

    $conflicts = @($runningServices | Where-Object { $_ -in @("api", "worker", "frontend", "frontend-dev") })
    if ($conflicts.Count -gt 0) {
        $joined = ($conflicts -join ", ")
        throw "Docker compose mode is already running ($joined). Stop the container stack before starting local development mode."
    }
}

function Assert-NoConflictingRqWorkers {
    param(
        [string]$RedisUrl,
        [string]$ConflictQueueName
    )

    $pythonCode = @"
from datetime import datetime, timedelta
from redis import Redis
from rq import Worker
import sys

redis_url = sys.argv[1]
conflict_queue = sys.argv[2]
r = Redis.from_url(redis_url)
cutoff = datetime.utcnow() - timedelta(seconds=90)
conflicts = []
for worker in Worker.all(connection=r):
    heartbeat = worker.last_heartbeat
    if heartbeat is None:
        continue
    heartbeat = heartbeat.replace(tzinfo=None)
    if heartbeat < cutoff:
        continue
    queue_names = [queue.name for queue in worker.queues]
    if conflict_queue in queue_names:
        conflicts.append(f"{worker.hostname} pid={worker.pid} queues={','.join(queue_names)}")

if conflicts:
    print("; ".join(conflicts))
    raise SystemExit(2)
"@

    $output = & $pythonExe -c $pythonCode $RedisUrl $ConflictQueueName
    if ($LASTEXITCODE -eq 2) {
        throw "Detected active docker-mode workers on queue '$ConflictQueueName'. Stop docker compose mode first. Conflicts: $output"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect Redis worker registrations."
    }
}

Assert-Exists -Path $pythonExe -Label "Python virtualenv"
Assert-Exists -Path $viteCmd -Label "Vite executable"
Assert-Exists -Path $composeFile -Label "docker-compose file"

Apply-EnvironmentFiles -Paths @(
    (Join-Path $projectRoot ".env"),
    (Join-Path $projectRoot ".env.dev")
)
[Environment]::SetEnvironmentVariable("USE_RQ", "true", "Process")
[Environment]::SetEnvironmentVariable("RQ_QUEUE_NAME", "default_local", "Process")
[Environment]::SetEnvironmentVariable("RQ_RUNTIME_MODE", "local", "Process")

Ensure-DockerDesktop
Assert-NoRunningComposeMode

Ensure-ComposeService -ServiceName "redis" `
    -HealthCheck { Test-RedisReady -PythonPath $pythonExe -TimeoutSeconds 30 } `
    -FailureMessage "Redis is still unavailable after startup."

Ensure-ComposeService -ServiceName "qdrant" `
    -HealthCheck { Test-HttpReady -Url "http://localhost:6333/collections" -TimeoutSeconds 30 } `
    -FailureMessage "Qdrant is still unavailable after startup."

Assert-NoConflictingRqWorkers -RedisUrl $env:REDIS_URL -ConflictQueueName "default_docker"

$workerProcess = Get-WorkerProcess
if ($workerProcess) {
    Write-Step "Worker already running, PID=$($workerProcess.Id)"
} else {
    Write-Step "Starting RQ worker"
    $workerProcess = Start-LoggedProcess -Name "worker" -FilePath $pythonExe -ArgumentList @("-m", "app.worker") -WorkingDirectory $backendDir
}

Start-Sleep -Seconds 2
if (-not (Get-Process -Id $workerProcess.Id -ErrorAction SilentlyContinue)) {
    throw "RQ worker failed to start. Check .codex-run\worker.err.log"
}
if (-not (Test-RedisReady -PythonPath $pythonExe -TimeoutSeconds 15)) {
    throw "Redis is still unavailable after worker startup."
}

$apiPid = Get-ListeningProcessId -Port 8000
if ($apiPid) {
    Write-Step "API already running, PID=$apiPid"
    Set-Content -Path (Join-Path $runDir "backend.pid") -Value $apiPid -Encoding ASCII
} else {
    Write-Step "Starting API"
    Start-LoggedProcess -Name "backend" -FilePath $pythonExe -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000") -WorkingDirectory $backendDir | Out-Null
}

if (-not (Test-HttpReady -Url "http://127.0.0.1:8000/api/health" -TimeoutSeconds 30)) {
    throw "API failed to start. Check .codex-run\backend.err.log"
}

$frontendPid = Get-ListeningProcessId -Port 5173
if ($frontendPid) {
    Write-Step "Frontend already running, PID=$frontendPid"
    Set-Content -Path (Join-Path $runDir "frontend.pid") -Value $frontendPid -Encoding ASCII
} else {
    Write-Step "Starting Frontend"
    Start-LoggedProcess -Name "frontend" -FilePath $viteCmd -ArgumentList @("--host", "0.0.0.0", "--port", "5173") -WorkingDirectory $frontendDir | Out-Null
}

if (-not (Test-HttpReady -Url "http://127.0.0.1:5173" -TimeoutSeconds 30)) {
    throw "Frontend failed to start. Check .codex-run\frontend.err.log"
}

Write-Host ""
Write-Host "Development environment is ready:"
Write-Host "  Frontend: http://localhost:5173"
Write-Host "  Admin:    http://localhost:5173/admin"
Write-Host "  API:      http://localhost:8000/docs"
Write-Host "  Qdrant:   http://localhost:6333/dashboard"
