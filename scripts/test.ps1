<#
.SYNOPSIS
    Chạy toàn bộ kiểm thử Pose Studio.
.EXAMPLE
    .\scripts\test.ps1              # Unit test BE/FE, smoke model, build
    .\scripts\test.ps1 -E2E         # Thêm Playwright E2E (tự bật/tắt BE và FE)
    .\scripts\test.ps1 -E2E -Video C:\path\to\clip.mp4
#>
param(
    [switch]$E2E,
    [string]$Video
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$feDir = Join-Path $projectRoot 'fe'
$results = [System.Collections.Generic.List[object]]::new()

$pythonExe = @('.venv', 'venv') |
    ForEach-Object { Join-Path $projectRoot "$_\Scripts\python.exe" } |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
if (-not $pythonExe) {
    throw 'Missing virtual environment (.venv or venv). Run: py -3.11 -m venv venv; .\venv\Scripts\python.exe -m pip install -r be/requirements-dev.txt'
}
if (-not (Test-Path -LiteralPath (Join-Path $feDir 'node_modules'))) {
    throw 'Missing frontend dependencies. Run npm.cmd ci in the fe directory.'
}

function Invoke-Step([string]$Name, [string]$WorkDir, [scriptblock]$Command) {
    Write-Host "`n==> $Name" -ForegroundColor Cyan
    $watch = [Diagnostics.Stopwatch]::StartNew()
    Push-Location -LiteralPath $WorkDir
    try {
        & $Command
        $ok = ($LASTEXITCODE -eq 0)
    } catch {
        Write-Host $_ -ForegroundColor Red
        $ok = $false
    } finally {
        Pop-Location
    }
    $results.Add([pscustomobject]@{
        Step    = $Name
        Result  = $(if ($ok) { 'PASS' } else { 'FAIL' })
        Seconds = [math]::Round($watch.Elapsed.TotalSeconds, 1)
    })
}

function Test-Url([string]$Url, [string]$Contains) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return ($response.StatusCode -eq 200) -and (-not $Contains -or $response.Content -match $Contains)
    } catch {
        return $false
    }
}

function Wait-Url([string]$Name, [string]$Url, [string]$Contains, [int]$TimeoutSec = 120) {
    Write-Host "Waiting for $Name ($Url)..."
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Url $Url $Contains) { return }
        Start-Sleep -Seconds 2
    }
    throw "$Name did not become ready within $TimeoutSec seconds: $Url"
}

Invoke-Step 'Backend pytest' $projectRoot { & $pythonExe -m pytest be/tests -q -p no:cacheprovider }
Invoke-Step 'Model smoke test' $projectRoot { & $pythonExe -m be.scripts.smoke_model }
Invoke-Step 'Frontend Vitest' $feDir { & npm.cmd test }
Invoke-Step 'Frontend build' $feDir { & npm.cmd run build }

if ($E2E) {
    $healthUrl = 'http://127.0.0.1:8000/api/health'
    $feUrl = 'http://127.0.0.1:5173'
    $started = @()
    $logDir = Join-Path $feDir 'test-results'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null

    try {
        if (Test-Url $healthUrl) {
            Write-Host 'Backend already running, reusing it.'
        } else {
            Write-Host 'Starting backend...'
            $started += Start-Process -FilePath $pythonExe -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $logDir 'be.out.log') -RedirectStandardError (Join-Path $logDir 'be.err.log') `
                -ArgumentList '-m', 'uvicorn', 'be.app.main:app', '--host', '127.0.0.1', '--port', '8000', '--ws-max-size', '2097152', '--ws-max-queue', '1'
        }
        if (Test-Url $feUrl) {
            Write-Host 'Frontend already running, reusing it.'
        } else {
            Write-Host 'Starting frontend...'
            $started += Start-Process -FilePath 'npm.cmd' -WorkingDirectory $feDir -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $logDir 'fe.out.log') -RedirectStandardError (Join-Path $logDir 'fe.err.log') `
                -ArgumentList 'run', 'dev', '--', '--host', '127.0.0.1', '--port', '5173', '--strictPort'
        }

        Wait-Url 'backend' $healthUrl '"ready"'
        Wait-Url 'frontend' $feUrl

        if (-not $Video) {
            $Video = Join-Path $projectRoot '.superpowers\sdd\2026-09-23-pose-studio\pose-test.webm'
        }
        if (Test-Path -LiteralPath $Video) {
            $env:POSE_TEST_VIDEO = (Resolve-Path -LiteralPath $Video).Path
            Write-Host "Video test: $env:POSE_TEST_VIDEO"
        } else {
            Remove-Item Env:POSE_TEST_VIDEO -ErrorAction SilentlyContinue
            Write-Host 'No test video found; the uploaded-video E2E test will be skipped.' -ForegroundColor Yellow
        }

        Invoke-Step 'Playwright browser install' $feDir { & npx.cmd playwright install chromium }
        Invoke-Step 'Playwright E2E' $feDir { & npx.cmd playwright test }
    } catch {
        Write-Host $_ -ForegroundColor Red
        $results.Add([pscustomobject]@{ Step = 'E2E setup'; Result = 'FAIL'; Seconds = 0 })
        Write-Host "Server logs: $logDir" -ForegroundColor Yellow
    } finally {
        foreach ($process in $started) {
            # npm.cmd spawns node children; kill the whole tree.
            & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
        }
        if ($started) { Write-Host 'Stopped servers started by this script.' }
    }
}

Write-Host "`n==> Summary" -ForegroundColor Cyan
$results | Format-Table -AutoSize | Out-String | Write-Host
$failed = @($results | Where-Object Result -eq 'FAIL').Count
if ($failed) {
    Write-Host "$failed step(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host 'All steps passed.' -ForegroundColor Green
exit 0
