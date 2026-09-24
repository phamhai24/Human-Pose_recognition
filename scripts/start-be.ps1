$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Missing .venv. Run: py -3.11 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r be/requirements-dev.txt'
}
Write-Host 'Pose Studio backend: http://127.0.0.1:8000 | API docs: /docs'
& $pythonExe -m uvicorn be.app.main:app --host 127.0.0.1 --port 8000 --ws-max-size 2097152 --ws-max-queue 1
exit $LASTEXITCODE
