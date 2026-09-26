$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = @('.venv', 'venv') |
    ForEach-Object { Join-Path $projectRoot "$_\Scripts\python.exe" } |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
if (-not $pythonExe) {
    throw 'Missing virtual environment (.venv or venv). Run: py -3.11 -m venv venv; .\venv\Scripts\python.exe -m pip install -r be/requirements-dev.txt'
}
Write-Host 'Pose Studio backend: http://127.0.0.1:8000 | API docs: /docs'
& $pythonExe -m uvicorn be.app.main:app --host 127.0.0.1 --port 8000 --ws-max-size 2097152 --ws-max-queue 1
exit $LASTEXITCODE
