$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath (Join-Path $projectRoot 'fe')
if (-not (Test-Path -LiteralPath 'node_modules')) {
    throw 'Missing frontend dependencies. Run npm.cmd ci in the fe directory.'
}
Write-Host 'Pose Studio frontend: http://127.0.0.1:5173'
& npm.cmd run dev -- --host 127.0.0.1
exit $LASTEXITCODE
