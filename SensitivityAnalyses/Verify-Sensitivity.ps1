& {
    $ErrorActionPreference = 'Stop'
    $Root = $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($Root)) { $Root = (Get-Location).Path }
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $Python) { throw 'Python was not found in PATH.' }
    & $Python.Source (Join-Path $Root 'verify_sensitivity.py')
    if ($LASTEXITCODE -ne 0) { throw "Addendum verification failed with code $LASTEXITCODE." }
}
