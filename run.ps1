$Root = $PSScriptRoot
if ($args.Count -eq 0) { & python (Join-Path $Root 'acet.py') list }
else { & python (Join-Path $Root 'acet.py') @args }
exit $LASTEXITCODE
