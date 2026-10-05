# Full run on the real Bossard files that sit one folder above this repo (Windows / PowerShell).
# Usage:  powershell -ExecutionPolicy Bypass -File run_bossard.ps1     (extra flags are passed through)
Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"
python -m smartbin run `
  --measurements ..\measurements.csv `
  --metadata ..\scale_metadata.csv `
  --preprocessing ..\scale_preprocessing.csv `
  --reports reports\bossard @args
