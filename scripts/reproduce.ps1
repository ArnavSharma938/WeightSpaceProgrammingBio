$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    py -3.12 -m venv (Join-Path $ProjectRoot '.venv')
}

& $Python -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
& $Python -m pip install -e $ProjectRoot --no-deps
& $Python (Join-Path $PSScriptRoot 'download_data.py')
& $Python (Join-Path $PSScriptRoot 'cnns.py') prepare
& $Python (Join-Path $PSScriptRoot 'cnns.py') train --device cpu
& $Python (Join-Path $PSScriptRoot 'cnns.py') evaluate --device cpu
