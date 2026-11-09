# ARBench one-command launcher (Windows PowerShell)
#   .\run.ps1                       -> standard profile
#   .\run.ps1 --profile quick       -> any run_benchmark.py arguments are passed through
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "[ARBench] Creating virtual environment (.venv) ..."
    $py = (Get-Command py -ErrorAction SilentlyContinue)
    if ($py) { & py -3 -m venv .venv } else { & python -m venv .venv }
}

$req = Join-Path $PSScriptRoot "requirements.txt"
$stamp = Join-Path $PSScriptRoot ".venv\.requirements.sha"
$hash = (Get-FileHash $req -Algorithm SHA256).Hash
$installed = if (Test-Path $stamp) { Get-Content $stamp -Raw } else { "" }
if ($installed.Trim() -ne $hash) {
    Write-Host "[ARBench] Installing Python dependencies ..."
    & $venvPy -m pip install --disable-pip-version-check -q --upgrade pip
    & $venvPy -m pip install --disable-pip-version-check -q -r $req
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    Set-Content -Path $stamp -Value $hash -Encoding ascii
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Warning "Ollama is not on PATH. Install it from https://ollama.com/download (or pass --host to a running server)."
}

& $venvPy (Join-Path $PSScriptRoot "run_benchmark.py") @args
exit $LASTEXITCODE
