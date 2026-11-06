# =============================================================================
# setup_windows.ps1 — project setup for Windows 10/11 (PowerShell).
#
# Installs everything into THIS project's virtual environment (.venv\) only.
#
# Usage (PowerShell, inside the repository folder):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#
# Afterwards:
#   .\.venv\Scripts\Activate.ps1     # turn the environment on
#   jupyter lab                      # or open the folder in VS Code and pick .venv
#   deactivate                       # turn it off
# If activation is blocked ("running scripts is disabled"), run once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# =============================================================================
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$root = (Get-Location).Path
# Keep all downloads and caches inside the project folder:
$env:PIP_CACHE_DIR = "$root\.cache\pip"; $env:HF_HOME = "$root\.cache\huggingface"; $env:TORCH_HOME = "$root\.cache\torch"

Write-Host "==> 1/4 Finding Python (3.10 or newer)"
$pyCmd = $null
& py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
if ($LASTEXITCODE -eq 0) { $pyCmd = @("py", "-3") }
if (-not $pyCmd) {
    Write-Host "Python 3.10+ not found. Install Python 3.12 or newer from https://www.python.org/downloads/windows/"
    Write-Host "(tick 'Add python.exe to PATH' and keep the 'py launcher' option), then re-run this script."
    exit 1
}
& $pyCmd[0] $pyCmd[1] --version

Write-Host "==> 2/4 Creating the project virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $pyCmd[0] $pyCmd[1] -m venv .venv }
$vpy = (Resolve-Path ".venv\Scripts\python.exe").Path

Write-Host "==> 3/4 Installing packages into .venv (first time: a few minutes)"
& $vpy -m pip install --upgrade pip
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host ""
    Write-Host "NVIDIA GPU detected. For GPU speed, install the CUDA build of PyTorch INTO THE VENV first:"
    Write-Host "  open https://pytorch.org/get-started/locally/  (Stable, Windows, Pip, Python, your CUDA version)"
    Write-Host "  and run the command it shows, replacing 'pip3' with:  .\.venv\Scripts\python -m pip"
    Write-Host "Then re-run this script. Continuing with the default (CPU) build for now..."
    Write-Host ""
}
& $vpy -m pip install -r requirements.txt
& $vpy -m pip install -e . --no-deps

Write-Host "==> 4/4 Quick check"
& $vpy -c "from dlhandson.device import get_device, describe_device; print(describe_device(get_device()))"
Write-Host ""
Write-Host "Setup complete. Next: open notebooks\00_preflight_check.ipynb in VS Code (kernel: .venv) and Run All."
