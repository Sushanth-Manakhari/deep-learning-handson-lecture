#!/usr/bin/env bash
# =============================================================================
# setup_mac.sh — project setup for macOS (also works on Linux).
#
# Everything is installed into THIS project's virtual environment (.venv/).
# Nothing is installed globally: every install uses .venv/bin/python -m pip.
#
# Usage (from anywhere):
#   bash scripts/setup_mac.sh            # students: create .venv + install packages
#   bash scripts/setup_mac.sh --verify   # instructor: + build the Beans pack +
#                                        #   real end-to-end run -> reports/verify_report.md
#
# Afterwards, to work in the project manually:
#   source .venv/bin/activate      # turn the environment on
#   jupyter lab                    # or open the folder in VS Code and pick .venv
#   deactivate                     # turn it off
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
mkdir -p reports
LOG="$ROOT/reports/setup_log.txt"
exec > >(tee "$LOG") 2>&1       # keep a copy of all output for troubleshooting
VERIFY=0; [[ "${1:-}" == "--verify" ]] && VERIFY=1

# Keep ALL downloads and caches inside the project folder (nothing in ~/.cache):
export PIP_CACHE_DIR="$ROOT/.cache/pip"           # pip's download cache
export HF_HOME="$ROOT/.cache/huggingface"         # Hugging Face models + datasets
export TORCH_HOME="$ROOT/.cache/torch"            # torchvision pretrained weights
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"     # matplotlib font cache

step() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }

# ---- 1. Use the Homebrew Python (`python3`), whatever version it is -----------
# No version is hard-coded: `brew upgrade python` just works. Minimum 3.10.
# Override with:  PYTHON=/path/to/python3 bash scripts/setup_mac.sh
step "1/5 Finding Python"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif command -v brew >/dev/null 2>&1 && [[ -x "$(brew --prefix)/bin/python3" ]]; then
  PY="$(brew --prefix)/bin/python3"          # Homebrew's python3 (e.g. /opt/homebrew/bin/python3)
else
  PY="$(command -v python3 || true)"
fi
if [[ -z "$PY" ]] || ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "Python 3.10 or newer is required. Install it with:  brew install python"; exit 1
fi
echo "using $PY ($("$PY" --version))"

# PyYAML (needed by Hugging Face + Gradio) has no prebuilt wheel for the newest
# Python versions, so pip compiles it: that needs Apple's command-line tools.
if [[ "$(uname)" == "Darwin" ]] && ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools are missing. Run:  xcode-select --install   then re-run this script."; exit 1
fi

# ---- 2. Create the project virtual environment ------------------------------
step "2/5 Creating the project virtual environment (.venv)"
if [[ ! -x .venv/bin/python ]]; then
  "$PY" -m venv .venv
  echo "created $ROOT/.venv"
else
  echo ".venv already exists — reusing it"
fi
VPY="$ROOT/.venv/bin/python"
echo "packages will be installed ONLY into: $("$VPY" -c 'import sys; print(sys.prefix)')"

# ---- 3. Install packages into .venv -----------------------------------------
step "3/5 Installing packages into .venv (first time: a few minutes)"
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt
"$VPY" -m pip install -e . --no-deps        # makes `import dlhandson` work from anywhere in the venv
"$VPY" -m pip freeze > reports/pip_freeze.txt  # exact versions installed (for the verification report)

# Use certifi's CA bundle for downloads made by this script only (fixes
# CERTIFICATE_VERIFY_FAILED on python.org builds of Python for macOS).
export SSL_CERT_FILE="$("$VPY" -c 'import certifi; print(certifi.where())')"

# ---- 4. Git hooks (slides/secret guard) --------------------------------------
step "4/5 Installing git hooks"
if [[ -d .git ]]; then bash scripts/install_git_hooks.sh; else echo "(not a git clone — skipped)"; fi

# ---- 5. Quick check / instructor verification ---------------------------------
if [[ "$VERIFY" -eq 1 ]]; then
  step "5/5 Instructor verification (downloads ~1 GB, ~15-20 min)"
  "$VPY" scripts/build_beans_pack.py
  "$VPY" scripts/prepare_stress_images.py || true
  export PYTORCH_ENABLE_MPS_FALLBACK=1        # run any op MPS lacks on the CPU instead of failing
  "$VPY" scripts/verify.py
  echo; echo "Report: $ROOT/reports/verify_report.md"
else
  step "5/5 Quick check"
  "$VPY" -c "import torch, dlhandson; from dlhandson.device import get_device, describe_device; \
print(describe_device(get_device()))"
  echo
  echo "✅ Setup complete. Next: open notebooks/00_preflight_check.ipynb (VS Code or 'jupyter lab')"
  echo "   with the .venv kernel and choose Run All. Activate later with: source .venv/bin/activate"
fi
