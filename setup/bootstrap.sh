#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m ipykernel install --user --name scam_shield
