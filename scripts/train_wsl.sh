#!/usr/bin/env bash
# Runs train_local.py inside the WSL venv from Windows, where ~ is the WSL home; extra arguments pass through,
# e.g. --max-steps 3 for a smoke test:
#   wsl -d Ubuntu-24.04 -- bash scripts/train_wsl.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. ~/ft/.venv/bin/activate
python scripts/train_local.py --out ~/ft/out "$@"
