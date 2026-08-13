#!/usr/bin/env bash
# Rebuilds the index from whatever's currently in the repos listed in
# config.yaml. Run after a work session on any of them.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate
python index.py
