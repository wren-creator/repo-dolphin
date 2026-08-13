#!/usr/bin/env bash
# ./ask.sh "a question"   -> one-shot answer
# ./ask.sh                -> interactive session (loads the index once)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

if [ "$#" -eq 0 ]; then
    python repl.py
else
    python ask.py "$@"
fi
