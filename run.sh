#!/usr/bin/env bash
# RecallOS - one-click launcher (macOS / Linux).
#
# Make it executable once, then run it:
#   chmod +x run.sh
#   ./run.sh
#
# The script checks for .env, copies .env.example if missing, and
# then starts Streamlit in the local venv (or with python3).

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
    echo "[RecallOS] .env not found - copying template from .env.example"
    cp .env.example .env
    echo "[RecallOS] .env created. Please edit it and set your DEEPSEEK_API_KEY."
    echo
fi

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python3"
fi

echo "[RecallOS] Launching with $PYTHON ..."
exec "$PYTHON" -m streamlit run app.py