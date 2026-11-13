#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python guitar_remover.py
fi
exec python3 guitar_remover.py
