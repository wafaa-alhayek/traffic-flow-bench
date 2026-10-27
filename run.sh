#!/usr/bin/env bash
# One command: ./run.sh [--download | --data-url URL | --release-root DIR] [--panel PANEL ...]
set -euo pipefail
cd "$(dirname "$0")"
python3 src/pipeline.py "$@"
