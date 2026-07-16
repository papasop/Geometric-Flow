#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_external_gpt2_validation.py --mode "${1:-smoke}" "${@:2}"
