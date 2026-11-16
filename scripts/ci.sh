#!/usr/bin/env bash
# Local CI gate for pm1: unit tests + benchmark-vocabulary leak check.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
"$PY" -m pytest -q -p no:cacheprovider tests
extra=()
[ -n "${PM1_VOCAB_EXTRA:-}" ] && [ -f "$PM1_VOCAB_EXTRA" ] && extra+=(--extra "$PM1_VOCAB_EXTRA")
"$PY" scripts/check_vocab_leak.py "${extra[@]}"
