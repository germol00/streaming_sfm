#!/usr/bin/env bash
# Convenience wrapper for the PAC experiment manifest runner.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" "${REPO_ROOT}/scripts/run_pac_experiments.py" "$@"
