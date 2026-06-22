#!/bin/bash
# .s6b-1-runs/scan_nan.sh -- thin wrapper around scan_nan.py
# Usage: ./scan_nan.sh <run_output_dir>
# Per CLAUDE.md project rule: prefer `uv` for Python. Falls back to a
# direct python3 invocation on hosts where uv is not provisioned
# (e.g. server compute nodes) — guarded by a check that `numpy` is
# already importable from the system Python so we never silently run
# without numpy.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if command -v uv >/dev/null 2>&1; then
    exec uv run --with numpy python3 "$HERE/scan_nan.py" "$@"
fi
python3 -c "import numpy" >/dev/null 2>&1 || {
    echo "scan_nan.sh: neither uv nor system numpy available" >&2
    exit 2
}
exec python3 "$HERE/scan_nan.py" "$@"
