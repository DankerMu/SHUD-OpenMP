#!/bin/bash
# tools/p9.spot/run_a5_for_spot.sh
#
# PR-Z1 — A5 hydrology-acceptance gate wrapper for the 2-cell heihe_x4
# spot check. Reads cell-0-baseline.output/ (reference) + cell-1-p9.output/
# (candidate) from a completed p9_spot_array run and invokes the A5 CLI
# via `uv --directory tools/a5 run a5 ...`.
#
# Usage:
#   bash run_a5_for_spot.sh <run_dir>
#
# Args:
#   run_dir   directory populated by p9_spot_array.sbatch, containing:
#             - cell-0-baseline.output/   (SHUD output tree, reference)
#             - cell-1-p9.output/         (SHUD output tree, candidate)
#             - cell-*.out / .err         (per-cell logs; NOT consumed by A5)
#
# Exit code:
#   0  A5 emitted PASS verdict (candidate ~= reference)
#   1  A5 emitted FAIL verdict (candidate fails hydrology gate)
#   2  malformed thresholds YAML
#   3  I/O error reading SHUD outputs
#   >3 harness precondition failure (RUN_DIR missing / cell trees absent)
#
# Emits (per A5 contract):
#   ${RUN_DIR}/a5-report/a5_metrics.json
#   ${RUN_DIR}/a5-report/a5_verdict.md
#   ${RUN_DIR}/a5-marker.log            (captured MARKER block + stdout)
#
# Refs:
#   - PR-Z1 brief §Stage 2 (harness scripts)
#   - tools/a5/README.md (A5 CLI contract)

set -uo pipefail

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
    echo "usage: $0 <run_dir>" >&2
    echo "  run_dir  directory with cell-0-baseline.output/ + cell-1-p9.output/" >&2
    exit 4
fi

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[p9-spot-a5] FATAL: run_dir not found: ${RUN_DIR}" >&2
    exit 4
fi

# Resolve repo root (parent of tools/p9.spot/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
A5_DIR="${REPO_ROOT}/tools/a5"

if [[ ! -d "${A5_DIR}" ]]; then
    echo "[p9-spot-a5] FATAL: A5 tool dir not found: ${A5_DIR}" >&2
    exit 4
fi

REF_TREE="${RUN_DIR}/cell-0-baseline.output"
CAND_TREE="${RUN_DIR}/cell-1-p9.output"

if [[ ! -d "${REF_TREE}" ]]; then
    echo "[p9-spot-a5] FATAL: reference tree missing: ${REF_TREE}" >&2
    echo "[p9-spot-a5]        did cell 0 (baseline) fail? check ${RUN_DIR}/cell-0-baseline.err" >&2
    exit 4
fi
if [[ ! -d "${CAND_TREE}" ]]; then
    echo "[p9-spot-a5] FATAL: candidate tree missing: ${CAND_TREE}" >&2
    echo "[p9-spot-a5]        did cell 1 (p9) fail? check ${RUN_DIR}/cell-1-p9.err" >&2
    exit 4
fi

A5_OUT="${RUN_DIR}/a5-report"
A5_MARKER_LOG="${RUN_DIR}/a5-marker.log"
mkdir -p "${A5_OUT}"

echo "[p9-spot-a5] repo_root:    ${REPO_ROOT}"
echo "[p9-spot-a5] a5_dir:       ${A5_DIR}"
echo "[p9-spot-a5] run_dir:      ${RUN_DIR}"
echo "[p9-spot-a5] reference:    ${REF_TREE}"
echo "[p9-spot-a5] candidate:    ${CAND_TREE}"
echo "[p9-spot-a5] a5_out:       ${A5_OUT}"
echo "[p9-spot-a5] marker_log:   ${A5_MARKER_LOG}"
echo ""

# Ensure `uv` is on PATH. If not, fall back to $HOME/.local/bin (Slurm
# jobs on the server typically populate this via .bashrc; login-node uv
# install lives there).
if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "${HOME}/.local/bin/uv" ]]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        echo "[p9-spot-a5] FATAL: 'uv' not on PATH and not at \$HOME/.local/bin/uv" >&2
        echo "[p9-spot-a5]        install uv per https://docs.astral.sh/uv/getting-started/installation/" >&2
        exit 4
    fi
fi

# Sync A5 deps (idempotent — noop if .venv already in sync).
echo "[p9-spot-a5] uv sync (A5)..."
uv --directory "${A5_DIR}" sync 2>&1 | tail -3

# Invoke A5. tee stdout to marker log so the aggregator can grep it later.
echo "[p9-spot-a5] invoking A5..."
set +e
uv --directory "${A5_DIR}" run a5 \
    --reference "${REF_TREE}" \
    --candidate "${CAND_TREE}" \
    --config "${A5_DIR}/config/a5_thresholds.default.yaml" \
    --case-name heihe_x4 \
    --out "${A5_OUT}" \
    2>&1 | tee "${A5_MARKER_LOG}"
A5_RC=${PIPESTATUS[0]}
set -e

echo ""
echo "[p9-spot-a5] A5 exit=${A5_RC}"
echo "[p9-spot-a5] verdict artifacts:"
[[ -f "${A5_OUT}/a5_metrics.json" ]] && echo "  - ${A5_OUT}/a5_metrics.json"
[[ -f "${A5_OUT}/a5_verdict.md" ]]   && echo "  - ${A5_OUT}/a5_verdict.md"
[[ -f "${A5_MARKER_LOG}" ]]          && echo "  - ${A5_MARKER_LOG}"

exit ${A5_RC}
