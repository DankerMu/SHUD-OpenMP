#!/bin/bash
# tools/release_v1.0_omp_scaling/run_a5_scaling.sh
#
# Release v1.0 OMP scaling -- A5 hydrology-acceptance gate wrapper.
# For each N in {2, 4, 8, 16}, invokes the A5 CLI comparing the
# corresponding cell output tree against the N=1 reference tree from
# the same RUN_DIR. Emits one A5 report per N under a5-report-nthreads-<N>/.
#
# Usage:
#   bash run_a5_scaling.sh <run_dir>
#
# Args:
#   run_dir   directory populated by scaling_array.sbatch, containing:
#             - cell-0-nthreads-1.output/   (SHUD output tree, reference)
#             - cell-1-nthreads-2.output/   (candidate for N=2)
#             - cell-2-nthreads-4.output/   (candidate for N=4)
#             - cell-3-nthreads-8.output/   (candidate for N=8)
#             - cell-4-nthreads-16.output/  (candidate for N=16)
#             - cell-*.out / .err           (per-cell logs; NOT consumed by A5)
#
# Exit code:
#   0  all four A5 comparisons dispatched (individual PASS/FAIL captured
#      in per-N a5-marker.log; downstream aggregator reads a5_metrics.json)
#   4  harness precondition failure (RUN_DIR missing / reference tree absent)
#
# Emits (per N):
#   ${RUN_DIR}/a5-report-nthreads-<N>/a5_metrics.json
#   ${RUN_DIR}/a5-report-nthreads-<N>/a5_verdict.md
#   ${RUN_DIR}/a5-report-nthreads-<N>/a5-marker.log
#
# Refs:
#   - RELEASE.md
#   - tools/p9.spot/run_a5_for_spot.sh (template)
#   - tools/a5/README.md (A5 CLI contract)

set -uo pipefail

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
    echo "usage: $0 <run_dir>" >&2
    echo "  run_dir  directory with cell-{0..4}-nthreads-{1,2,4,8,16}.output/" >&2
    exit 4
fi

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[release-v1-scaling-a5] FATAL: run_dir not found: ${RUN_DIR}" >&2
    exit 4
fi

# Resolve repo root (parent of tools/release_v1.0_omp_scaling/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
A5_DIR="${REPO_ROOT}/tools/a5"

if [[ ! -d "${A5_DIR}" ]]; then
    echo "[release-v1-scaling-a5] FATAL: A5 tool dir not found: ${A5_DIR}" >&2
    exit 4
fi

REF_TREE="${RUN_DIR}/cell-0-nthreads-1.output"
if [[ ! -d "${REF_TREE}" ]]; then
    echo "[release-v1-scaling-a5] FATAL: reference tree missing: ${REF_TREE}" >&2
    echo "[release-v1-scaling-a5]        did cell 0 (N=1) fail? check ${RUN_DIR}/cell-0-nthreads-1.err" >&2
    exit 4
fi

echo "[release-v1-scaling-a5] repo_root:  ${REPO_ROOT}"
echo "[release-v1-scaling-a5] a5_dir:     ${A5_DIR}"
echo "[release-v1-scaling-a5] run_dir:    ${RUN_DIR}"
echo "[release-v1-scaling-a5] reference:  ${REF_TREE}"
echo ""

# Ensure `uv` is on PATH.
if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "${HOME}/.local/bin/uv" ]]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        echo "[release-v1-scaling-a5] FATAL: 'uv' not on PATH and not at \$HOME/.local/bin/uv" >&2
        echo "[release-v1-scaling-a5]        install per https://docs.astral.sh/uv/getting-started/installation/" >&2
        exit 4
    fi
fi

# Sync A5 deps once up front (idempotent -- noop if .venv already in sync).
echo "[release-v1-scaling-a5] uv sync (A5)..."
uv --directory "${A5_DIR}" sync 2>&1 | tail -3
echo ""

# For each non-baseline thread count, invoke A5 against the N=1 reference.
# Iteration order mirrors the sbatch array indexing so per-N artifacts
# align with cell-<TASK>-nthreads-<N>.output/.
NTHREADS_LIST=(1 2 4 8 16)
OVERALL_RC=0

for TASK in 1 2 3 4; do
    N="${NTHREADS_LIST[${TASK}]}"
    CAND_TREE="${RUN_DIR}/cell-${TASK}-nthreads-${N}.output"
    OUT="${RUN_DIR}/a5-report-nthreads-${N}"
    A5_MARKER_LOG="${OUT}/a5-marker.log"
    mkdir -p "${OUT}"

    echo "[release-v1-scaling-a5] --- N=${N} ---"
    echo "[release-v1-scaling-a5] candidate:  ${CAND_TREE}"
    echo "[release-v1-scaling-a5] a5_out:     ${OUT}"

    if [[ ! -d "${CAND_TREE}" ]]; then
        echo "[release-v1-scaling-a5] WARN: candidate tree missing for N=${N}: ${CAND_TREE}" >&2
        echo "MARKER:A5_ERROR nthreads=${N} reason=candidate_tree_missing" | tee "${A5_MARKER_LOG}"
        OVERALL_RC=1
        echo ""
        continue
    fi

    set +e
    uv --directory "${A5_DIR}" run a5 \
        --reference "${REF_TREE}" \
        --candidate "${CAND_TREE}" \
        --config "${A5_DIR}/config/a5_thresholds.default.yaml" \
        --case-name heihe_x4 \
        --out "${OUT}" \
        2>&1 | tee "${A5_MARKER_LOG}"
    A5_RC=${PIPESTATUS[0]}
    set -e

    if [[ ${A5_RC} -ne 0 && ${A5_RC} -ne 1 ]]; then
        # RC=0 -> PASS, RC=1 -> FAIL both are captured verdicts.
        # RC=2/3/other -> genuine A5 harness error.
        echo "MARKER:A5_ERROR nthreads=${N} rc=${A5_RC}" >> "${A5_MARKER_LOG}"
        OVERALL_RC=1
    fi

    echo "[release-v1-scaling-a5] N=${N} a5_exit=${A5_RC}"
    echo ""
done

echo "[release-v1-scaling-a5] all N dispatched; overall_rc=${OVERALL_RC}"
exit ${OVERALL_RC}
