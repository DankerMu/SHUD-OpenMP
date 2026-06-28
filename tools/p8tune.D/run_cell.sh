#!/bin/bash
# tools/p8tune.D/run_cell.sh
#
# P8-tune.D KLU pattern-only spike (PR-A). Per-cell wrapper invoked by
# spike_array.sbatch for each of the 16 Slurm array tasks. Re-uses the
# 3-stage pipeline driver `spike_run.sh` authored in PR-0, but
# re-labels its outputs into the flat per-cell layout
# `cell-NN.{out,err,log,J.bin}` mandated by spec REQ-7 PR-A boundary L218.
#
# Per openspec change `p8tune-klu-spike` tasks.md §2.5.
#
# Usage:
#   bash run_cell.sh <case> <ordering> <btf> <NN> <RUN_ID>
#
# Args (whitelisted; refuses bogus values with exit 2 — mirrors PR-0
# spike_run.sh whitelist patterns):
#   case      ∈ {keliya, heihe, heihe_x4, heihe_x16}   per spec REQ-4
#   ordering  ∈ {natural, amd, colamd}                  per spec REQ-4
#   btf       ∈ {0, 1}                                  per spec REQ-4
#   NN        ∈ {00..15}                                Slurm array task id zero-padded
#   RUN_ID    arbitrary string used as the per-run directory name
#
# Layout (per RUN_ID, flat per spec REQ-7 PR-A boundary L218):
#   ${RUN_DIR}/cell-NN.log    combined stdout+stderr of the 3-stage pipeline
#   ${RUN_DIR}/cell-NN.J.bin  per-cell numeric J copy (for PR-B aggregator)
#   ${RUN_DIR}/cell-NN.time   /usr/bin/time -v output (peak RSS for RSS axis)
#
# Note: cell-NN.out and cell-NN.err are written DIRECTLY by Slurm per the
# spike_array.sbatch --output / --error directives; this wrapper does NOT
# need to manage them — sbatch redirects the stdout/stderr of the bash
# invocation into them automatically.
#
# Exit code (per spec REQ-5 OOM-as-data-point):
#   0  all 3 stages PASS, OR klu_analyze_factor emitted KLU_OOM_DETECTED
#      (OOM is decisive data, not failure)
#   1  dump_adjacency / fd_color_jacobian failed, OR klu_analyze_factor
#      exited non-zero AND no KLU_OOM_DETECTED line was emitted
#   2  CLI argument validation failure (caller bug; should never happen
#      under spike_array.sbatch dispatch)
#
# Refs:
#   openspec/changes/p8tune-klu-spike/tasks.md §2.5
#   openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-4, REQ-5, REQ-7

set -uo pipefail

# ---------- CLI ---------------------------------------------------------------
if [[ $# -ne 5 ]]; then
    echo "usage: $0 <case> <ordering> <btf> <NN> <RUN_ID>" >&2
    echo "  case      ∈ {keliya, heihe, heihe_x4, heihe_x16}" >&2
    echo "  ordering  ∈ {natural, amd, colamd}" >&2
    echo "  btf       ∈ {0, 1}" >&2
    echo "  NN        ∈ {00..15} (Slurm array task id zero-padded)" >&2
    echo "  RUN_ID    arbitrary string used as the per-run directory name" >&2
    exit 2
fi

CASE="$1"
ORDERING="$2"
BTF="$3"
NN="$4"
RUN_ID="$5"

# ---------- Whitelist validation (mirror PR-0 spike_run.sh) ------------------
case "${CASE}" in
    keliya|heihe|heihe_x4|heihe_x16) ;;
    *) echo "ERROR: case must be one of {keliya, heihe, heihe_x4, heihe_x16}, got '${CASE}'" >&2; exit 2 ;;
esac

case "${ORDERING}" in
    natural|amd|colamd) ;;
    *) echo "ERROR: ordering must be natural|amd|colamd, got '${ORDERING}'" >&2; exit 2 ;;
esac

case "${BTF}" in
    0|1) ;;
    *) echo "ERROR: btf must be 0|1, got '${BTF}'" >&2; exit 2 ;;
esac

if ! [[ "${NN}" =~ ^[0-9]{2}$ ]]; then
    echo "ERROR: NN must be a 2-digit zero-padded id (00..15), got '${NN}'" >&2
    exit 2
fi

if [[ -z "${RUN_ID}" || "${RUN_ID}" =~ [^A-Za-z0-9._-] ]]; then
    echo "ERROR: RUN_ID must be a non-empty string of [A-Za-z0-9._-], got '${RUN_ID}'" >&2
    exit 2
fi

# ---------- Path layout ------------------------------------------------------
SHUD_OPENMP_ROOT="/scratch/frd_muziyao/SHUD-OpenMP"
TOOL_DIR="${SHUD_OPENMP_ROOT}/tools/p8tune.D"
BASIN_ROOT="${SHUD_OPENMP_ROOT}/SHUD/Basins"
RUN_DIR="${SHUD_OPENMP_ROOT}/.p8tune.D-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

CELL_TAG="${CASE}_${ORDERING}_btf${BTF}"
CELL_LOG="${RUN_DIR}/cell-${NN}.log"
CELL_J_BIN="${RUN_DIR}/cell-${NN}.J.bin"
CELL_TIME="${RUN_DIR}/cell-${NN}.time"

# Per-cell scratch staging dir (avoids cross-cell collision on the shared
# numeric_J.bin / adjacency.csc artifacts that spike_run.sh writes to its
# OUTPUT_DIR). Each cell gets its own subdir; basin-level artifacts are
# regenerated per cell.
CELL_STAGE="${RUN_DIR}/_stage/cell-${NN}"
mkdir -p "${CELL_STAGE}"

# ---------- Identity banner --------------------------------------------------
echo "=== run_cell.sh NN=${NN} ${CELL_TAG} ==="
echo "date_utc:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname:    $(hostname)"
echo "run_id:      ${RUN_ID}"
echo "run_dir:     ${RUN_DIR}"
echo "tool_dir:    ${TOOL_DIR}"
echo "basin_root:  ${BASIN_ROOT}"
echo "cell_stage:  ${CELL_STAGE}"
echo "cell_log:    ${CELL_LOG}"
echo "cell_j_bin:  ${CELL_J_BIN}"
echo "cell_time:   ${CELL_TIME}"
echo ""

# ---------- Pipeline invocation (delegated to spike_run.sh) -------------------
# spike_run.sh:
#   - reads OUTPUT_DIR + BASIN_ROOT from env
#   - expects ./dump_adjacency / ./fd_color_jacobian / ./klu_analyze_factor
#     binaries in cwd; we cd into TOOL_DIR for that.
#   - writes its cell-<case>_<ordering>_btf<btf>.log to ${OUTPUT_DIR}
#   - writes adjacency / numeric_J binaries to the cwd of the binaries
#     (which is TOOL_DIR after our cd). To keep cells isolated we point
#     OUTPUT_DIR at the per-cell stage dir; numeric_J / adjacency get
#     regenerated per cell.
#
# Time wrapper: /usr/bin/time -v captures Maximum resident set size for the
# RSS axis verdict (spec REQ-5 Scenario "RSS axis threshold"). Output goes
# to a separate cell-NN.time file so the cell-NN.log stays parseable by
# the aggregator (which greps for klu_analyze_factor KV lines + the
# KLU_OOM_DETECTED marker).
TIME_BIN="/usr/bin/time"
if [[ ! -x "${TIME_BIN}" ]]; then
    echo "WARN: ${TIME_BIN} not found; RSS measurement will be missing for this cell" >&2
    TIME_BIN=""
fi

cd "${TOOL_DIR}" || { echo "ERROR: cannot cd to ${TOOL_DIR}" >&2; exit 1; }

OUTPUT_DIR="${CELL_STAGE}" \
BASIN_ROOT="${BASIN_ROOT}" \
${TIME_BIN:+${TIME_BIN} -v --output="${CELL_TIME}"} \
    bash "${TOOL_DIR}/spike_run.sh" "${CASE}" "${ORDERING}" "${BTF}" \
    > >(tee "${CELL_LOG}") 2>&1
RC=$?

# ---------- Per-cell artifact promotion --------------------------------------
# spike_run.sh writes its combined log to ${OUTPUT_DIR}/cell-${CELL_TAG}.log;
# we already captured the same content via tee into ${CELL_LOG}, but as a
# belt-and-suspenders promote the spike_run log too (in case spike_run
# changes its layout in a future revision):
SPIKE_LOG="${CELL_STAGE}/cell-${CELL_TAG}.log"
if [[ -f "${SPIKE_LOG}" && ! -s "${CELL_LOG}" ]]; then
    cp -p "${SPIKE_LOG}" "${CELL_LOG}"
fi

# Promote numeric_J binary to cell-NN.J.bin for PR-B aggregator archival
# (tasks §2.4 + §2.8 archive cell-*.J.bin alongside cell-*.{log,out,err}).
# fd_color_jacobian writes <case>_numeric_J.bin into the cwd of the binary
# (= TOOL_DIR after our cd). Locate it via a deterministic glob.
NUMERIC_J="${TOOL_DIR}/${CASE}_numeric_J.bin"
if [[ -f "${NUMERIC_J}" ]]; then
    cp -p "${NUMERIC_J}" "${CELL_J_BIN}"
else
    # Fall back: maybe fd_color_jacobian honored OUTPUT_DIR. Search the
    # cell stage for any *_numeric_J.bin to be defensive.
    FOUND_J=$(find "${CELL_STAGE}" "${TOOL_DIR}" -maxdepth 2 -name '*_numeric_J.bin' 2>/dev/null | head -n 1)
    if [[ -n "${FOUND_J}" ]]; then
        cp -p "${FOUND_J}" "${CELL_J_BIN}"
    else
        echo "WARN: numeric J binary not found for ${CELL_TAG}; aggregator will be missing cell-${NN}.J.bin" >&2
    fi
fi

# ---------- Exit semantics ---------------------------------------------------
# OOM-as-data-point: if klu_analyze_factor printed KLU_OOM_DETECTED, the
# cell counts as a clean PASS-as-Slurm-task (FAIL-on-RSS-axis is decided by
# the aggregator). spike_run.sh already returns 0 in that case per PR-0
# design; we just propagate.
echo ""
echo "=== run_cell.sh NN=${NN} ${CELL_TAG} done exit=${RC} ==="

if grep -q '^KLU_OOM_DETECTED ' "${CELL_LOG}" 2>/dev/null; then
    echo "INFO: KLU_OOM_DETECTED reported for ${CELL_TAG}; aggregator will classify as rss_overflow"
fi

exit ${RC}
