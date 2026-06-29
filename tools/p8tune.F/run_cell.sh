#!/bin/bash
# tools/p8tune.F/run_cell.sh
#
# P8-tune.F BoomerAMG/Hypre pattern-only spike (PR-B). Per-cell wrapper
# invoked by spike_array.sbatch for each of the 16 Slurm array tasks.
# Decodes SLURM_ARRAY_TASK_ID -> (CASE, interp_type, coarsen_type) per
# openspec change `p8tune-amg-spike` spec REQ-4 Scenario "4 case × 4
# combo matrix", then invokes the spike binary
# `boomeramg_setup_solve` with the right CLI args.
#
# Per openspec change `p8tune-amg-spike` issue #396 §In Scope §3.
#
# Usage:
#   bash run_cell.sh <NN> <RUN_ID>
#
# Args:
#   NN        ∈ {00..15}  Slurm array task id zero-padded
#   RUN_ID    arbitrary string used as the per-run dir name under
#             /scratch/.../.p8tune.F-runs/<RUN_ID>/
#
# CELL_NN = NN
# CASE_IDX  = NN / 4   →  {keliya, heihe, heihe_x4, heihe_x16}
# COMBO_IDX = NN % 4   →  (interp_type, coarsen_type) pairs:
#   0 → (6, 8)   Hypre default (classical-extended + HMIS)
#   1 → (14, 10) aggressive (extended+i + HMIS)
#   2 → (6, 21)  alt-coarsening (classical-extended + CGC)
#   3 → (8, 8)   fallback (standard + HMIS)
#
# Layout (flat per-cell, mirror P8-tune.D PR-A boundary):
#   ${RUN_DIR}/cell-NN.log    combined stdout+stderr of boomeramg_setup_solve
#   ${RUN_DIR}/cell-NN.time   /usr/bin/time -v output (peak RSS)
#   Slurm-managed (sbatch directives):
#     ${RUN_DIR}/cell-NN.out  Slurm-captured stdout
#     ${RUN_DIR}/cell-NN.err  Slurm-captured stderr
#
# Exit code (per spec REQ-4 marker-emission-as-data-point):
#   0  PASS, or marker emission (AMG_OOM / AMG_SETUP_DIVERGE /
#      AMG_SOLVE_DIVERGE / AMG_WALL_OVERFLOW); markers are valid data
#      points, not failures
#   1  real failure (pre-flight dump_adjacency / fd_color_jacobian
#      shell-out aborted, or boomeramg_setup_solve errored without any
#      recognized marker)
#   2  CLI argument validation failure (caller bug)
#
# Refs:
#   openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md REQ-4
#   issue #396 (P8-tune.F PR-B)

set -uo pipefail

# ---------- CLI ---------------------------------------------------------------
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <NN> <RUN_ID>" >&2
    echo "  NN        ∈ {00..15} (Slurm array task id zero-padded)" >&2
    echo "  RUN_ID    arbitrary string used as the per-run directory name" >&2
    exit 2
fi

NN="$1"
RUN_ID="$2"

# ---------- Whitelist validation ---------------------------------------------
if ! [[ "${NN}" =~ ^[0-9]{2}$ ]]; then
    echo "ERROR: NN must be a 2-digit zero-padded id (00..15), got '${NN}'" >&2
    exit 2
fi
NN_INT=$((10#${NN}))   # force decimal interpretation (avoid octal trap on 08/09)
if [[ ${NN_INT} -lt 0 || ${NN_INT} -gt 15 ]]; then
    echo "ERROR: NN=${NN} out of range [00,15]" >&2
    exit 2
fi

if [[ -z "${RUN_ID}" || "${RUN_ID}" =~ [^A-Za-z0-9._-] ]]; then
    echo "ERROR: RUN_ID must be a non-empty string of [A-Za-z0-9._-], got '${RUN_ID}'" >&2
    exit 2
fi

# ---------- Path layout ------------------------------------------------------
SHUD_OPENMP_ROOT="/scratch/frd_muziyao/SHUD-OpenMP"
TOOL_DIR="${SHUD_OPENMP_ROOT}/tools/p8tune.F"
BASIN_ROOT="${SHUD_OPENMP_ROOT}/SHUD/Basins"
RUN_DIR="${SHUD_OPENMP_ROOT}/.p8tune.F-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

# ---------- Spec REQ-4 cell decoder (4 case × 4 combo matrix) ----------------
# Per spec REQ-4 Scenario "4 case × 4 combo matrix":
#   CASE  = case_for(NN/4)   where case_for(0)=keliya, (1)=heihe,
#                                  (2)=heihe_x4, (3)=heihe_x16
#   COMBO = combo_for(NN%4)  where combo_for(0)=(6,8), (1)=(14,10),
#                                   (2)=(6,21), (3)=(8,8)
CASE_ARR=(keliya heihe heihe_x4 heihe_x16)
INTERP_ARR=(6 14 6 8)
COARSEN_ARR=(8 10 21 8)

CASE_IDX=$((NN_INT / 4))
COMBO_IDX=$((NN_INT % 4))

if [[ ${CASE_IDX} -lt 0 || ${CASE_IDX} -gt 3 ]]; then
    echo "ERROR: derived CASE_IDX=${CASE_IDX} from NN=${NN} out of range [0,3]" >&2
    exit 2
fi
if [[ ${COMBO_IDX} -lt 0 || ${COMBO_IDX} -gt 3 ]]; then
    echo "ERROR: derived COMBO_IDX=${COMBO_IDX} from NN=${NN} out of range [0,3]" >&2
    exit 2
fi

CASE="${CASE_ARR[${CASE_IDX}]}"
INTERP_TYPE="${INTERP_ARR[${COMBO_IDX}]}"
COARSEN_TYPE="${COARSEN_ARR[${COMBO_IDX}]}"

CELL_TAG="${CASE}_interp${INTERP_TYPE}_coarsen${COARSEN_TYPE}"
CELL_LOG="${RUN_DIR}/cell-${NN}.log"
CELL_TIME="${RUN_DIR}/cell-${NN}.time"

# ---------- Identity banner (cell header for parser robustness) -------------
echo "=== CELL_HEADER NN=${NN} ${CELL_TAG} ==="
echo "date_utc:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname:       $(hostname)"
echo "run_id:         ${RUN_ID}"
echo "run_dir:        ${RUN_DIR}"
echo "tool_dir:       ${TOOL_DIR}"
echo "basin_root:     ${BASIN_ROOT}"
echo "cell_log:       ${CELL_LOG}"
echo "cell_time:      ${CELL_TIME}"
echo "case:           ${CASE}      (case_idx=${CASE_IDX})"
echo "interp_type:    ${INTERP_TYPE}  (combo_idx=${COMBO_IDX})"
echo "coarsen_type:   ${COARSEN_TYPE}"
echo ""

# ---------- Pre-flight: confirm .csc + .bin already exist (or let cpp gen) ---
# boomeramg_setup_solve.cpp internally shells out to ./dump_adjacency +
# ./fd_color_jacobian when the .csc + .bin are missing or empty. The
# README convention (§5) runs them ONCE per case outside the 4-combo
# loop to avoid 4× re-generation. We respect that convention by checking
# here + emitting an INFO banner; the cpp will short-circuit if files
# present.
CSC_PATH="${BASIN_ROOT}/${CASE}/${CASE}_adjacency.csc"
JBIN_PATH="${BASIN_ROOT}/${CASE}/${CASE}_numeric_J.bin"
if [[ -s "${CSC_PATH}" && -s "${JBIN_PATH}" ]]; then
    echo "INFO: pre-flight artifacts present (will skip dump_adjacency + fd_color_jacobian inside cpp)"
    echo "INFO:   ${CSC_PATH}  ($(stat -c %s "${CSC_PATH}" 2>/dev/null || stat -f %z "${CSC_PATH}" 2>/dev/null) bytes)"
    echo "INFO:   ${JBIN_PATH} ($(stat -c %s "${JBIN_PATH}" 2>/dev/null || stat -f %z "${JBIN_PATH}" 2>/dev/null) bytes)"
else
    echo "INFO: pre-flight artifacts missing or empty; cpp will shell-out to dump_adjacency + fd_color_jacobian"
fi
echo ""

# ---------- Time wrapper (peak RSS for cell_summary KV peak_rss_bytes) ------
# /usr/bin/time -v Maximum resident set size populates the cell_summary
# KV `peak_rss_bytes` field. The cpp internally calls getrusage too, but
# the wrapper measurement is a robust cross-check.
TIME_BIN="/usr/bin/time"
if [[ ! -x "${TIME_BIN}" ]]; then
    echo "WARN: ${TIME_BIN} not found; RSS measurement will be missing for this cell" >&2
    TIME_BIN=""
fi

# ---------- Invoke boomeramg_setup_solve ------------------------------------
# Per boomeramg_setup_solve.cpp:
#   - expects to be invoked from tools/p8tune.F/ (resolves ./dump_adjacency +
#     ./fd_color_jacobian symlinks)
#   - --basin-root defaults to ../../SHUD/Basins; we pass explicit absolute
#     path for robustness
#   - reads <basin_root>/<case>/<case>_{numeric_J.bin,adjacency.csc}
#
# Output: combined stdout+stderr captured via `tee` into ${CELL_LOG} so the
# Slurm-managed cell-NN.out/.err stays a simple mirror (sbatch directives
# already redirect stdout/stderr to those paths automatically; tee gives us
# a separate authoritative .log for the aggregator parser).
cd "${TOOL_DIR}" || { echo "ERROR: cannot cd to ${TOOL_DIR}" >&2; exit 1; }

${TIME_BIN:+${TIME_BIN} -v --output="${CELL_TIME}"} \
    ./boomeramg_setup_solve \
        --case "${CASE}" \
        --interp-type "${INTERP_TYPE}" \
        --coarsen-type "${COARSEN_TYPE}" \
        --basin-root "${BASIN_ROOT}" \
    > >(tee "${CELL_LOG}") 2>&1
RC=$?

# ---------- Cell footer + marker classification ------------------------------
echo ""
echo "=== CELL_FOOTER NN=${NN} ${CELL_TAG} done exit=${RC} ==="

# Spec REQ-4: marker emission = valid data point. If the cpp printed any
# of the 4 AMG_*_DETECTED markers, classify as a clean data-point exit
# (rc=0). The cpp already returns 0 in those branches; we just verify +
# log the classification for human-readable cell log.
if grep -q '^MARKER:AMG_OOM_DETECTED' "${CELL_LOG}" 2>/dev/null; then
    echo "INFO: MARKER:AMG_OOM_DETECTED present → verdict_class=AMG_OOM"
fi
if grep -q '^MARKER:AMG_SETUP_DIVERGE_DETECTED' "${CELL_LOG}" 2>/dev/null; then
    echo "INFO: MARKER:AMG_SETUP_DIVERGE_DETECTED present → verdict_class=AMG_SETUP_DIVERGE"
fi
if grep -q '^MARKER:AMG_SOLVE_DIVERGE_DETECTED' "${CELL_LOG}" 2>/dev/null; then
    echo "INFO: MARKER:AMG_SOLVE_DIVERGE_DETECTED present → verdict_class=AMG_SOLVE_DIVERGE"
fi
if grep -q '^MARKER:AMG_WALL_OVERFLOW_DETECTED' "${CELL_LOG}" 2>/dev/null; then
    echo "INFO: MARKER:AMG_WALL_OVERFLOW_DETECTED present → verdict_class=AMG_WALL_OVERFLOW"
fi

# Per spec REQ-4 marker-as-data-point: marker emission → exit 0 even if
# the cpp returned non-zero due to mid-marker abort. The cpp design
# (boomeramg_setup_solve.cpp:484 emit_cell_summary) returns 0 in those
# branches, so RC=0 is the expected case; we propagate RC unchanged.
exit ${RC}
