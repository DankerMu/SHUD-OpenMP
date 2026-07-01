#!/bin/bash
# tools/release_v1.0_omp_scaling/run_scaling_cell.sh
#
# Release v1.0 OMP scaling — single-cell heihe_x4 SPGMR runner. Accepts
# an explicit thread count from the sbatch array dispatcher (or a local
# smoke wrapper); enforces production env (SPGMR + unset research
# hooks); verifies 90-day cfg.para truncation; invokes ./shud_omp;
# copies the SHUD *.out/ tree next to the sidecar logs (for downstream
# A5 comparison); and emits a per-cell MARKER:CELL_SCALING_SUMMARY
# block.
#
# Per CLAUDE.md 项目级铁律 "所有 case <=90 天截断". Mirrors
# tools/p9.spot/run_spot_cell.sh but:
#   - Accepts (N, RUN_LABEL) instead of (LABEL, RELTOL, RUN_LABEL)
#   - Always production env — unsets research hooks, sets SPGMR + OMP
#     binding
#   - No research env-hook toggle path
#   - Emits MARKER:CELL_SCALING_SUMMARY (scaling-specific format)
#
# Usage:
#   bash run_scaling_cell.sh <nthreads> <run_label>
#
# Args:
#   nthreads    integer thread count (1|2|4|8|16 in release v1.0 scope)
#   run_label   identifier used in sidecar filenames, e.g., '0-nthreads-1'
#
# Exit code:
#   0  PASS -- shud_omp exited 0 + summary emitted + output copied
#   2  precondition failed (bad args, cfg.para not 90-day, missing basin)
#   *  non-zero -- propagated from shud_omp
#
# Refs:
#   - RELEASE.md (v1.0 production manifest)
#   - tools/p9.spot/run_spot_cell.sh (template)
#   - CLAUDE.md 项目级铁律 90-day truncation

set -uo pipefail

# ---------- CLI ------------------------------------------------------------
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <nthreads> <run_label>" >&2
    echo "  nthreads    integer thread count" >&2
    echo "  run_label   identifier, e.g., '0-nthreads-1'" >&2
    exit 2
fi

N="$1"
RUN_LABEL="$2"

# Sanity-check thread count is a positive integer.
if ! [[ "${N}" =~ ^[0-9]+$ ]] || [[ "${N}" -lt 1 ]]; then
    echo "[release-v1-scaling] FATAL: nthreads='${N}' is not a positive integer" >&2
    exit 2
fi

# Release scaling scope: heihe_x4 only (hardcoded per brief).
CELL="heihe_x4"
PROJECT_NAME="heihe_x4"

# ---------- Environment ----------------------------------------------------
# Production solver selector (defense-in-depth alongside sbatch).
export SHUD_LINSOL=spgmr

# Enforce production behavior: strip all research env hooks. This
# guarantees the scaling numbers reflect the production baseline, not
# any research configuration a prior task may have leaked in.
unset SHUD_AMG_TOL
unset SHUD_CVODE_EPSLIN
unset SHUD_CVODE_RELTOL

export OMP_NUM_THREADS="${N}"

# OMP binding — defense-in-depth. sbatch also sets these; local Mac
# smoke path may not, so we default them here too.
: "${OMP_PROC_BIND:=close}"
: "${OMP_PLACES:=cores}"
export OMP_PROC_BIND OMP_PLACES

# ---------- Locate cell artifacts -----------------------------------------
# Auto-cd to Basins/heihe_x4 when invoked from SHUD/ root (sbatch cwd).
if [[ -d "Basins/${CELL}" && ! -d "input/${PROJECT_NAME}" ]]; then
    cd "Basins/${CELL}" || {
        echo "[release-v1-scaling] FATAL: cannot cd to Basins/${CELL}" >&2
        exit 2
    }
fi

if [[ ! -d "input/${PROJECT_NAME}" ]]; then
    echo "[release-v1-scaling] FATAL: input/${PROJECT_NAME}/ not found under $(pwd) -- heihe_x4 not deployed" >&2
    exit 2
fi

CFG_PARA="$(find "input/${PROJECT_NAME}" -name '*.cfg.para' 2>/dev/null | head -n 1)"
if [[ -z "${CFG_PARA}" || ! -f "${CFG_PARA}" ]]; then
    echo "[release-v1-scaling] FATAL: no *.cfg.para under input/${PROJECT_NAME}/" >&2
    exit 2
fi

# ---------- 90-day truncation precondition ---------------------------------
END_START_DELTA=$(awk '
    /^[[:space:]]*(START|END)[[:space:]]/ {
        kv[$1] = $2
    }
    END {
        if (!("START" in kv) || !("END" in kv)) {
            print "NA"
        } else {
            print (kv["END"] - kv["START"]) + 0
        }
    }' "${CFG_PARA}")

if [[ "${END_START_DELTA}" != "90" ]]; then
    echo "[release-v1-scaling] FATAL: 90-day truncation precondition failed for case=${CELL} END-START=${END_START_DELTA}" >&2
    echo "[release-v1-scaling]        cfg.para=${CFG_PARA}" >&2
    exit 2
fi

# ---------- Banner --------------------------------------------------------
echo "=== run_scaling_cell.sh cell=${CELL} run_label=${RUN_LABEL} ==="
echo "date_utc:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname:           $(hostname)"
echo "cwd:                $(pwd)"
echo "cell:               ${CELL}"
echo "project_name:       ${PROJECT_NAME}"
echo "cfg_para:           ${CFG_PARA}"
echo "end_start_delta:    ${END_START_DELTA} (90-day precondition PASS)"
echo "shud_linsol:        ${SHUD_LINSOL}"
echo "omp_num_threads:    ${OMP_NUM_THREADS}"
echo "omp_proc_bind:      ${OMP_PROC_BIND}"
echo "omp_places:         ${OMP_PLACES}"
echo "shud_amg_tol:       ${SHUD_AMG_TOL:-<unset:production>}"
echo "shud_cvode_epslin:  ${SHUD_CVODE_EPSLIN:-<unset:production>}"
echo "shud_cvode_reltol:  ${SHUD_CVODE_RELTOL:-<unset:production>}"

# ---------- Resolve RUN_DIR for sidecars ----------------------------------
if [[ -n "${RUN_DIR:-}" ]]; then
    RUN_DIR_RESOLVED="${RUN_DIR}"
else
    RUN_DIR_RESOLVED="$(pwd)/.release-v1-scaling-local"
fi
mkdir -p "${RUN_DIR_RESOLVED}"

CELL_OUT="${RUN_DIR_RESOLVED}/cell-${RUN_LABEL}.out"
CELL_ERR="${RUN_DIR_RESOLVED}/cell-${RUN_LABEL}.err"
CELL_OUTPUT_TREE="${RUN_DIR_RESOLVED}/cell-${RUN_LABEL}.output"

echo "run_dir:            ${RUN_DIR_RESOLVED}"
echo "cell_out:           ${CELL_OUT}"
echo "cell_err:           ${CELL_ERR}"
echo "cell_output_tree:   ${CELL_OUTPUT_TREE}"
echo ""

# ---------- Binary preflight ----------------------------------------------
if [[ -n "${SHUD_BIN:-}" && -x "${SHUD_BIN}" ]]; then
    SHUD_BIN_RESOLVED="${SHUD_BIN}"
elif [[ -x "../../shud_omp" ]]; then
    SHUD_BIN_RESOLVED="../../shud_omp"
elif [[ -x "./shud_omp" ]]; then
    SHUD_BIN_RESOLVED="./shud_omp"
else
    echo "[release-v1-scaling] FATAL: shud_omp not found (checked \$SHUD_BIN, ../../shud_omp, ./shud_omp) under $(pwd)" >&2
    exit 2
fi
echo "shud_bin:           ${SHUD_BIN_RESOLVED}"
echo ""

# ---------- Capture wall + invoke shud_omp --------------------------------
START_WALL=$(date +%s)
set +e
"${SHUD_BIN_RESOLVED}" "${PROJECT_NAME}" \
    > >(tee "${CELL_OUT}") \
    2> >(tee "${CELL_ERR}" >&2)
RC=$?
wait
set -e
END_WALL=$(date +%s)
WALL_TOTAL=$((END_WALL - START_WALL))

# ---------- Copy SHUD *.out/ tree to RUN_DIR for A5 consumption -----------
# SHUD writes to output/<project>.out/ under the basin cwd.
SHUD_OUTPUT_SRC="output/${PROJECT_NAME}.out"
if [[ ${RC} -eq 0 && -d "${SHUD_OUTPUT_SRC}" ]]; then
    rm -rf "${CELL_OUTPUT_TREE}"
    cp -R "${SHUD_OUTPUT_SRC}" "${CELL_OUTPUT_TREE}"
    echo "[release-v1-scaling] copied ${SHUD_OUTPUT_SRC}/ -> ${CELL_OUTPUT_TREE}/"
else
    echo "[release-v1-scaling] WARN: shud_omp exit=${RC} or ${SHUD_OUTPUT_SRC} missing; A5 input tree NOT staged" >&2
fi

# ---------- Extract CVODE Final Statistics --------------------------------
# CVODE prints "KEY = VAL" tokens in the Final Statistics footer. Same
# parse shape as tools/p9.spot/run_spot_cell.sh; kept minimal here for
# scaling table needs (nst / ncfn / ncfl).
extract_cvode_stat() {
    local key="$1"
    awk -v key="${key}" '
        {
            for (i=1; i<=NF; i++) {
                if ($i == key) {
                    if (i+1 <= NF && $(i+1) == "=") {
                        if (i+2 <= NF) {
                            print $(i+2)
                            exit
                        }
                    } else if (i+1 <= NF && $(i+1) ~ /^=/) {
                        v = substr($(i+1), 2)
                        if (v != "") {
                            print v
                            exit
                        }
                        if (i+2 <= NF) {
                            print $(i+2)
                            exit
                        }
                    }
                }
            }
        }
    ' "${CELL_OUT}" 2>/dev/null
}

set +e
NST=$(extract_cvode_stat nst)
NCFN=$(extract_cvode_stat ncfn)
NCFL=$(extract_cvode_stat ncfl)
set -e

# ---------- Wall breakdown: setup vs solve --------------------------------
# SHUD does not (currently) print a canonical "setup" vs "solve" wall
# breakdown. Fallback: report both as WALL_TOTAL so downstream analysis
# is still well-formed. If a future SHUD version emits the breakdown,
# a token like "setup_wall_sec=..." or "solve_wall_sec=..." can be
# grepped here.
set +e
WALL_SETUP=$(grep -oE 'setup_wall_sec=[0-9.]+' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk -F= '{print $2}')
WALL_SOLVE=$(grep -oE 'solve_wall_sec=[0-9.]+' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk -F= '{print $2}')
set -e
: "${WALL_SETUP:=${WALL_TOTAL}}"
: "${WALL_SOLVE:=${WALL_TOTAL}}"

# ---------- Verdict classification ---------------------------------------
VERDICT_CLASS="SPGMR_OK"
if [[ ${RC} -ne 0 ]]; then
    VERDICT_CLASS="SHUD_NONZERO_EXIT"
fi
if grep -q '^MARKER:RELEASE_V1_SCALING_WALL_OVERFLOW_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="SCALING_WALL_OVERFLOW"
fi
if [[ -z "${NST:-}" || "${NST}" == "NA" ]]; then
    # If CVODE Final Statistics is missing entirely, the run is malformed
    # (early crash, truncated output, etc.).
    if [[ ${RC} -eq 0 ]]; then
        VERDICT_CLASS="MALFORMED"
    fi
fi

# Fill-in NA for any missing stat.
: "${NST:=NA}"
: "${NCFN:=NA}"
: "${NCFL:=NA}"

# ---------- Emit MARKER:CELL_SCALING_SUMMARY block ------------------------
{
    echo ""
    echo "MARKER:CELL_SCALING_SUMMARY_BEGIN"
    echo "case=${CELL}"
    echo "nthreads=${N}"
    echo "run_label=${RUN_LABEL}"
    echo "wall_total_sec=${WALL_TOTAL}"
    echo "wall_setup_sec=${WALL_SETUP}"
    echo "wall_solve_sec=${WALL_SOLVE}"
    echo "nst=${NST}"
    echo "ncfn=${NCFN}"
    echo "ncfl=${NCFL}"
    echo "omp_place_setting=${OMP_PLACES}"
    echo "verdict_class=${VERDICT_CLASS}"
    echo "MARKER:CELL_SCALING_SUMMARY_END"
    echo ""
    echo "[release-v1-scaling] cell=${CELL} run_label=${RUN_LABEL} nthreads=${N} exit=${RC} wall_total_sec=${WALL_TOTAL} verdict_class=${VERDICT_CLASS}"
} | tee -a "${CELL_OUT}"

exit ${RC}
