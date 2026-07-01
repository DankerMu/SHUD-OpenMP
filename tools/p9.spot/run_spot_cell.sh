#!/bin/bash
# tools/p9.spot/run_spot_cell.sh
#
# PR-Z1 — single-cell heihe_x4 SPGMR runner for the P9 upper-bound spot
# check array. Accepts explicit (LABEL, RELTOL, RUN_LABEL) args from the
# sbatch array dispatcher; sets the corresponding env hook (PR-Z1
# SHUD_CVODE_RELTOL); verifies 90-day cfg.para truncation; invokes
# `./shud_omp heihe_x4`; copies the SHUD *.out/ tree next to the sidecar
# logs (so run_a5_for_spot.sh can consume it); and emits a per-cell
# cell_summary KV block.
#
# Per CLAUDE.md 项目级铁律 "所有 case <=90 天截断". Mirrors
# tools/p8tune.G0-rca/run_rca_cell.sh but:
#   - Always SPGMR (SHUD_LINSOL=spgmr, no AMG)
#   - Accepts (LABEL, RELTOL) instead of (AMG_TOL, EPSLIN)
#   - Only sets SHUD_CVODE_RELTOL when RELTOL != "default"
#   - Copies output tree to RUN_DIR/cell-<RUN_LABEL>.output/ for A5
#
# Usage:
#   bash run_spot_cell.sh <cell_name> <label> <reltol> <run_label>
#
# Args:
#   cell_name   MUST be 'heihe_x4' (PR-Z1 scope lock per brief)
#   label       matrix axis identifier, e.g., 'baseline', 'p9'
#   reltol      float in (0, 1) OR 'default' (preserves cfg.para reltol)
#   run_label   identifier used in sidecar filenames, e.g., '0-baseline'
#
# Exit code:
#   0  PASS — shud_omp exited 0 + cell_summary emitted + output copied
#   2  precondition failed (cell != heihe_x4, cfg.para not 90-day, missing
#      basin dir)
#   *  non-zero — propagated from shud_omp
#
# Refs:
#   - PR-Z1 brief
#   - tools/p8tune.G0-rca/run_rca_cell.sh (template)
#   - CLAUDE.md 项目级铁律 90-day truncation

set -euo pipefail

# ---------- CLI ------------------------------------------------------------
if [[ $# -ne 4 ]]; then
    echo "usage: $0 <cell_name> <label> <reltol> <run_label>" >&2
    echo "  cell_name   MUST be 'heihe_x4' (PR-Z1 scope)" >&2
    echo "  label       'baseline' or 'p9'" >&2
    echo "  reltol      float in (0, 1) OR 'default'" >&2
    echo "  run_label   identifier, e.g., '0-baseline'" >&2
    exit 2
fi

CELL="$1"
LABEL="$2"
RELTOL="$3"
RUN_LABEL="$4"

# PR-Z1 scope lock: enforce heihe_x4 only.
if [[ "${CELL}" != "heihe_x4" ]]; then
    echo "[shud-P9-SPOT] FATAL: cell_name='${CELL}' is out of PR-Z1 scope " \
         "(only heihe_x4 is allowed)" >&2
    exit 2
fi

PROJECT_NAME="heihe_x4"

# ---------- Environment ----------------------------------------------------
# SPGMR path selector (defense-in-depth: also set in sbatch dispatcher).
export SHUD_LINSOL=spgmr

# OMP_NUM_THREADS=1 per PR-Z1 brief (fair comparison vs G0 SPGMR baseline).
export OMP_NUM_THREADS=1

# PR-Z1 env hook. Only export when RELTOL != "default" so the baseline cell
# gets the cfg.para-parsed value untouched.
if [[ "${RELTOL}" != "default" ]]; then
    export SHUD_CVODE_RELTOL="${RELTOL}"
else
    unset SHUD_CVODE_RELTOL
fi

# ---------- Locate cell artifacts -----------------------------------------
# Auto-cd to Basins/heihe_x4 when invoked from SHUD/ root (sbatch cwd).
if [[ -d "Basins/${CELL}" && ! -d "input/${PROJECT_NAME}" ]]; then
    cd "Basins/${CELL}" || {
        echo "[shud-P9-SPOT] FATAL: cannot cd to Basins/${CELL}" >&2
        exit 2
    }
fi

if [[ ! -d "input/${PROJECT_NAME}" ]]; then
    echo "[shud-P9-SPOT] FATAL: input/${PROJECT_NAME}/ not found under $(pwd) — heihe_x4 not deployed" >&2
    exit 2
fi

CFG_PARA="$(find "input/${PROJECT_NAME}" -name '*.cfg.para' 2>/dev/null | head -n 1)"
if [[ -z "${CFG_PARA}" || ! -f "${CFG_PARA}" ]]; then
    echo "[shud-P9-SPOT] FATAL: no *.cfg.para under input/${PROJECT_NAME}/" >&2
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
    echo "[shud-P9-SPOT] FATAL: 90-day truncation precondition failed for case=${CELL} END-START=${END_START_DELTA}" >&2
    echo "[shud-P9-SPOT]        cfg.para=${CFG_PARA}" >&2
    exit 2
fi

# ---------- Banner --------------------------------------------------------
echo "=== run_spot_cell.sh cell=${CELL} run_label=${RUN_LABEL} ==="
echo "date_utc:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname:           $(hostname)"
echo "cwd:                $(pwd)"
echo "cell:               ${CELL}"
echo "project_name:       ${PROJECT_NAME}"
echo "cfg_para:           ${CFG_PARA}"
echo "end_start_delta:    ${END_START_DELTA} (90-day precondition PASS)"
echo "shud_linsol:        ${SHUD_LINSOL}"
echo "omp_num_threads:    ${OMP_NUM_THREADS}"
echo "label:              ${LABEL}"
echo "reltol:             ${RELTOL}"
echo "shud_cvode_reltol:  ${SHUD_CVODE_RELTOL:-<unset:default_from_cfg_para>}"

# ---------- Resolve RUN_DIR for sidecars ----------------------------------
if [[ -n "${RUN_DIR:-}" ]]; then
    RUN_DIR_RESOLVED="${RUN_DIR}"
else
    RUN_DIR_RESOLVED="$(pwd)/.p9-spot-local"
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
    echo "[shud-P9-SPOT] FATAL: shud_omp not found (checked \$SHUD_BIN, ../../shud_omp, ./shud_omp) under $(pwd)" >&2
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
    echo "[shud-P9-SPOT] copied ${SHUD_OUTPUT_SRC}/ -> ${CELL_OUTPUT_TREE}/"
else
    echo "[shud-P9-SPOT] WARN: shud_omp exit=${RC} or ${SHUD_OUTPUT_SRC} missing; A5 input tree NOT staged" >&2
fi

# ---------- Extract cell_summary fields -----------------------------------
set +e
NST=$(grep -E '^[[:space:]]*nst[[:space:]]*=' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk -F'=' '{gsub(/[[:space:]]/,"",$2); print $2}')
NFE=$(grep -E '^[[:space:]]*nfe[[:space:]]*=' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk -F'=' '{gsub(/[[:space:]]/,"",$2); print $2}')
NLI=$(grep -E '^[[:space:]]*nli[[:space:]]*=' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk '{
    # nni = X    nli = Y   -> extract nli via second =
    for (i=1;i<=NF;i++) if ($i=="nli") { print $(i+2); exit }
}')
NCFN=$(grep -E '^[[:space:]]*ncfn[[:space:]]*=' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk -F'=' '{gsub(/[[:space:]]/,"",$2); print $2}')
NCFL=$(grep -E 'ncfl[[:space:]]*=' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk '{
    for (i=1;i<=NF;i++) if ($i=="ncfl") { print $(i+2); exit }
}')
NETF=$(grep -E 'netf[[:space:]]*=' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk '{
    for (i=1;i<=NF;i++) if ($i=="netf") { print $(i+2); exit }
}')

VERDICT_CLASS="SPGMR_OK"
if [[ ${RC} -ne 0 ]]; then
    VERDICT_CLASS="SHUD_NONZERO_EXIT"
fi
if grep -q '^MARKER:P9_SPOT_WALL_OVERFLOW_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="WALL_OVERFLOW"
fi
set -e

# Fill-in NA for any missing stat
: "${NST:=NA}"
: "${NFE:=NA}"
: "${NLI:=NA}"
: "${NCFN:=NA}"
: "${NCFL:=NA}"
: "${NETF:=NA}"

# ---------- Emit cell_summary KV block ------------------------------------
{
    echo ""
    echo "CELL_SUMMARY_BEGIN"
    echo "case=${CELL} run_label=${RUN_LABEL} label=${LABEL} reltol_effective=${RELTOL}"
    echo "wall_total_sec=${WALL_TOTAL}"
    echo "nst=${NST} nfe=${NFE} nli=${NLI} ncfn=${NCFN} ncfl=${NCFL} netf=${NETF}"
    echo "verdict_class=${VERDICT_CLASS}"
    echo "CELL_SUMMARY_END"
    echo ""
    echo "[shud-P9-SPOT] cell=${CELL} run_label=${RUN_LABEL} label=${LABEL} reltol=${RELTOL} exit_code=${RC} wall_total_sec=${WALL_TOTAL} verdict_class=${VERDICT_CLASS}"
} | tee -a "${CELL_OUT}"

exit ${RC}
