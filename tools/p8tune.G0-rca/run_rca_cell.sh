#!/bin/bash
# tools/p8tune.G0-rca/run_rca_cell.sh
#
# P8-tune.G0 PR-X1 — single-cell heihe_x4 AMG tolerance + CVODE EpsLin
# RCA runner. Invoked per matrix-cell from amg_rca_array.sbatch with the
# explicit (AMG_TOL, EPSLIN) tuple decoded from SLURM_ARRAY_TASK_ID; sets
# the corresponding env hooks (PR-X1 SHUD_AMG_TOL + SHUD_CVODE_EPSLIN),
# verifies 90-day cfg.para truncation (IA-7 fix mirrored from G0 PR-A),
# then invokes `./shud_omp heihe_x4` and emits a per-cell cell_summary
# KV block extended with the RCA matrix coordinates.
#
# Per CLAUDE.md 项目级铁律 "所有 case ≤90 天截断". Mirrors
# tools/p8tune.G0/run_smoke_cell.sh but:
#   - CELL is always heihe_x4 (hardcoded per PR-X1 scope; rejects others)
#   - Accepts explicit (AMG_TOL, EPSLIN, RUN_LABEL) args from sbatch
#   - cell_summary block emits extra fields:
#         shud_amg_tol=<value>
#         shud_cvode_epslin=<value-or-default>
#         run_label=<value>
#
# Usage:
#   bash run_rca_cell.sh <cell_name> <amg_tol> <epslin> <run_label>
#
# Args:
#   cell_name   MUST be 'heihe_x4' (PR-X1 scope lock per brief)
#   amg_tol     float in (0, 1), e.g., 1e-7, 1e-9, 1e-11, 1e-13
#   epslin      float in (0, 1) OR 'default' (preserves CVODE default 0.05)
#   run_label   identifier used in sidecar filenames, e.g., '0-tol1e-7-epsdefault'
#
# Exit code:
#   0  PASS — `shud_omp` exited 0 + cell_summary emitted
#   2  precondition failed (cell != heihe_x4, cfg.para !=90, or unable to
#      locate heihe_x4 basin dir)
#   *  non-zero — propagated from shud_omp (divergence / OOM / SIGTERM)
#
# Refs:
#   - PR-X1 brief (RCA for G0 verdict #412 NO-GO; ncfn=100138 hypothesis)
#   - tools/p8tune.G0/run_smoke_cell.sh (template; this file's PR-X1
#     extension adds the matrix-coord fields)
#   - CLAUDE.md 项目级铁律 90-day truncation

set -euo pipefail

# ---------- CLI -------------------------------------------------------------
if [[ $# -ne 4 ]]; then
    echo "usage: $0 <cell_name> <amg_tol> <epslin> <run_label>" >&2
    echo "  cell_name   MUST be 'heihe_x4' (PR-X1 scope)" >&2
    echo "  amg_tol     float in (0, 1), e.g., 1e-7" >&2
    echo "  epslin      float in (0, 1) OR 'default'" >&2
    echo "  run_label   identifier, e.g., '0-tol1e-7-epsdefault'" >&2
    exit 2
fi

CELL="$1"
AMG_TOL="$2"
EPSLIN="$3"
RUN_LABEL="$4"

# PR-X1 scope lock: enforce heihe_x4 only. Other cells are out of scope
# per the brief; rejecting up front prevents accidental sweep expansion.
if [[ "${CELL}" != "heihe_x4" ]]; then
    echo "[shud-G0-RCA] FATAL: cell_name='${CELL}' is out of PR-X1 scope " \
         "(only heihe_x4 is allowed; PR-X2 may expand)" >&2
    exit 2
fi

# Project name mirrors SHUD CLI convention (basin folder == project name
# for heihe_x4 per docs/case_deployment_map.md).
PROJECT_NAME="heihe_x4"

# ---------- Environment -----------------------------------------------------
# AMG path selector (defense-in-depth: also set in amg_rca_array.sbatch).
export SHUD_LINSOL=amg

# OMP_NUM_THREADS=1 per the G0 OMP_CUTOFF policy + PR-X1 brief.
export OMP_NUM_THREADS=1

# PR-X1 env hooks. AMG_TOL is always exported. EPSLIN is only exported when
# != "default" (preserves the SUNDIALS default 0.05 unchanged).
export SHUD_AMG_TOL="${AMG_TOL}"
if [[ "${EPSLIN}" != "default" ]]; then
    export SHUD_CVODE_EPSLIN="${EPSLIN}"
else
    unset SHUD_CVODE_EPSLIN
fi

# ---------- Locate cell artifacts -------------------------------------------
# Auto-cd to Basins/heihe_x4 when invoked from SHUD/ root (sbatch cwd).
if [[ -d "Basins/${CELL}" && ! -d "input/${PROJECT_NAME}" ]]; then
    cd "Basins/${CELL}" || {
        echo "[shud-G0-RCA] FATAL: cannot cd to Basins/${CELL}" >&2
        exit 2
    }
fi

if [[ ! -d "input/${PROJECT_NAME}" ]]; then
    echo "[shud-G0-RCA] FATAL: input/${PROJECT_NAME}/ not found under $(pwd) — heihe_x4 not deployed" >&2
    exit 2
fi

CFG_PARA="$(find "input/${PROJECT_NAME}" -name '*.cfg.para' 2>/dev/null | head -n 1)"
if [[ -z "${CFG_PARA}" || ! -f "${CFG_PARA}" ]]; then
    echo "[shud-G0-RCA] FATAL: no *.cfg.para under input/${PROJECT_NAME}/ — heihe_x4 not deployed" >&2
    exit 2
fi

# ---------- IA-7 fix: 90-day cfg.para safety check --------------------------
# Mirrors run_smoke_cell.sh — KEY<whitespace>VALUE form, awk extract delta.
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
    echo "[shud-G0-RCA] FATAL: 90-day truncation precondition failed for case=${CELL} END-START=${END_START_DELTA}" >&2
    echo "[shud-G0-RCA]        cfg.para=${CFG_PARA}" >&2
    echo "[shud-G0-RCA]        expected END-START=90 (per CLAUDE.md 项目级铁律)" >&2
    exit 2
fi

# ---------- Banner ----------------------------------------------------------
echo "=== run_rca_cell.sh cell=${CELL} run_label=${RUN_LABEL} ==="
echo "date_utc:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname:           $(hostname)"
echo "cwd:                $(pwd)"
echo "cell:               ${CELL}"
echo "project_name:       ${PROJECT_NAME}"
echo "cfg_para:           ${CFG_PARA}"
echo "end_start_delta:    ${END_START_DELTA} (90-day precondition PASS)"
echo "shud_linsol:        ${SHUD_LINSOL}"
echo "omp_num_threads:    ${OMP_NUM_THREADS}"
echo "shud_amg_tol:       ${SHUD_AMG_TOL}"
echo "shud_cvode_epslin:  ${SHUD_CVODE_EPSLIN:-<unset:default>}"

# ---------- Resolve RUN_DIR for tee output ---------------------------------
if [[ -n "${RUN_DIR:-}" ]]; then
    RUN_DIR_RESOLVED="${RUN_DIR}"
else
    RUN_DIR_RESOLVED="$(pwd)/.g0-rca-local"
fi
mkdir -p "${RUN_DIR_RESOLVED}"

# Per-cell sidecar filenames encode the matrix coordinate via RUN_LABEL.
CELL_OUT="${RUN_DIR_RESOLVED}/cell-${RUN_LABEL}.out"
CELL_ERR="${RUN_DIR_RESOLVED}/cell-${RUN_LABEL}.err"

# PR-B telemetry drain TSV (per-cell). Wrapper drains its ring buffer on
# SUNLinSolFree (shutdown). Aggregator picks up these TSVs for per-cell
# AMG telemetry attribution.
export SHUD_TELEMETRY_TSV="${RUN_DIR_RESOLVED}/cell-${RUN_LABEL}.telemetry.tsv"

echo "run_dir:            ${RUN_DIR_RESOLVED}"
echo "cell_out:           ${CELL_OUT}"
echo "cell_err:           ${CELL_ERR}"
echo "telemetry_tsv:      ${SHUD_TELEMETRY_TSV}"
echo ""

# ---------- Binary preflight -----------------------------------------------
if [[ -n "${SHUD_BIN:-}" && -x "${SHUD_BIN}" ]]; then
    SHUD_BIN_RESOLVED="${SHUD_BIN}"
elif [[ -x "../../shud_omp" ]]; then
    SHUD_BIN_RESOLVED="../../shud_omp"
elif [[ -x "./shud_omp" ]]; then
    SHUD_BIN_RESOLVED="./shud_omp"
else
    echo "[shud-G0-RCA] FATAL: shud_omp not found (checked \$SHUD_BIN, ../../shud_omp, ./shud_omp) under $(pwd)" >&2
    exit 2
fi
echo "shud_bin:           ${SHUD_BIN_RESOLVED}"
echo ""

# ---------- Capture wall + invoke shud_omp ----------------------------------
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

# ---------- Extract minimal cell_summary fields -----------------------------
# All extraction below tolerates grep/awk no-match (set +e covers).
set +e

# nst — CVODE step count from PrintFinalStats output.
NST=$(grep -E '^[[:space:]]*(nst|NumSteps)[[:space:]]*=' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk -F'=' '{gsub(/[[:space:]]/,"",$2); print $2}')
if [[ -z "${NST}" ]]; then
    NST="NA"
fi

# hypre_release — from MARKER:AMG_TELEMETRY_REAL.
HYPRE_RELEASE=$(grep -oE 'hypre[_ ]release=([0-9]+\.[0-9]+\.[0-9]+|[0-9]+)' "${CELL_OUT}" 2>/dev/null \
                | head -n 1 | awk -F= '{print $2}')
if [[ -z "${HYPRE_RELEASE}" ]]; then
    HYPRE_RELEASE="NA"
fi

# verdict_class — derive from exit code + marker presence.
VERDICT_CLASS="AMG_OK"
if grep -q '^MARKER:AMG_SETUP_DIVERGE_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="AMG_SETUP_DIVERGE"
elif grep -q '^MARKER:AMG_SOLVE_DIVERGE_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="AMG_SOLVE_DIVERGE"
elif grep -q '^MARKER:AMG_OOM_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="AMG_OOM"
elif grep -q '^MARKER:AMG_WALL_OVERFLOW_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="AMG_WALL_OVERFLOW"
fi
set -e

# ---------- Emit cell_summary KV block --------------------------------------
# Same shape as run_smoke_cell.sh + 3 extra fields (shud_amg_tol,
# shud_cvode_epslin, run_label) so aggregate_rca.sh can attribute each
# matrix cell to its (AMG_TOL, EPSLIN) coordinate.
{
    echo ""
    echo "CELL_SUMMARY_BEGIN"
    echo "case=${CELL} run_label=${RUN_LABEL} shud_amg_tol=${AMG_TOL} shud_cvode_epslin=${EPSLIN}"
    echo "interp_type=6 coarsen_type=8 NumY=NA nnz_A=NA"
    echo "setup_wall_sec=NA apply_wall_sec=NA peak_rss_bytes=NA"
    echo "cycle_complexity=NA operator_complexity=NA"
    echo "verdict_class=${VERDICT_CLASS}"
    echo "hypre_version=${HYPRE_RELEASE} colpack_version=absent shud_pin=NA"
    echo "cvode_nfe=NA cvode_nli=NA cvode_nfeLS=NA cvode_ncfn=NA cvode_ncfl=NA cvode_netf=NA"
    echo "amg_telemetry_mean_iters=NA amg_telemetry_mean_op_count=NA amg_telemetry_setup_calls=NA amg_total_setup_wall_sec=NA amg_total_solve_wall_sec=NA"
    echo "n_cvode_steps=${NST} amg_wall_per_step_sec=NA"
    echo "CELL_SUMMARY_END"
    echo ""
    echo "[shud-G0-RCA] cell=${CELL} run_label=${RUN_LABEL} amg_tol=${AMG_TOL} epslin=${EPSLIN} exit_code=${RC} wall_total_sec=${WALL_TOTAL} verdict_class=${VERDICT_CLASS}"
} | tee -a "${CELL_OUT}"

exit ${RC}
