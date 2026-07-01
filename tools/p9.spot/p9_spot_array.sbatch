#!/bin/bash
# tools/p9.spot/p9_spot_array.sbatch
#
# PR-Z1 — P9 upper-bound spot check: 2-cell heihe_x4 SPGMR array
# comparing (baseline reltol, reltol=1e-3) under A5 hydrology gate.
#
# Motivation: G0 6-gate NO-GO (ADR-0007/ADR-0008) closed the AMG solver-
# substitution line. Per PR-X2 §Forward action step 2, the next lever
# is CVODE outer-policy tuning; SHUD_CVODE_RELTOL relaxation is the
# cheapest tri-variable to test. This 2-cell array runs heihe_x4 90-day
# under (a) the cfg.para baseline reltol and (b) SHUD_CVODE_RELTOL=1e-3,
# then hands the two output trees to A5 for hydrology-acceptance judging.
#
# 2-cell matrix (heihe_x4, 90-day, SHUD_LINSOL=spgmr, OMP_NUM_THREADS=1):
#   TASK_ID  LABEL      RELTOL
#   0        baseline   default (cfg.para MD->CS.reltol)
#   1        p9         1e-3
#
# Slurm 三铁律 compliance (per CLAUDE.md, mirrors amg_rca_array.sbatch):
#   1) sbatch SHALL be invoked from
#        /scratch/frd_muziyao/SHUD-OpenMP/.p9-spot-runs/<RUN_ID>/
#      (orchestrator sets up RUN_DIR + cd's before sbatch).
#   2) Slurm meta-logs default to slurm-<jobid>_<task>.out under sbatch
#      invocation cwd (= RUN_DIR per rule 1). Per-cell stdout/stderr
#      are additionally tee'd to ${RUN_DIR}/cell-<TASK_ID>-<LABEL>.{out,err}
#      by run_spot_cell.sh.
#   3) All referenced scripts (run_spot_cell.sh, shud_omp binary) resolve
#      under /scratch/frd_muziyao/SHUD-OpenMP/.
#
# Memory: --mem=32G matches heihe_x4 SPGMR baseline envelope (~12 GiB
# RSS peak from G0 PR-A).
#
# Wall: --time=03:00:00. SPGMR heihe_x4 baseline = 1566.5s (~26 min);
# reltol=1e-3 cell EXPECTED to be faster (fewer nst); 3h envelope gives
# headroom for worst-case regression + A5 downstream steps. SIGTERM trap
# emits WALL_OVERFLOW marker per PR-X1 pattern.
#
# Submit (from server):
#   RUN_ID="pr-z1-$(date -u +%Y%m%d-%H%M%S)"
#   mkdir -p /scratch/frd_muziyao/SHUD-OpenMP/.p9-spot-runs/${RUN_ID}
#   cd      /scratch/frd_muziyao/SHUD-OpenMP/.p9-spot-runs/${RUN_ID}
#   sbatch --export=ALL,RUN_ID=${RUN_ID} \
#       /scratch/frd_muziyao/SHUD-OpenMP/tools/p9.spot/p9_spot_array.sbatch
#
# Output (per array task NN, flat under .p9-spot-runs/<RUN_ID>/):
#   cell-<TASK_ID>-<LABEL>.out            stdout sidecar
#   cell-<TASK_ID>-<LABEL>.err            stderr sidecar
#   cell-<TASK_ID>-<LABEL>.output/        copied SHUD *.out/ tree (A5 input)
#
# Refs:
#   - PR-Z1 brief (P9 upper-bound spot check + A5 gate)
#   - tools/p8tune.G0-rca/amg_rca_array.sbatch (template)
#   - CLAUDE.md Slurm 三铁律
#
#SBATCH --job-name=p9-spot-array
#SBATCH --partition=CPU
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# Intentionally omitting --output / --error: Slurm defaults to
# slurm-<jobid>_<array-task>.out under sbatch invocation cwd
# (= .p9-spot-runs/<RUN_ID>/ per 三铁律 rule 1).

set -uo pipefail

# ---------- PATH ------------------------------------------------------------
export PATH="${HOME}/.local/bin:${PATH}"

# ---------- SPGMR environment (defense-in-depth alongside run_spot_cell.sh)
export OMP_NUM_THREADS=1
export SHUD_LINSOL=spgmr

# Hypre runtime dylib resolution kept for parity with PR-X1 harness (SPGMR
# path never reaches Hypre; the dlopen probe path is gated behind
# SHUD_LINSOL=amg — set here only for defensive symmetry).
export HYPRE_LIBDIR=/scratch/frd_muziyao/local/hypre-3.1.0/lib
export LD_LIBRARY_PATH="${HYPRE_LIBDIR}:/scratch/frd_muziyao/local/openblas/lib:${LD_LIBRARY_PATH:-}"

# ---------- RUN_ID resolution ----------------------------------------------
RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-p9-spot-array}.${SLURM_ARRAY_JOB_ID:-0}}"
RUN_DIR="/scratch/frd_muziyao/SHUD-OpenMP/.p9-spot-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
export RUN_DIR

SHUD_OPENMP_ROOT="/scratch/frd_muziyao/SHUD-OpenMP"
TOOL_DIR="${SHUD_OPENMP_ROOT}/tools/p9.spot"
SHUD_DIR="${SHUD_OPENMP_ROOT}/SHUD"

# ---------- Decode SLURM_ARRAY_TASK_ID -> (LABEL, RELTOL) ------------------
# Format per element: "<LABEL>:<RELTOL>" where RELTOL="default" means do NOT
# set SHUD_CVODE_RELTOL (preserves cfg.para MD->CS.reltol).
CONFIGS=(
    "baseline:default"  # TASK 0
    "p9:1e-3"           # TASK 1
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ${TASK_ID} -lt 0 || ${TASK_ID} -ge ${#CONFIGS[@]} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,${#CONFIGS[@]})" >&2
    exit 1
fi

CONFIG="${CONFIGS[${TASK_ID}]}"
IFS=':' read -r LABEL RELTOL <<< "${CONFIG}"

# Cell is ALWAYS heihe_x4 (PR-Z1 scope lock per brief).
CELL="heihe_x4"

# Run label encodes the matrix coordinate for the sidecar filenames.
RUN_LABEL="${TASK_ID}-${LABEL}"

# Export PR-Z1 env hook when RELTOL != "default". Explicit unset otherwise.
if [[ "${RELTOL}" != "default" ]]; then
    export SHUD_CVODE_RELTOL="${RELTOL}"
else
    unset SHUD_CVODE_RELTOL
fi

# ---------- Banner ---------------------------------------------------------
echo "=== p9-spot-array task=${TASK_ID} cell=${CELL} config=${CONFIG} ==="
echo "node=$(hostname) date_utc=$(date -u +%FT%TZ)"
echo "run_id=${RUN_ID}  run_dir=${RUN_DIR}"
echo "shud_dir=${SHUD_DIR}  tool_dir=${TOOL_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID:-unset} slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo "omp_num_threads=${OMP_NUM_THREADS} shud_linsol=${SHUD_LINSOL}"
echo "shud_cvode_reltol=${SHUD_CVODE_RELTOL:-<unset:default_from_cfg_para>}"
echo "run_label=${RUN_LABEL}"
echo ""

# ---------- SIGTERM trap (per PR-X1 template) ------------------------------
START_TS=$(date +%s)
emit_wall_overflow() {
    local now elapsed wall_budget
    now=$(date +%s)
    elapsed=$((now - START_TS))
    wall_budget="unknown"
    if [[ -n "${SLURM_JOB_END_TIME:-}" && -n "${SLURM_JOB_START_TIME:-}" ]]; then
        wall_budget=$((SLURM_JOB_END_TIME - SLURM_JOB_START_TIME))
    fi
    {
        echo ""
        echo "MARKER:P9_SPOT_WALL_OVERFLOW_DETECTED elapsed_sec=${elapsed} wall_budget_sec=${wall_budget}"
        echo "CELL_SUMMARY_BEGIN"
        echo "case=${CELL} run_label=${RUN_LABEL} label=${LABEL} reltol=${RELTOL}"
        echo "wall_total_sec=NA nst=NA nfe=NA nli=NA ncfn=NA ncfl=NA netf=NA"
        echo "verdict_class=WALL_OVERFLOW"
        echo "CELL_SUMMARY_END"
    } 2>/dev/null
    echo "MARKER:P9_SPOT_WALL_OVERFLOW_DETECTED emitted (caught SIGTERM at elapsed=${elapsed}s, task=${TASK_ID})" >&2
    exit 0
}
trap emit_wall_overflow TERM

# ---------- Invoke run_spot_cell.sh ---------------------------------------
# cwd is SHUD/ so ./shud_omp resolves via the basin-relative path.
cd "${SHUD_DIR}" || { echo "ERROR: cannot cd to ${SHUD_DIR}" >&2; exit 1; }
bash "${TOOL_DIR}/run_spot_cell.sh" "${CELL}" "${LABEL}" "${RELTOL}" "${RUN_LABEL}"
RC=$?

echo ""
echo "=== task ${TASK_ID} (cell=${CELL} config=${CONFIG}) exit=${RC} ==="
exit ${RC}
