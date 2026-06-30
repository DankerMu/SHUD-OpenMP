#!/bin/bash
# tools/p8tune.G0/smoke_single_test.sbatch
#
# P8-tune.G0 PR-A — keliya-only dry-run gate per issue #410 task 6.4.
# Same shape as smoke_array.sbatch but `--array=0` only and
# `--time=00:30:00`. Submitted BEFORE the full 4-cell smoke_array.sbatch
# as a fast pre-flight on the server cn-node:
#   - validates precheck_env.sh 7/7 PASS
#   - validates the keliya 90-day AMG path completes
#   - validates MARKER:AMG_TELEMETRY_REAL + `[shud-amg] Hypre threads=1`
#     appear in stdout (single_test gate per task 6.5)
#
# Slurm 三铁律 compliance (per CLAUDE.md):
#   1) sbatch from /scratch/.../.p8tune.G0-runs/<RUN_ID>/
#   2) --output / --error use absolute /scratch/... paths
#   3) referenced scripts under /scratch/frd_muziyao/SHUD-OpenMP/
#
# Submit (from server):
#   RUN_ID="pr-a-single-$(date -u +%Y%m%d-%H%M%S)"
#   mkdir -p /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/${RUN_ID}
#   cd      /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/${RUN_ID}
#   sbatch --export=ALL,RUN_ID=${RUN_ID} \
#       /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.G0/smoke_single_test.sbatch
#
# Refs:
#   - issue #410 task 6.4 + 6.5
#
#SBATCH --job-name=g0-smoke-single
#SBATCH --partition=CPU
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/g0-smoke-single/cell-%a.out
#SBATCH --error=/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/g0-smoke-single/cell-%a.err

set -uo pipefail

# ---------- PATH ------------------------------------------------------------
export PATH="${HOME}/.local/bin:${PATH}"

# ---------- AMG environment -------------------------------------------------
export OMP_NUM_THREADS=1
export SHUD_LINSOL=amg
export HYPRE_LIBDIR=/scratch/frd_muziyao/local/hypre-3.1.0/lib
export LD_LIBRARY_PATH="${HYPRE_LIBDIR}:/scratch/frd_muziyao/local/openblas/lib:${LD_LIBRARY_PATH:-}"

# ---------- RUN_ID resolution ----------------------------------------------
RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-g0-smoke-single}.${SLURM_ARRAY_JOB_ID:-0}}"
RUN_DIR="/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
export RUN_DIR

SHUD_OPENMP_ROOT="/scratch/frd_muziyao/SHUD-OpenMP"
TOOL_DIR="${SHUD_OPENMP_ROOT}/tools/p8tune.G0"
SHUD_DIR="${SHUD_OPENMP_ROOT}/SHUD"

# ---------- Decode SLURM_ARRAY_TASK_ID -> CELL -----------------------------
# Single-test pins to TASK_ID=0 (keliya); validate just to be safe.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ${TASK_ID} -ne 0 ]]; then
    echo "ERROR: smoke_single_test.sbatch only supports TASK_ID=0 (keliya), got ${TASK_ID}" >&2
    exit 1
fi
CELL="keliya"

echo "=== g0-smoke-single task=${TASK_ID} cell=${CELL} (dry-run gate) ==="
echo "node=$(hostname) date_utc=$(date -u +%FT%TZ)"
echo "run_id=${RUN_ID}  run_dir=${RUN_DIR}"
echo "omp_num_threads=${OMP_NUM_THREADS} shud_linsol=${SHUD_LINSOL}"
echo ""

# ---------- SIGTERM trap ---------------------------------------------------
# 30-min wall is generous for keliya 90-day SHORT (Mac local 90-day ~30s);
# overflow would indicate a real hang, not a wall-budget shortfall. Still
# emit AMG_WALL_OVERFLOW marker for aggregator consistency.
START_TS=$(date +%s)
emit_wall_overflow() {
    local now elapsed
    now=$(date +%s)
    elapsed=$((now - START_TS))
    {
        echo ""
        echo "MARKER:AMG_WALL_OVERFLOW_DETECTED elapsed_sec=${elapsed} wall_budget_sec=1800"
        echo "CELL_SUMMARY_BEGIN"
        echo "case=${CELL} interp_type=6 coarsen_type=8 NumY=NA nnz_A=NA"
        echo "setup_wall_sec=NA apply_wall_sec=NA peak_rss_bytes=NA"
        echo "cycle_complexity=NA operator_complexity=NA"
        echo "verdict_class=AMG_WALL_OVERFLOW"
        echo "hypre_version=NA colpack_version=absent shud_pin=NA"
        echo "cvode_nfe=NA cvode_nli=NA cvode_nfeLS=NA cvode_ncfn=NA cvode_ncfl=NA cvode_netf=NA"
        echo "amg_telemetry_mean_iters=NA amg_telemetry_mean_op_count=NA amg_telemetry_setup_calls=NA amg_total_setup_wall_sec=NA amg_total_solve_wall_sec=NA"
        echo "n_cvode_steps=NA amg_wall_per_step_sec=NA"
        echo "CELL_SUMMARY_END"
    } 2>/dev/null
    echo "MARKER:AMG_WALL_OVERFLOW_DETECTED emitted (SIGTERM at elapsed=${elapsed}s)" >&2
    exit 0
}
trap emit_wall_overflow TERM

# ---------- precheck_env.sh gate -------------------------------------------
cd "${SHUD_DIR}" || { echo "ERROR: cannot cd to ${SHUD_DIR}" >&2; exit 1; }

PRECHECK_LOG="${RUN_DIR}/precheck.log"
bash "${TOOL_DIR}/precheck_env.sh" > "${PRECHECK_LOG}" 2>&1
PRECHECK_RC=$?
if [[ ${PRECHECK_RC} -ne 0 ]]; then
    echo "ERROR: precheck_env.sh failed (rc=${PRECHECK_RC}); see ${PRECHECK_LOG}" >&2
    tail -30 "${PRECHECK_LOG}" >&2
    exit 1
fi
echo "[single_test] precheck_env.sh 7/7 PASS (log: ${PRECHECK_LOG})"

# ---------- Invoke run_smoke_cell.sh ---------------------------------------
bash "${TOOL_DIR}/run_smoke_cell.sh" "${CELL}"
RC=$?

echo ""
echo "=== smoke_single_test cell=${CELL} exit=${RC} (dry-run gate complete) ==="
exit ${RC}
