#!/bin/bash
# tools/p8tune.G0/smoke_array.sbatch
#
# P8-tune.G0 PR-A — 4-cell AMG integrated smoke Slurm array per issue
# #410 task 6.3. Submits {keliya, xinanjiang_upstream, heihe_x4,
# heihe_x16} as 4 parallel array tasks, each invoking
# tools/p8tune.G0/run_smoke_cell.sh with SHUD_LINSOL=amg +
# OMP_NUM_THREADS=1.
#
# Array task -> cell mapping:
#   TASK_ID=0 -> keliya              (small, NumEle=484; quick sanity)
#   TASK_ID=1 -> xinanjiang_upstream (mid,   NumEle=801)
#   TASK_ID=2 -> heihe_x4            (large, NumEle=40046)
#   TASK_ID=3 -> heihe_x16           (XL,    NumEle=~160k; production target)
#
# Slurm 三铁律 compliance (per CLAUDE.md):
#   1) sbatch SHALL be invoked from /scratch/.../.p8tune.G0-runs/<RUN_ID>/
#      (the orchestrator sets up RUN_DIR + cd's before sbatch)
#   2) --output / --error use absolute /scratch/... paths so Slurm doesn't
#      drop the per-task .out / .err on the compute node's /tmp
#   3) all referenced scripts (run_smoke_cell.sh, precheck_env.sh,
#      shud_omp binary) resolve under /scratch/frd_muziyao/SHUD-OpenMP/
#
# Memory: --mem=32G chosen for 173 GiB cn-node cap with comfort margin.
# AMG hierarchy + IJMatrix peak on heihe_x16 fits comfortably in 32 GiB
# (the spike test on heihe_x4 measured peak RSS ~12 GiB; x16 scales ~4×
# in element count, so ~16 GiB is the headroom upper bound). Doubling
# to 32 GiB keeps the OOM marker semantically meaningful.
#
# Wall: --time=08:00:00 headroom for heihe_x16. Smaller cells (keliya,
# xinanjiang_upstream, heihe_x4) finish much faster but inherit the same
# array-level budget. SIGTERM trap (per P8-tune.F M3 fix) emits the
# AMG_WALL_OVERFLOW marker + NA-sentinel cell_summary if a cell hits the
# 8h cap (only heihe_x16 might).
#
# CPUs: --cpus-per-task=1 — sequential measurements per cell so wall
# times are comparable across the 4-cell smoke (matching the wall-signal
# G0-5 anchor convention in spgmr_baseline_walls_g0.h).
#
# Submit (from server):
#   RUN_ID="pr-a-$(date -u +%Y%m%d-%H%M%S)"
#   mkdir -p /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/${RUN_ID}
#   cd      /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/${RUN_ID}
#   sbatch --export=ALL,RUN_ID=${RUN_ID} \
#       /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.G0/smoke_array.sbatch
#
# Output (per array task NN, flat layout under .p8tune.G0-runs/<RUN_ID>):
#   cell-<NN>.out    Slurm-captured stdout
#   cell-<NN>.err    Slurm-captured stderr
#   precheck.log     precheck_env.sh output (TASK_ID=0 only)
#
# Refs:
#   - issue #410 task 6.3 + 10.2 (Slurm 三铁律) + 10.5 (OMP_NUM_THREADS=1)
#   - spec amg-integrated-smoke-verdict REQ "G0 verdict evaluates six PASS/FAIL gates"
#     (G0-4 integrated-completes — this sbatch is the harness)
#
#SBATCH --job-name=g0-smoke-array
#SBATCH --partition=CPU
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/g0-smoke-array/cell-%a.out
#SBATCH --error=/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/g0-smoke-array/cell-%a.err

set -uo pipefail

# ---------- PATH ------------------------------------------------------------
# Compute nodes don't source ~/.bashrc; explicitly prepend ~/.local/bin so
# `uv` is on PATH if any helper shells out. Harmless if absent.
export PATH="${HOME}/.local/bin:${PATH}"

# ---------- AMG environment (defense-in-depth alongside run_smoke_cell.sh) -
export OMP_NUM_THREADS=1
export SHUD_LINSOL=amg

# Hypre runtime dylib resolution. The dlopen runtime probe in
# cvode_config.cpp validates libHYPRE-3.1.0.so loadability at startup;
# LD_LIBRARY_PATH ensures the loader can find the .so under the
# /scratch/.../local/hypre-3.1.0/lib path even if the binary's RPATH
# was stripped.
export HYPRE_LIBDIR=/scratch/frd_muziyao/local/hypre-3.1.0/lib
export LD_LIBRARY_PATH="${HYPRE_LIBDIR}:/scratch/frd_muziyao/local/openblas/lib:${LD_LIBRARY_PATH:-}"

# ---------- RUN_ID resolution ----------------------------------------------
RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-g0-smoke-array}.${SLURM_ARRAY_JOB_ID:-0}}"
RUN_DIR="/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
export RUN_DIR

SHUD_OPENMP_ROOT="/scratch/frd_muziyao/SHUD-OpenMP"
TOOL_DIR="${SHUD_OPENMP_ROOT}/tools/p8tune.G0"
SHUD_DIR="${SHUD_OPENMP_ROOT}/SHUD"

# ---------- Decode SLURM_ARRAY_TASK_ID -> CELL -----------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CELLS=(keliya xinanjiang_upstream heihe_x4 heihe_x16)
if [[ ${TASK_ID} -lt 0 || ${TASK_ID} -ge ${#CELLS[@]} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,${#CELLS[@]})" >&2
    exit 1
fi
CELL="${CELLS[${TASK_ID}]}"

# ---------- Banner ----------------------------------------------------------
echo "=== g0-smoke-array task=${TASK_ID} cell=${CELL} ==="
echo "node=$(hostname) date_utc=$(date -u +%FT%TZ)"
echo "run_id=${RUN_ID}  run_dir=${RUN_DIR}"
echo "shud_dir=${SHUD_DIR}  tool_dir=${TOOL_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID:-unset} slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo "omp_num_threads=${OMP_NUM_THREADS} shud_linsol=${SHUD_LINSOL}"
echo "hypre_libdir=${HYPRE_LIBDIR}"
echo ""

# ---------- SIGTERM trap (per P8-tune.F M3 fix) -----------------------------
# On Slurm time-cancel, emit a well-formed cell_summary KV block with NA
# sentinels plus MARKER:AMG_WALL_OVERFLOW_DETECTED. The aggregator (PR-B)
# uses the marker to classify the cell as AMG_WALL_OVERFLOW instead of
# a hard Slurm failure. Exit 0 so Slurm sees this as a clean data point.
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
        echo "MARKER:AMG_WALL_OVERFLOW_DETECTED elapsed_sec=${elapsed} wall_budget_sec=${wall_budget}"
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
    echo "MARKER:AMG_WALL_OVERFLOW_DETECTED emitted (caught SIGTERM at elapsed=${elapsed}s)" >&2
    exit 0
}
trap emit_wall_overflow TERM

# ---------- precheck_env.sh gate (TASK_ID=0 inline) ------------------------
# Per issue #410 task 6.3 + P8-tune.F H1 fix: precheck runs BEFORE cell
# body, only on TASK_ID=0 (environment-invariant across the 4 cells).
# Other tasks proceed without re-paying the precheck cost. On failure,
# this cell aborts; peer cells continue (Slurm doesn't cancel array on
# one-task fail unless explicitly requested).
cd "${SHUD_DIR}" || { echo "ERROR: cannot cd to ${SHUD_DIR}" >&2; exit 1; }

if [[ ${TASK_ID} -eq 0 ]]; then
    PRECHECK_LOG="${RUN_DIR}/precheck.log"
    bash "${TOOL_DIR}/precheck_env.sh" > "${PRECHECK_LOG}" 2>&1
    PRECHECK_RC=$?
    if [[ ${PRECHECK_RC} -ne 0 ]]; then
        echo "ERROR: precheck_env.sh failed (rc=${PRECHECK_RC}); see ${PRECHECK_LOG}" >&2
        tail -30 "${PRECHECK_LOG}" >&2
        exit 1
    fi
    echo "[task 0] precheck_env.sh 7/7 PASS (log: ${PRECHECK_LOG})"
fi

# ---------- Invoke run_smoke_cell.sh ---------------------------------------
# RUN_DIR exported above is consumed by run_smoke_cell.sh for the
# per-cell .out / .err sidecar files. cwd is SHUD/ so ./shud_omp resolves.
bash "${TOOL_DIR}/run_smoke_cell.sh" "${CELL}"
RC=$?

echo ""
echo "=== task ${TASK_ID} (cell=${CELL}) exit=${RC} ==="
exit ${RC}
