#!/bin/bash
# tools/p8tune.G0-rca/amg_rca_array.sbatch
#
# P8-tune.G0 PR-X1 — 8-cell heihe_x4 AMG tolerance + CVODE EpsLin RCA
# Slurm array for G0 verdict (#412) NO-GO root-cause analysis.
#
# Background: G0 aggregate.tsv showed heihe_x4 AMG run had ncfl=0
# (Hypre never reported convergence failure) but ncfn=100138 (CVODE
# outer Newton control failure exploded), nst=254756 vs SPGMR
# baseline 6572 (38.8x step inflation), and total wall 15.1x WORSE
# than SPGMR. Hypothesis: AMG's default solve tolerance (1e-8) is too
# loose -> Hypre "early-outs" with insufficient Newton-correction
# quality -> outer Newton fails -> CVODE retries/shrinks step ->
# ncfn explodes. Lever: HYPRE_BoomerAMGSetTol + CVodeSetEpsLin.
#
# 8-cell matrix (heihe_x4, 90-day, OMP_NUM_THREADS=1, SHUD_LINSOL=amg):
#   TASK_ID  AMG_TOL    EPSLIN
#   0        1e-7       (default 0.05)
#   1        1e-9       (default 0.05)
#   2        1e-11      (default 0.05)
#   3        1e-13      (default 0.05)
#   4        1e-7       0.005
#   5        1e-9       0.005
#   6        1e-11      0.005
#   7        1e-13      0.005
#
# Slurm 三铁律 compliance (per CLAUDE.md, mirrors smoke_array.sbatch
# header §2):
#   1) sbatch SHALL be invoked from /scratch/.../.p8tune.G0-rca-runs/<RUN_ID>/
#      (orchestrator sets up RUN_DIR + cd's before sbatch).
#   2) Slurm meta-logs default to slurm-<jobid>_<task>.out under invocation
#      cwd (= /scratch/.../.p8tune.G0-rca-runs/<RUN_ID>/ per rule 1).
#      Per-cell stdout/stderr are also tee'd to ${RUN_DIR}/cell-<NN>-*.{out,err}
#      by run_rca_cell.sh.
#   3) all referenced scripts (run_rca_cell.sh, shud_omp binary) resolve
#      under /scratch/frd_muziyao/SHUD-OpenMP/.
#
# Memory: --mem=32G chosen to match the G0 PR-A heihe_x4 cell memory
# envelope (peak RSS ~12 GiB on G0 heihe_x4 AMG run; 32G keeps OOM
# marker semantically meaningful).
#
# Wall: --time=08:00:00 headroom. Baseline G0 PR-A heihe_x4 AMG cell
# (with ncfn=100138 explosion) took ~6.5h; tighter tolerances may
# extend further. SIGTERM trap emits AMG_WALL_OVERFLOW marker per
# the smoke_array.sbatch convention if a cell hits the cap.
#
# Submit (from server):
#   RUN_ID="pr-x1-$(date -u +%Y%m%d-%H%M%S)"
#   mkdir -p /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-rca-runs/${RUN_ID}
#   cd      /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-rca-runs/${RUN_ID}
#   sbatch --export=ALL,RUN_ID=${RUN_ID} \
#       /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.G0-rca/amg_rca_array.sbatch
#
# Output (per array task NN, flat layout under .p8tune.G0-rca-runs/<RUN_ID>):
#   cell-<NN>-tol<AMG_TOL>-eps<EPSLIN>.out    stdout sidecar
#   cell-<NN>-tol<AMG_TOL>-eps<EPSLIN>.err    stderr sidecar
#   cell-<NN>-tol<AMG_TOL>-eps<EPSLIN>.telemetry.tsv  PR-B drain hook output
#
# Refs:
#   - PR-X1 brief (RCA for G0 verdict #412 NO-GO)
#   - tools/p8tune.G0/smoke_array.sbatch (template)
#   - CLAUDE.md Slurm 三铁律
#
#SBATCH --job-name=g0-rca-array
#SBATCH --partition=CPU
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# Intentionally omitting --output / --error: Slurm defaults to
# slurm-<jobid>_<array-task>.out under sbatch invocation cwd
# (= /scratch/.../.p8tune.G0-rca-runs/<RUN_ID>/ per 三铁律 rule 1).
# Per-cell sidecars from run_rca_cell.sh complement the Slurm meta-log.

set -uo pipefail

# ---------- PATH ------------------------------------------------------------
export PATH="${HOME}/.local/bin:${PATH}"

# ---------- AMG environment (defense-in-depth alongside run_rca_cell.sh) ---
export OMP_NUM_THREADS=1
export SHUD_LINSOL=amg

# Hypre runtime dylib resolution.
export HYPRE_LIBDIR=/scratch/frd_muziyao/local/hypre-3.1.0/lib
export LD_LIBRARY_PATH="${HYPRE_LIBDIR}:/scratch/frd_muziyao/local/openblas/lib:${LD_LIBRARY_PATH:-}"

# ---------- RUN_ID resolution ----------------------------------------------
RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-g0-rca-array}.${SLURM_ARRAY_JOB_ID:-0}}"
RUN_DIR="/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-rca-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
export RUN_DIR

SHUD_OPENMP_ROOT="/scratch/frd_muziyao/SHUD-OpenMP"
TOOL_DIR="${SHUD_OPENMP_ROOT}/tools/p8tune.G0-rca"
SHUD_DIR="${SHUD_OPENMP_ROOT}/SHUD"

# ---------- Decode SLURM_ARRAY_TASK_ID -> (AMG_TOL, EPSLIN) ----------------
# Format per element: "<AMG_TOL>:<EPSLIN>" where EPSLIN="default" means do
# NOT set SHUD_CVODE_EPSLIN (preserves SUNDIALS default 0.05).
CONFIGS=(
    "1e-7:default"   # TASK 0
    "1e-9:default"   # TASK 1
    "1e-11:default"  # TASK 2
    "1e-13:default"  # TASK 3
    "1e-7:0.005"     # TASK 4
    "1e-9:0.005"     # TASK 5
    "1e-11:0.005"    # TASK 6
    "1e-13:0.005"    # TASK 7
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ${TASK_ID} -lt 0 || ${TASK_ID} -ge ${#CONFIGS[@]} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,${#CONFIGS[@]})" >&2
    exit 1
fi

CONFIG="${CONFIGS[${TASK_ID}]}"
IFS=':' read -r AMG_TOL EPSLIN <<< "${CONFIG}"

# Cell is ALWAYS heihe_x4 in this RCA harness (per PR-X1 scope lock:
# "heihe_x4 90-day only, no heihe_x16").
CELL="heihe_x4"

# Run label encodes the matrix coordinates for the sidecar filenames.
RUN_LABEL="${TASK_ID}-tol${AMG_TOL}-eps${EPSLIN}"

# Export PR-X1 env hooks. AMG_TOL is always exported; EPSLIN is only
# exported when != "default" (preserves SUNDIALS 0.05 default).
export SHUD_AMG_TOL="${AMG_TOL}"
if [[ "${EPSLIN}" != "default" ]]; then
    export SHUD_CVODE_EPSLIN="${EPSLIN}"
else
    # Explicit unset to prevent inheritance from prior task in the same
    # job (defensive — Slurm spawns fresh shells per array task, but
    # double-checks help).
    unset SHUD_CVODE_EPSLIN
fi

# ---------- Banner ----------------------------------------------------------
echo "=== g0-rca-array task=${TASK_ID} cell=${CELL} config=${CONFIG} ==="
echo "node=$(hostname) date_utc=$(date -u +%FT%TZ)"
echo "run_id=${RUN_ID}  run_dir=${RUN_DIR}"
echo "shud_dir=${SHUD_DIR}  tool_dir=${TOOL_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID:-unset} slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo "omp_num_threads=${OMP_NUM_THREADS} shud_linsol=${SHUD_LINSOL}"
echo "shud_amg_tol=${SHUD_AMG_TOL} shud_cvode_epslin=${SHUD_CVODE_EPSLIN:-<unset:default>}"
echo "hypre_libdir=${HYPRE_LIBDIR}"
echo "run_label=${RUN_LABEL}"
echo ""

# ---------- SIGTERM trap (per P8-tune.F M3 + G0 PR-A pattern) --------------
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
        echo "case=${CELL} run_label=${RUN_LABEL} shud_amg_tol=${SHUD_AMG_TOL} shud_cvode_epslin=${SHUD_CVODE_EPSLIN:-default}"
        echo "interp_type=6 coarsen_type=8 NumY=NA nnz_A=NA"
        echo "setup_wall_sec=NA apply_wall_sec=NA peak_rss_bytes=NA"
        echo "cycle_complexity=NA operator_complexity=NA"
        echo "verdict_class=AMG_WALL_OVERFLOW"
        echo "hypre_version=NA colpack_version=absent shud_pin=NA"
        echo "cvode_nfe=NA cvode_nli=NA cvode_nfeLS=NA cvode_ncfn=NA cvode_ncfl=NA cvode_netf=NA"
        echo "amg_telemetry_mean_iters=NA amg_telemetry_mean_op_count=NA amg_telemetry_setup_calls=NA amg_total_setup_wall_sec=NA amg_total_solve_wall_sec=NA"
        echo "n_cvode_steps=NA amg_wall_per_step_sec=NA"
        echo "CELL_SUMMARY_END"
    } 2>/dev/null
    echo "MARKER:AMG_WALL_OVERFLOW_DETECTED emitted (caught SIGTERM at elapsed=${elapsed}s, task=${TASK_ID})" >&2
    exit 0
}
trap emit_wall_overflow TERM

# ---------- Invoke run_rca_cell.sh -----------------------------------------
# RUN_DIR + RUN_LABEL exported above are consumed by run_rca_cell.sh for the
# per-cell .out / .err / .telemetry.tsv sidecar files. cwd is SHUD/ so
# ./shud_omp resolves via the basin-relative path.
cd "${SHUD_DIR}" || { echo "ERROR: cannot cd to ${SHUD_DIR}" >&2; exit 1; }
bash "${TOOL_DIR}/run_rca_cell.sh" "${CELL}" "${AMG_TOL}" "${EPSLIN}" "${RUN_LABEL}"
RC=$?

echo ""
echo "=== task ${TASK_ID} (cell=${CELL} config=${CONFIG}) exit=${RC} ==="
exit ${RC}
