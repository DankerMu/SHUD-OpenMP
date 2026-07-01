#!/bin/bash
# tools/release_v1.0_omp_scaling/scaling_array.sbatch
#
# Release v1.0 OMP scaling validation — heihe_x4 N in {1,2,4,8,16}
# with A5 hydrology-acceptance gate per non-baseline N.
#
# Motivation: RELEASE.md currently claims "P1e StrictOMP RHS -- ~1.7x
# heihe_x4" from the earlier P1e capstone (a single-point measurement).
# The formal v1.0 release deserves a proper serial-vs-parallel scaling
# table + per-N A5 trajectory-equivalence verification. This 5-cell
# array captures the wall/solve breakdown at each thread count and
# stages the *.out/ trees for downstream A5 comparison against the
# N=1 reference.
#
# 5-cell matrix (heihe_x4, 90-day, SHUD_LINSOL=spgmr, production env):
#   TASK_ID  NTHREADS
#   0        1  (baseline / A5 reference)
#   1        2
#   2        4
#   3        8
#   4        16
#
# Slurm 三铁律 compliance (per CLAUDE.md, mirrors amg_rca_array.sbatch
# + p9_spot_array.sbatch templates):
#   1) sbatch SHALL be invoked from
#        /scratch/frd_muziyao/SHUD-OpenMP/.release-v1.0-scaling-runs/<RUN_ID>/
#      (orchestrator sets up RUN_DIR + cd's before sbatch).
#   2) Slurm meta-logs default to slurm-<jobid>_<task>.out under sbatch
#      invocation cwd (= RUN_DIR per rule 1). Per-cell stdout/stderr are
#      additionally tee'd to ${RUN_DIR}/cell-<TASK>-nthreads-<N>.{out,err}
#      by run_scaling_cell.sh.
#   3) All referenced scripts (run_scaling_cell.sh, shud_omp binary)
#      resolve under /scratch/frd_muziyao/SHUD-OpenMP/.
#
# Node-exclusive (critical): PR-X1 cn23 evidence showed 7-way node
# sharing induced ~2x memory-bandwidth contention noise that would
# invalidate scaling ratios. --exclusive prevents any cross-tenant
# interference on the compute node.
#
# CPU-per-task = 16 to bound the N=16 cell to a single socket / avoid
# NUMA cliff. --cpus-per-task=16 with OMP_PROC_BIND=close + OMP_PLACES=cores
# keeps threads adjacent for cache locality.
#
# Memory: --mem=32G matches heihe_x4 SPGMR envelope (~12 GiB RSS peak
# from G0 PR-A). Node-exclusive means we could take more, but 32G is
# both sufficient and portable.
#
# Wall: --time=03:00:00. SPGMR heihe_x4 N=1 baseline = ~1600s (~26 min);
# N=16 expected ~600s but 3h envelope covers worst-case regression.
# SIGTERM trap emits SCALING_WALL_OVERFLOW marker per PR-X1 convention.
#
# Submit (from server):
#   RUN_ID="release-v1.0-scaling-$(date -u +%Y%m%d-%H%M%S)"
#   mkdir -p /scratch/frd_muziyao/SHUD-OpenMP/.release-v1.0-scaling-runs/${RUN_ID}
#   cd      /scratch/frd_muziyao/SHUD-OpenMP/.release-v1.0-scaling-runs/${RUN_ID}
#   sbatch --export=ALL,RUN_ID=${RUN_ID} \
#       /scratch/frd_muziyao/SHUD-OpenMP/tools/release_v1.0_omp_scaling/scaling_array.sbatch
#
# Output (per array task NN, flat under .release-v1.0-scaling-runs/<RUN_ID>/):
#   cell-<TASK>-nthreads-<N>.out       stdout sidecar
#   cell-<TASK>-nthreads-<N>.err       stderr sidecar
#   cell-<TASK>-nthreads-<N>.output/   copied SHUD *.out/ tree (A5 input)
#
# Refs:
#   - RELEASE.md (v1.0 production manifest, single-point 1.7x claim)
#   - ADR-0010 §CPU acceleration status consolidation
#   - tools/p9.spot/p9_spot_array.sbatch (template)
#   - tools/p8tune.G0-rca/amg_rca_array.sbatch (template, --exclusive precedent)
#   - CLAUDE.md Slurm 三铁律
#
#SBATCH --job-name=release-v1-scaling
#SBATCH --partition=CPU
#SBATCH --array=0-4
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# Intentionally omitting --output / --error: Slurm defaults to
# slurm-<jobid>_<array-task>.out under sbatch invocation cwd
# (= .release-v1.0-scaling-runs/<RUN_ID>/ per 三铁律 rule 1).

set -uo pipefail

# ---------- PATH -----------------------------------------------------------
export PATH="${HOME}/.local/bin:${PATH}"

# ---------- Production environment enforcement -----------------------------
# Explicitly unset all research env hooks so the scaling numbers reflect
# what a production user gets, not a research configuration. Defense-in-
# depth alongside run_scaling_cell.sh.
unset SHUD_AMG_TOL
unset SHUD_CVODE_EPSLIN
unset SHUD_CVODE_RELTOL

# Production solver selector.
export SHUD_LINSOL=spgmr

# OMP thread pinning — close binding + core places keeps threads
# adjacent for cache locality on the exclusive node.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Hypre libdir kept present but inert (SPGMR path does not dlopen Hypre;
# only preserved for parity with prior sbatch templates).
export HYPRE_LIBDIR=/scratch/frd_muziyao/local/hypre-3.1.0/lib
export LD_LIBRARY_PATH="${HYPRE_LIBDIR}:/scratch/frd_muziyao/local/openblas/lib:${LD_LIBRARY_PATH:-}"

# ---------- RUN_ID resolution ----------------------------------------------
RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-release-v1-scaling}.${SLURM_ARRAY_JOB_ID:-0}}"
RUN_DIR="/scratch/frd_muziyao/SHUD-OpenMP/.release-v1.0-scaling-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
export RUN_DIR

SHUD_OPENMP_ROOT="/scratch/frd_muziyao/SHUD-OpenMP"
TOOL_DIR="${SHUD_OPENMP_ROOT}/tools/release_v1.0_omp_scaling"
SHUD_DIR="${SHUD_OPENMP_ROOT}/SHUD"

# ---------- Decode SLURM_ARRAY_TASK_ID -> N -------------------------------
NTHREADS_LIST=(1 2 4 8 16)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ${TASK_ID} -lt 0 || ${TASK_ID} -ge ${#NTHREADS_LIST[@]} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,${#NTHREADS_LIST[@]})" >&2
    exit 1
fi

N="${NTHREADS_LIST[${TASK_ID}]}"
export OMP_NUM_THREADS="${N}"

# ROOT-CAUSE 2026-07-01 (Slurm 10578 v1 + 10601 v2 both flat @ ~1180s):
# SHUD_ENABLE_OPENMP_RHS is a *compile-time* macro (#ifdef in
# SHUD/src/Model/f.cpp L55-L65 + Makefile L140-158 gate), NOT a runtime env.
# The default `make shud_omp HYPRE=1` build produces Config A (Serial NVec +
# Serial RHS) regardless of runtime env, so both v1 (no runtime env) and v2
# (runtime SHUD_ENABLE_OPENMP_RHS=1 exported) showed no speedup.
#
# Fix: rebuild with `make shud_omp SHUD_ENABLE_OPENMP_RHS=1 HYPRE=1 ...`
# which defines -DSHUD_ENABLE_OPENMP_RHS=1 at compile time and gets Config C
# (Serial NVec + StrictOMP RHS). Then at runtime SHUD_RHS_THREADS is the
# canonical RHS thread knob (mode C looks only at this; falls back to
# omp_get_max_threads() if unset — see SHUD src/Model/shud.cpp L154-L166).
export SHUD_RHS_THREADS="${N}"

# ---------- PREFLIGHT: verify shud_omp was built with mode C ---------------
# The `P1e startup: SHUD_RHS_THREADS=...` log-format string only exists in
# the binary if SHUD_RHS_THREADS_STARTUP_LOG was compiled in — a proxy for
# `-DSHUD_ENABLE_OPENMP_RHS=1`. If missing, the binary is Config A and
# scaling would be flat regardless of runtime env; hard-abort so we don't
# waste another Slurm array on a serial binary.
SHUD_OMP_BIN="/scratch/frd_muziyao/SHUD-OpenMP/SHUD/shud_omp"
# NB: use `grep -c ... || true` (count-based) not `grep -q` — with set -uo
# pipefail, grep -q short-circuits on first match, SIGPIPEs `strings`, and
# pipefail flips the pipeline exit code non-zero → false-positive PREFLIGHT_FAIL.
preflight_match_count=$(strings "${SHUD_OMP_BIN}" 2>/dev/null | grep -c "SHUD_RHS_THREADS" || true)
if [[ "${preflight_match_count:-0}" -eq 0 ]]; then
    echo "MARKER:RELEASE_V1_SCALING_PREFLIGHT_FAIL binary=${SHUD_OMP_BIN} missing_symbol=SHUD_RHS_THREADS" >&2
    echo "  Config A (serial RHS) detected — rebuild with:" >&2
    echo "  make shud_omp SHUD_ENABLE_OPENMP_RHS=1 HYPRE=1 HYPRE_INCDIR=... HYPRE_LIBDIR=... OPENBLAS_LIBDIR=..." >&2
    exit 3
fi

# Cell is ALWAYS heihe_x4 (release scaling scope lock).
CELL="heihe_x4"
RUN_LABEL="${TASK_ID}-nthreads-${N}"

# ---------- Banner ---------------------------------------------------------
echo "=== release-v1-scaling task=${TASK_ID} cell=${CELL} nthreads=${N} ==="
echo "node=$(hostname) date_utc=$(date -u +%FT%TZ)"
echo "run_id=${RUN_ID}  run_dir=${RUN_DIR}"
echo "shud_dir=${SHUD_DIR}  tool_dir=${TOOL_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID:-unset} slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo "omp_num_threads=${OMP_NUM_THREADS} shud_linsol=${SHUD_LINSOL}"
echo "shud_rhs_threads=${SHUD_RHS_THREADS} (compile-time SHUD_ENABLE_OPENMP_RHS verified via preflight)"
echo "omp_proc_bind=${OMP_PROC_BIND} omp_places=${OMP_PLACES}"
echo "shud_amg_tol=${SHUD_AMG_TOL:-<unset:production>}"
echo "shud_cvode_epslin=${SHUD_CVODE_EPSLIN:-<unset:production>}"
echo "shud_cvode_reltol=${SHUD_CVODE_RELTOL:-<unset:production>}"
echo "run_label=${RUN_LABEL}"
echo ""

# ---------- SIGTERM trap (per PR-X1 / PR-Z1 template) ---------------------
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
        echo "MARKER:RELEASE_V1_SCALING_WALL_OVERFLOW_DETECTED elapsed_sec=${elapsed} wall_budget_sec=${wall_budget}"
        echo "MARKER:CELL_SCALING_SUMMARY_BEGIN"
        echo "case=${CELL} run_label=${RUN_LABEL} nthreads=${N}"
        echo "wall_total_sec=NA wall_setup_sec=NA wall_solve_sec=NA"
        echo "nst=NA ncfn=NA ncfl=NA"
        echo "omp_place_setting=${OMP_PLACES}"
        echo "verdict_class=SCALING_WALL_OVERFLOW"
        echo "MARKER:CELL_SCALING_SUMMARY_END"
    } 2>/dev/null
    echo "MARKER:RELEASE_V1_SCALING_WALL_OVERFLOW_DETECTED emitted (SIGTERM caught at elapsed=${elapsed}s, task=${TASK_ID})" >&2
    exit 0
}
trap emit_wall_overflow TERM

# ---------- Invoke run_scaling_cell.sh ------------------------------------
# cwd = SHUD/ so ./shud_omp resolves via the basin-relative path.
cd "${SHUD_DIR}" || { echo "ERROR: cannot cd to ${SHUD_DIR}" >&2; exit 1; }
bash "${TOOL_DIR}/run_scaling_cell.sh" "${N}" "${RUN_LABEL}"
RC=$?

echo ""
echo "=== task ${TASK_ID} (cell=${CELL} nthreads=${N}) exit=${RC} ==="
exit ${RC}
