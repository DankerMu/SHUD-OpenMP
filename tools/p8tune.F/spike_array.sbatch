#!/bin/bash
# tools/p8tune.F/spike_array.sbatch
#
# P8-tune.F BoomerAMG/Hypre pattern-only spike (PR-B). 16-cell Slurm
# array dispatcher for the 4-case × 4-(interp_type, coarsen_type)-combo
# matrix per openspec change `p8tune-amg-spike` spec REQ-4 (16-cell
# matrix) + REQ-5 (5-axis verdict) + issue #396 §In Scope.
#
# Matrix (16 cells; case-major NN/4, combo-minor NN%4). Per spec REQ-4
# Scenario "4 case × 4 combo matrix":
#
#   case_idx=0  keliya       NN=00  (interp=6,  coarsen=8)   Hypre default
#                            NN=01  (interp=14, coarsen=10)  aggressive
#                            NN=02  (interp=6,  coarsen=21)  alt-coarsening
#                            NN=03  (interp=8,  coarsen=8)   fallback
#   case_idx=1  heihe        NN=04  (interp=6,  coarsen=8)
#                            NN=05  (interp=14, coarsen=10)
#                            NN=06  (interp=6,  coarsen=21)
#                            NN=07  (interp=8,  coarsen=8)
#   case_idx=2  heihe_x4     NN=08  (interp=6,  coarsen=8)
#                            NN=09  (interp=14, coarsen=10)
#                            NN=10  (interp=6,  coarsen=21)
#                            NN=11  (interp=8,  coarsen=8)
#   case_idx=3  heihe_x16    NN=12  (interp=6,  coarsen=8)
#                            NN=13  (interp=14, coarsen=10)
#                            NN=14  (interp=6,  coarsen=21)
#                            NN=15  (interp=8,  coarsen=8)
#
# Slurm 三铁律 compliance (per CLAUDE.md):
#   1) sbatch SHALL be invoked from /scratch/.../.p8tune.F-runs/<RUN_ID>/
#   2) --output / --error use bare relative names (`cell-%a.out` /
#      `cell-%a.err`) which Slurm resolves against the submission cwd
#      (= /scratch/.../.p8tune.F-runs/<RUN_ID>/) — NEVER /tmp
#   3) all referenced scripts (run_cell.sh, boomeramg_setup_solve,
#      dump_adjacency, fd_color_jacobian) live under
#      /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.F/ (or sibling
#      /scratch/.../tools/p8tune.D/ via symlinks for the reused
#      P8-tune.D pre-flight binaries)
#
# Memory: --mem=120G chosen for 173 GiB cn-node cap with ample headroom
# for AMG hierarchy peak (large coarsening factors can balloon temp
# memory mid-setup); BoomerAMG `_OOM_DETECTED` marker classifies the
# cell as a clean data point if peak RSS exceeds 0.7×173 ≈ 121 GiB
# threshold per spec REQ-5 Axis 3.
#
# Wall: --time=08:00:00 covers heihe_x16 worst-case setup+solve
# (combo_for(1) (14,10) is aggressive interp + HMIS, can balloon
# operator complexity; combo_for(2) (6,21) is CGC coarsening, slower
# setup). Wall exceedance → SIGTERM → trap → MARKER:AMG_WALL_OVERFLOW
# emitted → cell exits 0 as a clean wall_overflow data point.
#
# CPUs: --cpus-per-task=1 (this spike is sequential; BoomerAMG's
# internal OMP is left disabled by the cpp via setting OMP_NUM_THREADS=1
# inside boomeramg_setup_solve to keep wall measurements deterministic +
# comparable across the 16-cell sweep; OMP-scaling exploration is
# explicitly out of scope for PR-B per issue #396).
#
# Submit (from server):
#   RUN_ID="pr-b-$(date -u +%Y%m%d-%H%M%S)"
#   mkdir -p /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}
#   cd      /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}
#   sbatch --export=ALL,RUN_ID=${RUN_ID} \
#       /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.F/spike_array.sbatch
#
# Output (per array task NN, flat layout):
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}/cell-NN.out
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}/cell-NN.err
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}/cell-NN.log
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}/cell-NN.time
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}/cells/cell-NN.log  (post-completion copy)
#
# References:
#   - spec : openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md REQ-4, REQ-5
#   - tasks: openspec/changes/p8tune-amg-spike/tasks.md PR-B §3.4
#   - issue: https://github.com/DankerMu/SHUD-OpenMP/issues/396
#
#SBATCH --job-name=p8f_amg_spike
#SBATCH --partition=CPU
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=cell-%a.out
#SBATCH --error=cell-%a.err

set -uo pipefail

# ---------- PATH ------------------------------------------------------------
# Compute nodes don't source ~/.bashrc for non-interactive sbatch; explicitly
# prepend ~/.local/bin so `uv` (CLAUDE.md "Python 一律用 uv") is on PATH
# if the cpp ever shells out to a Python helper. Harmless if absent.
export PATH="${HOME}/.local/bin:${PATH}"

# Pin OMP_NUM_THREADS=1 — sequential measurements per cell so wall times
# are comparable across the 16-cell sweep. The cpp does this internally
# too, but belt-and-suspenders.
export OMP_NUM_THREADS=1

# ---------- RUN_ID resolution -----------------------------------------------
# Prefer the externally-injected RUN_ID (--export=RUN_ID=...) so the
# per-run scratch directory matches what the submit-time mkdir created.
# Fall back to job-id-based dir if unset.
RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-p8f_amg_spike}.${SLURM_ARRAY_JOB_ID:-0}}"
RUN_DIR="/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/cells"

# ---------- Decode SLURM_ARRAY_TASK_ID -> (NN, CASE, INTERP, COARSEN) ------
# All decoding for the cpp invocation is delegated to run_cell.sh; here
# we just need NN (zero-padded) for the log promotion step below.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ${TASK_ID} -lt 0 || ${TASK_ID} -gt 15 ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,15]" >&2
    exit 1
fi
printf -v NN "%02d" "${TASK_ID}"

CELL_LOG="${RUN_DIR}/cell-${NN}.log"

echo "=== p8f_amg_spike array task ${TASK_ID} (NN=${NN}) ==="
echo "node=$(hostname) date_utc=$(date -u +%FT%TZ)"
echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID:-unset} slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo ""

# ---------- SIGTERM trap (spec REQ-4 SIGTERM → AMG_WALL_OVERFLOW marker) ---
# When Slurm hits the cell's --time=08:00:00 wall budget, it sends SIGTERM,
# waits KillWait seconds (default 30), then SIGKILL. The trap catches
# SIGTERM and emits MARKER:AMG_WALL_OVERFLOW_DETECTED to cell-NN.log
# stdout on a best-effort basis so the PR-C aggregator can classify the
# cell as AMG_WALL_OVERFLOW rather than a real Slurm failure. If the trap
# window is insufficient (e.g., KillWait truncated), the aggregator
# falls back to sacct State=TIMEOUT detection.
START_TS=$(date +%s)
emit_wall_overflow() {
    local now elapsed wall_budget_sec
    now=$(date +%s)
    elapsed=$((now - START_TS))
    wall_budget_sec="unknown"
    if [[ "${SLURM_JOB_END_TIME:-}" != "" && "${SLURM_JOB_START_TIME:-}" != "" ]]; then
        wall_budget_sec=$((SLURM_JOB_END_TIME - SLURM_JOB_START_TIME))
    fi
    {
        echo ""
        echo "MARKER:AMG_WALL_OVERFLOW_DETECTED elapsed_sec=${elapsed} wall_budget_sec=${wall_budget_sec}"
        # Best-effort cell_summary stub so PR-C parser sees verdict_class
        # even when the cpp main thread is killed before emit_cell_summary.
        echo "CELL_SUMMARY_BEGIN"
        echo "verdict_class=AMG_WALL_OVERFLOW"
        echo "setup_wall_sec=NA apply_wall_sec=NA peak_rss_bytes=NA"
        echo "wall_budget_sec=${wall_budget_sec} elapsed_sec=${elapsed}"
        echo "CELL_SUMMARY_END"
    } >> "${CELL_LOG}" 2>/dev/null
    echo "MARKER:AMG_WALL_OVERFLOW_DETECTED emitted to ${CELL_LOG} (caught SIGTERM at elapsed=${elapsed}s)" >&2
    # Copy the partial log into cells/ for evidence collection (same
    # promotion the success path does at the bottom of this script).
    cp -p "${CELL_LOG}" "${RUN_DIR}/cells/cell-${NN}.log" 2>/dev/null || true
    # Exit 0 so Slurm sees the cell as a clean wall_overflow data point.
    exit 0
}
trap emit_wall_overflow TERM

# ---------- Delegate to per-cell wrapper -------------------------------------
bash /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.F/run_cell.sh "${NN}" "${RUN_ID}"
RC=$?

# ---------- Post-completion evidence promotion ------------------------------
# Copy cell-NN.log into cells/ subdir for evidence collection (PR-B
# acceptance §3.9 rsync evidence). Also keep the original at run dir
# top-level for ergonomic browsing during the sweep.
if [[ -s "${CELL_LOG}" ]]; then
    cp -p "${CELL_LOG}" "${RUN_DIR}/cells/cell-${NN}.log" 2>/dev/null || \
        echo "WARN: failed to copy ${CELL_LOG} → ${RUN_DIR}/cells/ (continuing)" >&2
fi

echo ""
echo "=== task ${TASK_ID} (NN=${NN}) exit=${RC} ==="

# Exit semantics (per spec REQ-4 marker-as-data-point):
#   rc=0 : PASS, or marker emission (AMG_OOM / AMG_SETUP_DIVERGE /
#          AMG_SOLVE_DIVERGE / AMG_WALL_OVERFLOW from trap above)
#   rc=1 : real failure (pre-flight aborted, or cpp errored without
#          any recognized marker)
# Slurm does NOT --requeue on rc=1; data points are decisive, and real
# failures need root-cause debugging not blind retry.
exit ${RC}
