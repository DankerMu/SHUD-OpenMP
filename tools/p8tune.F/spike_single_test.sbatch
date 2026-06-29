#!/bin/bash
# tools/p8tune.F/spike_single_test.sbatch
#
# P8-tune.F BoomerAMG/Hypre pattern-only spike (PR-B). SMOKE TEST for
# ONE cell (NN=00 = keliya, interp_type=6, coarsen_type=8 — Hypre
# default combo on the smallest case) to validate the pipeline before
# submitting the full 16-cell array via spike_array.sbatch.
#
# Use this to:
#   - shake out env/path/PATH/Hypre-link issues with a 30-min wall budget
#   - verify SIGTERM trap → MARKER:AMG_WALL_OVERFLOW_DETECTED emission
#     (manually `scancel --signal=TERM <jobid>` to test trap)
#   - validate cell_summary KV block schema before the full sweep
#
# Same Slurm 三铁律 conformance as spike_array.sbatch (rules 1-3):
#   - sbatch invoked from /scratch/.../.p8tune.F-runs/<RUN_ID>/
#   - --output / --error bare relative names resolve under /scratch/
#   - all referenced scripts under /scratch/
#
# Submit (from server):
#   RUN_ID="pr-b-smoke-$(date -u +%Y%m%d-%H%M%S)"
#   mkdir -p /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}
#   cd      /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}
#   sbatch --export=ALL,RUN_ID=${RUN_ID} \
#       /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.F/spike_single_test.sbatch
#
# Output (single cell, NN=00):
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}/cell-00.{out,err,log,time}
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}/cells/cell-00.log
#
# References:
#   - spec : openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md REQ-4
#   - issue: https://github.com/DankerMu/SHUD-OpenMP/issues/396
#
#SBATCH --job-name=p8f_amg_smoke
#SBATCH --partition=CPU
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=cell-%a.out
#SBATCH --error=cell-%a.err

set -uo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export OMP_NUM_THREADS=1

RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-p8f_amg_smoke}.${SLURM_ARRAY_JOB_ID:-0}}"
RUN_DIR="/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/cells"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ${TASK_ID} -ne 0 ]]; then
    echo "ERROR: spike_single_test.sbatch expects --array=0 only, got ${TASK_ID}" >&2
    exit 1
fi
printf -v NN "%02d" "${TASK_ID}"

CELL_LOG="${RUN_DIR}/cell-${NN}.log"

echo "=== p8f_amg_smoke task ${TASK_ID} (NN=${NN}) ==="
echo "node=$(hostname) date_utc=$(date -u +%FT%TZ)"
echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo ""

# SIGTERM trap mirrors spike_array.sbatch for trap-emission smoke test.
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
        echo "CELL_SUMMARY_BEGIN"
        echo "verdict_class=AMG_WALL_OVERFLOW"
        echo "setup_wall_sec=NA apply_wall_sec=NA peak_rss_bytes=NA"
        echo "wall_budget_sec=${wall_budget_sec} elapsed_sec=${elapsed}"
        echo "CELL_SUMMARY_END"
    } >> "${CELL_LOG}" 2>/dev/null
    echo "MARKER:AMG_WALL_OVERFLOW_DETECTED emitted to ${CELL_LOG} (caught SIGTERM at elapsed=${elapsed}s)" >&2
    cp -p "${CELL_LOG}" "${RUN_DIR}/cells/cell-${NN}.log" 2>/dev/null || true
    exit 0
}
trap emit_wall_overflow TERM

# Delegate to the same run_cell.sh used by the full array.
bash /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.F/run_cell.sh "${NN}" "${RUN_ID}"
RC=$?

if [[ -s "${CELL_LOG}" ]]; then
    cp -p "${CELL_LOG}" "${RUN_DIR}/cells/cell-${NN}.log" 2>/dev/null || \
        echo "WARN: failed to copy ${CELL_LOG} → ${RUN_DIR}/cells/" >&2
fi

echo ""
echo "=== smoke task ${TASK_ID} (NN=${NN}) exit=${RC} ==="
exit ${RC}
