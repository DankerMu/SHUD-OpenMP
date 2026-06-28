#!/bin/bash
# tools/p8tune/submit_maxl_sweep_template.sbatch
#
# p8tune-spgmr-maxl PR-D — 60-cell maxl sweep submit template.
#
# Submits a Slurm array job: 60 array tasks, one per sweep cell.
#   - matrix: maxl ∈ {5, 10, 15, 20, 30} × case ∈ {heihe, heihe_x4}
#                × N ∈ {1, 8} × rep ∈ {1, 2, 3}  =  5 × 2 × 2 × 3  =  60
#   - concurrency cap: 11 (matches idle CPU node count cn[06,09,14-19,21-22,24])
#   - each task delegates to tools/p8tune/run_maxl_cell.sh
#
# Slurm三铁律 compliance:
#   1) sbatch SHALL be invoked from /scratch/.../p8tune-runs/maxl_sweep/  (not /users/)
#   2) --output / --error point to /scratch/...                          (not /tmp)
#   3) all referenced scripts (run_maxl_cell.sh) live under /scratch/    (tools/p8tune/)
#
# Submit:
#   cd /scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep
#   sbatch /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune/submit_maxl_sweep_template.sbatch
#
# References:
#   - spec  : openspec/changes/p8tune-spgmr-maxl/specs/maxl-sweep-verdict/spec.md L37-53
#   - tasks : openspec/changes/p8tune-spgmr-maxl/tasks.md §4.4, §4.5
#   - PR-C  : .review-evidence/p8tune-pr-c/g3_4way_evidence.tar (run_4way_g3.sbatch pattern)
#
#SBATCH --job-name=p8d_maxl_sweep
#SBATCH --partition=CPU
#SBATCH --array=1-60%11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_slurm/cell_%A_%a.out
#SBATCH --error=/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_slurm/cell_%A_%a.err

set -uo pipefail

# ----- Decode SLURM_ARRAY_TASK_ID -> (case, N, maxl, rep) --------------------
# Enumeration order (most-significant -> least-significant on task_id):
#   case_idx ∈ {0, 1}        — outer dim, 30 cells each (cases: heihe, heihe_x4)
#   N_idx    ∈ {0, 1}        — 15 cells each (N values: 1, 8)
#   maxl_idx ∈ {0, 1, 2, 3, 4} — 3 cells each (maxl values: 5, 10, 15, 20, 30)
#   rep_idx  ∈ {0, 1, 2}     — 1 cell each (reps: 1, 2, 3)
#
# Decoding (task_id ∈ [1, 60], use idx = task_id - 1):
#   case_idx =  idx / 30
#   N_idx    = (idx % 30) / 15
#   maxl_idx = (idx % 15) / 3
#   rep_idx  =  idx % 3
#
# This gives a deterministic 60-row enumeration covering the full
# {heihe, heihe_x4} × {1, 8} × {5, 10, 15, 20, 30} × {1, 2, 3} cross product.

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
IDX=$((TASK_ID - 1))

CASE_ARR=(heihe heihe_x4)
N_ARR=(1 8)
MAXL_ARR=(5 10 15 20 30)
REP_ARR=(1 2 3)

CASE_IDX=$((IDX / 30))
N_IDX=$(((IDX % 30) / 15))
MAXL_IDX=$(((IDX % 15) / 3))
REP_IDX=$((IDX % 3))

CASE="${CASE_ARR[${CASE_IDX}]}"
N="${N_ARR[${N_IDX}]}"
MAXL="${MAXL_ARR[${MAXL_IDX}]}"
REP="${REP_ARR[${REP_IDX}]}"

echo "=== p8d_maxl_sweep array task ${TASK_ID} ==="
echo "decoded: case=${CASE} N=${N} maxl=${MAXL} rep=${REP}"
echo "node=$(hostname) date_utc=$(date -u +%FT%TZ)"
echo ""

# ----- Delegate to per-cell driver -------------------------------------------
bash /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune/run_maxl_cell.sh \
    "${CASE}" "${N}" "${MAXL}" "${REP}"
RC=$?

echo ""
echo "=== task ${TASK_ID} exit=${RC} ==="
exit ${RC}
