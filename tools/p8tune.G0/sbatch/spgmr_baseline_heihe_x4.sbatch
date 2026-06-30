#!/bin/bash
#SBATCH --job-name=g0-spgmr-baseline-heihe_x4
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/heihe_x4-spgmr.%j.out
#SBATCH --error=/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/heihe_x4-spgmr.%j.err
#
# P8-tune.G0 PR-0 §task 4.9 — heihe_x4 SPGMR per-step wall baseline.
# Feeds tools/p8tune.G0/spgmr_baseline_walls_g0.h hot-patch
# (SPGMR_PER_STEP_HEIHE_X4_S sentinel re-pin) and PR-B G0-5 wall-signal gate.
#
# Slurm 三铁律 compliance:
#   ① sbatch invoked from /scratch
#   ② --output/--error on /scratch
#   ③ script itself rsynced to /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.G0/sbatch/
#
# Convention: setup_included (per PR-0 spike 1.2)

set -euo pipefail

CASE=heihe_x4
PROJECT_ROOT=/scratch/frd_muziyao/SHUD-OpenMP
SHUD_DIR=${PROJECT_ROOT}/SHUD
RUN_DIR=${PROJECT_ROOT}/.p8tune.G0-runs/heihe_x4-spgmr-baseline
mkdir -p "${RUN_DIR}"

cd "${SHUD_DIR}"

# Force serial linsol behavior (default-compat invariant should hold under
# SHUD_LINSOL=spgmr OR unset; we set explicit =spgmr for log-explicit attribution).
export SHUD_LINSOL=spgmr
export OMP_NUM_THREADS=1

# Hypre runtime is link-NEEDED via PR-0's Makefile -lHYPRE; expose libdir at
# runtime so ld.so resolves the dependency even on cn-nodes without default
# /scratch in /etc/ld.so.conf.
export HYPRE_LIBDIR=/scratch/frd_muziyao/local/hypre-3.1.0/lib
export LD_LIBRARY_PATH=${HYPRE_LIBDIR}:${LD_LIBRARY_PATH:-}

# Capture environment fingerprint for evidence integrity
{
  echo "=== sbatch env fingerprint ==="
  echo "JOB_ID=${SLURM_JOB_ID:-NA}"
  echo "NODE=$(hostname)"
  echo "DATE=$(date -Iseconds)"
  echo "SHUD_PIN=$(git -C "${SHUD_DIR}" log --oneline -1 HEAD 2>&1)"
  echo "SHUD_BRANCH=$(git -C "${SHUD_DIR}" branch --show-current 2>/dev/null || echo detached)"
  echo "SHUD_LINSOL=${SHUD_LINSOL}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  ldd "${SHUD_DIR}/shud_omp" 2>&1 | grep -E "HYPRE|libmpi|openblas|sundials" || true
} > "${RUN_DIR}/env-fingerprint.txt"

# Run heihe_x4 (NumEle=40046, 90-day SHORT per cfg.para END=91)
START_NS=$(date +%s.%N)
./shud_omp "${CASE}" > "${RUN_DIR}/run.log" 2>&1
RC=$?
END_NS=$(date +%s.%N)

WALL_TOTAL_SEC=$(python3 -c "print(${END_NS} - ${START_NS})")
echo "WALL_TOTAL_SEC=${WALL_TOTAL_SEC}" > "${RUN_DIR}/wall-total.txt"
echo "EXIT_CODE=${RC}" >> "${RUN_DIR}/wall-total.txt"

# Extract n_cvode_steps + n_solves from CVODE stats epilogue
# (cvode_config.cpp writes cvode_stats.txt at end of run)
if [ -f "${SHUD_DIR}/Basins/${CASE}/cvode_stats.txt" ]; then
  cp "${SHUD_DIR}/Basins/${CASE}/cvode_stats.txt" "${RUN_DIR}/cvode_stats.txt"
fi

# Snapshot Output for invariant tracking (manifest.sha256 generation for PR-B)
OUTPUT_DIR="${SHUD_DIR}/Basins/${CASE}/Output"
if [ -d "${OUTPUT_DIR}" ]; then
  cd "${OUTPUT_DIR}"
  sha256sum *.dat > "${RUN_DIR}/manifest.sha256" 2>&1 || true
fi

echo "=== sbatch complete (rc=${RC}) ==="
exit ${RC}
