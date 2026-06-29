#!/usr/bin/env bash
# P8-tune.F PR-0 (#394 / #386) — server-side acceptance commands
# Phase 6 round-1 cross-review F6 fix: ① mkdir -p BEFORE cd into runs dir;
# ② --dependency=afterok:$BUILD_JID so valgrind + dump_adj don't silently run
# against a stale or absent binary if the build job fails or hasn't finished.
#
# 由 orchestrator Phase 2 在 Mac local 跑 (ssh tunnel through to server)
# PREREQ: SHUD openmp-baseline 已 push (Mac orchestrator Phase 3-half:
#         cd SHUD && git push origin openmp-baseline) 让 server fetch 到
# DO NOT run from inside this implementer subagent — it is the orchestrator's
# Phase 2 verification step. This script is the runbook.

set -euo pipefail

SSH_HOST="frd_muziyao@210.77.77.22"
SSH_PORT=32099
SCRATCH="/scratch/frd_muziyao/SHUD-OpenMP"
EVIDENCE_DIR=".review-evidence/p8tune-amg-pr-0"
SHUD_FIX_SHA="$(awk '/^  fix SHA:/ {print $3}' ${EVIDENCE_DIR}/shud_fix_sha.txt | tail -1)"

echo "============================================================"
echo "[1/4] Server: pull outer + SHUD submodule update --recursive"
echo "============================================================"
ssh -p "${SSH_PORT}" "${SSH_HOST}" "cd ${SCRATCH} && \
  git fetch origin && \
  git checkout feat/issue-394-p8tune-amg-spike 2>/dev/null || git checkout -b feat/issue-394-p8tune-amg-spike origin/feat/issue-394-p8tune-amg-spike && \
  git pull --recurse-submodules && \
  cd SHUD && git fetch origin && git checkout openmp-baseline && git pull && \
  git rev-parse HEAD"
echo "Expected SHUD HEAD on server: ${SHUD_FIX_SHA}"

echo "============================================================"
echo "[2/4] Server: build SHUD on cn-node via Slurm (NOT login)"
echo "         submit build, capture JID, chain downstream jobs"
echo "         with --dependency=afterok:\$BUILD_JID."
echo "============================================================"
# Phase 6 F6 fix: mkdir -p the runs dir BEFORE `cd` (script previously cd'd
# in before the mkdir which fails on first run / fresh deployment).
ssh -p "${SSH_PORT}" "${SSH_HOST}" "mkdir -p ${SCRATCH}/.p8tune.F-runs && \
  cd ${SCRATCH}/.p8tune.F-runs && \
  cat > build_shud.sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --job-name=p8f-pr0-build
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=${SCRATCH}/.p8tune.F-runs/build_shud_%j.out
#SBATCH --error=${SCRATCH}/.p8tune.F-runs/build_shud_%j.err
set -euo pipefail
cd ${SCRATCH}/SHUD
make clean
./configure
make shud -j4
make libshud.a -j4
make -C ${SCRATCH}/tools/p8tune.D clean
make -C ${SCRATCH}/tools/p8tune.D
echo BUILD_OK
EOF
"

# Capture BUILD JID into a local variable so we can pass --dependency
# to the downstream jobs. --parsable strips the surrounding text and
# emits just the JID (or JID;clustername).
BUILD_JID=$(ssh -p "${SSH_PORT}" "${SSH_HOST}" \
  "cd ${SCRATCH}/.p8tune.F-runs && sbatch --parsable build_shud.sbatch" \
  | awk -F';' '{print $1}')
echo "BUILD_JID=${BUILD_JID}"

echo "============================================================"
echo "[3/4] Server: valgrind ./shud heihe (NumY=19515) on cn-node via Slurm"
echo "         (afterok:\${BUILD_JID}; will not run if build fails)"
echo "============================================================"
ssh -p "${SSH_PORT}" "${SSH_HOST}" "cd ${SCRATCH}/.p8tune.F-runs && \
  cat > valgrind_heihe.sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --job-name=p8f-pr0-vgheihe
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=${SCRATCH}/.p8tune.F-runs/valgrind_heihe_%j.out
#SBATCH --error=${SCRATCH}/.p8tune.F-runs/valgrind_heihe_%j.err
set -euo pipefail
cd ${SCRATCH}/SHUD
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --error-exitcode=42 \\
  ./shud heihe 2>&1
echo VG_DONE
EOF
"
VG_JID=$(ssh -p "${SSH_PORT}" "${SSH_HOST}" \
  "cd ${SCRATCH}/.p8tune.F-runs && sbatch --parsable --dependency=afterok:${BUILD_JID} valgrind_heihe.sbatch" \
  | awk -F';' '{print $1}')
echo "VG_JID=${VG_JID} (depends on BUILD_JID=${BUILD_JID})"

echo "============================================================"
echo "[4/4] Server: dump_adjacency heihe_x4 + heihe_x16 dtor-full smoke on cn-node"
echo "         (afterok:\${BUILD_JID}; runs in parallel with valgrind once build ok)"
echo "============================================================"
ssh -p "${SSH_PORT}" "${SSH_HOST}" "cd ${SCRATCH}/.p8tune.F-runs && \
  cat > dump_adj_x4_x16.sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --job-name=p8f-pr0-dadj
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=${SCRATCH}/.p8tune.F-runs/dump_adj_%j.out
#SBATCH --error=${SCRATCH}/.p8tune.F-runs/dump_adj_%j.err
set -euo pipefail
cd ${SCRATCH}
echo === heihe_x4 NumEle=40046 ===
./tools/p8tune.D/dump_adjacency --case heihe_x4
echo === heihe_x16 NumEle ~160k ===
./tools/p8tune.D/dump_adjacency --case heihe_x16
echo === BOTH_OK — return path exit (no _exit, no SEGV, no free invalid ptr) ===
EOF
"
DA_JID=$(ssh -p "${SSH_PORT}" "${SSH_HOST}" \
  "cd ${SCRATCH}/.p8tune.F-runs && sbatch --parsable --dependency=afterok:${BUILD_JID} dump_adj_x4_x16.sbatch" \
  | awk -F';' '{print $1}')
echo "DA_JID=${DA_JID} (depends on BUILD_JID=${BUILD_JID})"

echo "============================================================"
echo "All 3 jobs submitted (build → valgrind+dump_adj DAG):"
echo "  BUILD_JID=${BUILD_JID}"
echo "  VG_JID=${VG_JID}    (afterok:${BUILD_JID})"
echo "  DA_JID=${DA_JID}    (afterok:${BUILD_JID})"
echo
echo "# RESULTS PENDING — orchestrator polls Slurm and rsyncs logs back"
echo "# AFTER all 3 reach COMPLETED state."
echo
echo "Monitor:"
echo "  ssh -p ${SSH_PORT} ${SSH_HOST} 'squeue -u frd_muziyao'"
echo "Expected acceptance: each .out has the OK sentinel, .err is empty/clean"
echo "  - build_shud: BUILD_OK"
echo "  - valgrind_heihe: VG_DONE + 0 errors + 0 definite/indirect leaks (search ERROR SUMMARY: 0)"
echo "  - dump_adj heihe_x4 + heihe_x16: BOTH_OK sentinel + exit 0 + no 'free(): invalid pointer'"
echo "Rsync logs back to local:"
echo "  rsync -e 'ssh -p ${SSH_PORT}' ${SSH_HOST}:${SCRATCH}/.p8tune.F-runs/*.out ${SCRATCH}/.p8tune.F-runs/*.err ${EVIDENCE_DIR}/"
