#!/bin/bash
# tools/p8tune.D/spike_array.sbatch
#
# P8-tune.D KLU pattern-only spike (PR-A). 16-cell Slurm array dispatcher
# for the 4-case × 4-ordering matrix per openspec change `p8tune-klu-spike`
# spec REQ-4 (16-cell matrix) + REQ-5 (3-axis verdict) + REQ-7 (PR-A
# boundary).
#
# Matrix (16 cells; case-major, ordering-minor). Per spec REQ-4 "4-combo
# definition" L94-99: the 4 orderings SHALL be (natural,+BTF), (AMD,-BTF),
# (AMD,+BTF), (COLAMD,+BTF) exactly.
#
#   case_idx=0  keliya      NN=0  natural btf=1
#                           NN=1  amd     btf=0
#                           NN=2  amd     btf=1
#                           NN=3  colamd  btf=1
#   case_idx=1  heihe       NN=4  natural btf=1
#                           NN=5  amd     btf=0
#                           NN=6  amd     btf=1
#                           NN=7  colamd  btf=1
#   case_idx=2  heihe_x4    NN=8  natural btf=1
#                           NN=9  amd     btf=0
#                           NN=10 amd     btf=1
#                           NN=11 colamd  btf=1
#   case_idx=3  heihe_x16   NN=12 natural btf=1
#                           NN=13 amd     btf=0
#                           NN=14 amd     btf=1
#                           NN=15 colamd  btf=1
#
# Slurm 三铁律 compliance (per CLAUDE.md):
#   1) sbatch SHALL be invoked from /scratch/.../  (NOT /users/...)
#   2) --output / --error point to /scratch/...   (NOT /tmp/; compute node
#      /tmp is node-local + ephemeral, ExitCode 127 risk)
#   3) all referenced scripts (run_cell.sh, spike_run.sh) live under
#      /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.D/
#
# Memory: --mem=170G sized just under cn14's 173 GiB cap per cn_node_ram.h
# (CN_NODE_RAM_BYTES=185528156160 bytes / 1024^3 ≈ 172.8 GiB; 170G leaves
# 2.8 GiB headroom for OS overhead + our 0.7× safety multiplier in
# klu_analyze_factor's preflight). Smaller cases under-utilize but
# cell-uniform job-spec is KISS.
#
# Wall: --time=08:00:00 covers heihe_x16 numeric-factor probe (some cells
# may OOM after 1-2h; OOM exits 0 per REQ-5 OOM-as-data-point). Smaller
# cases finish in minutes.
#
# Submit (from server):
#   RUN_ID="pr-a-$(date -u +%Y%m%d-%H%M%S)"
#   mkdir -p /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/${RUN_ID}
#   cd      /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/${RUN_ID}
#   sbatch --export=ALL,RUN_ID=${RUN_ID} \
#       /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.D/spike_array.sbatch
#
# The --output/--error directives below use bare `cell-%a.out` / `cell-%a.err`
# (NOT absolute paths). Slurm resolves the bare names relative to the
# submission cwd — which the submitter sets to ${RUN_DIR} per 三铁律 rule 1
# above. This keeps the per-RUN_ID dir self-contained AND avoids the
# `%x.%A` substitution trap (Slurm does NOT auto-create the substituted
# dir before opening the file, so an absolute path with `%x.%A` would race
# the submit-time mkdir).
#
# Output (per array task NN, flat layout per REQ-7 PR-A boundary L218):
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/${RUN_ID}/cell-NN.out
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/${RUN_ID}/cell-NN.err
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/${RUN_ID}/cell-NN.log
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/${RUN_ID}/cell-NN.J.bin
#
# References:
#   - spec   : openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-4, REQ-5, REQ-7
#   - tasks  : openspec/changes/p8tune-klu-spike/tasks.md §2.4 (this file), §2.5 (run_cell.sh)
#   - issue  : https://github.com/DankerMu/SHUD-OpenMP/issues/381
#   - PR-0   : commit 09650ed (spike tool authoring)
#
#SBATCH --job-name=p8d_klu_spike
#SBATCH --partition=CPU
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=170G
#SBATCH --time=08:00:00
#SBATCH --output=cell-%a.out
#SBATCH --error=cell-%a.err

set -uo pipefail

# ---------- RUN_ID resolution -----------------------------------------------
# Prefer the externally-injected RUN_ID (--export=RUN_ID=...) so the per-run
# scratch directory matches what the submit-time `mkdir -p` created. If
# unset, fall back to job-id-based dir (matches the Slurm %x.%A pattern in
# the --output/--error directives above).
RUN_ID="${RUN_ID:-${SLURM_JOB_NAME:-p8d_klu_spike}.${SLURM_ARRAY_JOB_ID:-0}}"
RUN_DIR="/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

# ---------- Decode SLURM_ARRAY_TASK_ID -> (case, ordering, btf) -------------
# Enumeration: NN = case_idx * 4 + ordering_idx, where ordering_idx walks
# (natural, +BTF), (AMD, -BTF), (AMD, +BTF), (COLAMD, +BTF) per spec REQ-4
# "4-combo definition" L94-99.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

CASE_ARR=(keliya heihe heihe_x4 heihe_x16)
# Per-ordering pair: (ordering, btf). Index 0..3 within a case block.
ORDERING_ARR=(natural amd amd colamd)
BTF_ARR=(1 0 1 1)

CASE_IDX=$((TASK_ID / 4))
ORD_IDX=$((TASK_ID % 4))

if [[ ${CASE_IDX} -lt 0 || ${CASE_IDX} -gt 3 ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,15]" >&2
    exit 1
fi

CASE="${CASE_ARR[${CASE_IDX}]}"
ORDERING="${ORDERING_ARR[${ORD_IDX}]}"
BTF="${BTF_ARR[${ORD_IDX}]}"

# Zero-pad NN for the cell-NN.* layout (NN ∈ {00..15}).
printf -v NN "%02d" "${TASK_ID}"

echo "=== p8d_klu_spike array task ${TASK_ID} (NN=${NN}) ==="
echo "decoded: case=${CASE} ordering=${ORDERING} btf=${BTF}"
echo "node=$(hostname) date_utc=$(date -u +%FT%TZ)"
echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID:-unset} slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
echo ""

# ---------- Delegate to per-cell wrapper -------------------------------------
bash /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.D/run_cell.sh \
    "${CASE}" "${ORDERING}" "${BTF}" "${NN}" "${RUN_ID}"
RC=$?

echo ""
echo "=== task ${TASK_ID} (NN=${NN}) exit=${RC} ==="

# Exit semantics (per spec REQ-5 OOM-as-data-point):
#   - rc=0 : PASS (or OOM cleanly reported via KLU_OOM_DETECTED line)
#   - rc=1 : real failure (dump_adjacency / fd_color_jacobian aborted,
#            or klu_analyze_factor errored without OOM detection)
# Slurm does NOT --requeue on rc=1 (no retry; per REQ-5 OOM is decisive
# data, and real failures need root-cause debugging not blind retry).
exit ${RC}
