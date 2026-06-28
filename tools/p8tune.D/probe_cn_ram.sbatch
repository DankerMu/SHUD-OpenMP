#!/bin/bash
# tools/p8tune.D/probe_cn_ram.sbatch
#
# P8-tune.D KLU pattern-only spike (PR-0). One-shot cn-node RAM probe per
# openspec change `p8tune-klu-spike` spec §REQ-5 Scenario "RSS axis
# threshold": `CN_NODE_RAM_BYTES` SHALL be measured at PR-0 via
# `cat /proc/meminfo` on cn14 (or equivalent representative cn-node) and
# embedded into `tools/p8tune.D/cn_node_ram.h`.
#
# This sbatch is DISTINCT from the PR-A spike_array.sbatch (which runs
# the 16-cell KLU matrix). It does ZERO KLU/CVODE work — pure /proc
# read + machine-identity dump. Safe to land in PR-0 per spec REQ-7
# Scenario "PR-0 tool PR boundary".
#
# Slurm 三铁律 compliance:
#   1) submit from /scratch/.../  (NOT /users/)
#   2) --output/--error in /scratch/  (NOT /tmp; compute node /tmp is node-local + ephemeral)
#   3) sbatch script lives in /scratch/.../tools/p8tune.D/  (referenced from /scratch)
#
# Submit (from server):
#   cd /scratch/frd_muziyao/SHUD-OpenMP/.review-evidence/p8tune-klu-spike-pr-0
#   sbatch /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.D/probe_cn_ram.sbatch
#
# Output:
#   /scratch/frd_muziyao/SHUD-OpenMP/.review-evidence/p8tune-klu-spike-pr-0/cn_ram_probe.log
#
#SBATCH --job-name=cn_ram_probe
#SBATCH --partition=CPU
#SBATCH --nodelist=cn14
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.review-evidence/p8tune-klu-spike-pr-0/cn_ram_probe.log
#SBATCH --error=/scratch/frd_muziyao/SHUD-OpenMP/.review-evidence/p8tune-klu-spike-pr-0/cn_ram_probe.err

set -uo pipefail

echo "=== cn-RAM probe identity ==="
echo "hostname: $(hostname)"
echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "slurm_job_id: ${SLURM_JOB_ID:-unset}"
echo "slurm_nodelist: ${SLURM_NODELIST:-unset}"

echo ""
echo "=== /proc/meminfo ==="
cat /proc/meminfo

echo ""
echo "=== lscpu ==="
lscpu

echo ""
echo "=== numactl --hardware ==="
numactl --hardware || echo "(numactl not installed)"

echo ""
echo "=== probe_cn_ram done ==="
