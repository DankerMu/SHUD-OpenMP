#!/bin/bash
# tools/p1e_2x2_sbatch_template.sbatch
#
# P1e PR-B (issue #310) — Slurm sbatch TEMPLATE for one 2x2 build matrix
# cell on the server (210.77.77.22:32099 / user frd_muziyao). PR-D will
# materialize the manifest's 48 server cells (96 over both Phase 1+2) by
# substituting __BUILD__ / __CASE__ / __N__ / __REP__ then writing the
# concrete sbatch files to /scratch/frd_muziyao/SHUD-OpenMP/.p1e-runs/2x2/
# (which is gitignored).
#
# Substitution markers (PR-D wrapper SHALL replace before sbatch submit):
#   __BUILD__   one of A|B|C|D
#   __CASE__    one of heihe|heihe_x4  (server cases only — Mac cells run via local driver)
#   __N__       1|2|4|8
#   __REP__     1|2|3
#
# After substitution the file is suitable for `sbatch
# /scratch/.../<rendered>.sbatch` invocation from a working dir under
# /scratch (per Slurm 三铁律 #1: policy bars sbatch from /users/).
#
# Slurm 三铁律 compliance (CLAUDE.md):
#   1. sbatch is launched from /scratch (PR-D wrapper SHALL cd there first;
#      this template itself is path-agnostic but #SBATCH paths are pinned
#      to /scratch below).
#   2. --output / --error paths are under /scratch (compute node /tmp is
#      node-local and gets wiped → sacct ExitCode 127). Logs go to
#      /scratch/frd_muziyao/SHUD-OpenMP/.p1e-runs/2x2/logs/.
#   3. All referenced artifacts (driver script, SHUD binary, cfg.para)
#      live under /scratch via the SHUD-OpenMP working tree at
#      /scratch/frd_muziyao/SHUD-OpenMP.
#
# Reference: /scratch/frd_muziyao/NWM/workers/shud_runtime/run_shud_forecast.sbatch
# is the canonical Slurm script style used elsewhere in this environment
# (per CLAUDE.md Slurm 三铁律 section). Mirror its conventions.

# ----------------------------------------------------------------------------
# Slurm directives
# ----------------------------------------------------------------------------

#SBATCH --job-name=p1e_2x2___BUILD____CASE___N__N___rep__REP__
#SBATCH --partition=CPU
#SBATCH --nodelist=cn05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__N__
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.p1e-runs/2x2/logs/p1e_2x2___BUILD____CASE___N__N___rep__REP__.out
#SBATCH --error=/scratch/frd_muziyao/SHUD-OpenMP/.p1e-runs/2x2/logs/p1e_2x2___BUILD____CASE___N__N___rep__REP__.err

# ----------------------------------------------------------------------------
# Environment + reproducibility
# ----------------------------------------------------------------------------

set -euo pipefail

REPO=/scratch/frd_muziyao/SHUD-OpenMP
cd "$REPO"

# Determinism guards: deterministic NUMA + thread binding so cross-N
# bitwise comparisons aren't subject to placement noise (per CLAUDE.md
# §"Slurm 三铁律" + P1d era findings).
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_NUM_THREADS=__N__
# Pre-PR-G mode C/D RHS path falls back on omp_set_num_threads default;
# post-PR-G this env will be honored by the StrictOMP region (per design D3).
export SHUD_RHS_THREADS=__N__

# Provenance: log what we're about to do so logs/__BUILD__*.out is
# self-contained when sacct refers to it.
echo "[p1e_2x2_sbatch] cell=__BUILD____CASE___N__N___rep__REP__"
echo "[p1e_2x2_sbatch] host=$(hostname) numa=$(numactl --show 2>/dev/null | head -1 || echo n/a)"
echo "[p1e_2x2_sbatch] git=$(git rev-parse --short HEAD)"
echo "[p1e_2x2_sbatch] SHUD pin=$(cd SHUD && git rev-parse --short HEAD)"
echo "[p1e_2x2_sbatch] OMP_NUM_THREADS=$OMP_NUM_THREADS SHUD_RHS_THREADS=$SHUD_RHS_THREADS"
echo "[p1e_2x2_sbatch] OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
date -u +"[p1e_2x2_sbatch] start_utc=%Y-%m-%dT%H:%M:%SZ"

# ----------------------------------------------------------------------------
# Cell execution (delegates to driver script)
# ----------------------------------------------------------------------------

tools/p1e_2x2_runner.sh \
    --case __CASE__ \
    --build __BUILD__ \
    --N __N__ \
    --rep __REP__ \
    --out-dir "/scratch/frd_muziyao/SHUD-OpenMP/.p1e-runs/2x2/__BUILD____CASE___N__N___rep__REP__"

RC=$?
date -u +"[p1e_2x2_sbatch] end_utc=%Y-%m-%dT%H:%M:%SZ"
echo "[p1e_2x2_sbatch] driver_exit_code=$RC"
exit $RC
