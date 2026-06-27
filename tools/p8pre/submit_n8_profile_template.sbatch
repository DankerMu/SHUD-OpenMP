#!/bin/bash
# tools/p8pre/submit_n8_profile_template.sbatch
#
# p8pre-spike Step 1 PR-A (issue #339/#340) — Slurm sbatch TEMPLATE for one
# (case, N, rep) cell of the 18-cell N=8 Mode C profile recheck matrix
# (heihe + heihe_x4 × N ∈ {1,4,8} × 3 reps; N=2 skipped per design D4).
#
# Substitution markers (render_n8_profile.sh wrapper SHALL replace before
# `sbatch <rendered>.sbatch` invocation from a working dir under /scratch):
#   __CASE__  one of heihe | heihe_x4
#   __N__     1 | 4 | 8 (N=2 explicitly NOT submitted, per design D4)
#   __REP__   1 | 2 | 3
#   __NODE__  cn14 (heihe stream) | cn15 (heihe_x4 stream)
#
# Differences from tools/p1e_2x2_sbatch_template.sbatch prototype:
#   - NO __BUILD__ marker — always Mode C (Serial NVec + StrictOMP RHS),
#     build target == `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`.
#   - __NODE__ parameterized (wrapper sets cn14 for heihe, cn15 for heihe_x4
#     per P1e PR-I deployment SOP — keeps cross-cell CPU SKU comparable).
#   - Per-cell artifact directory under
#     /scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/<case>_N<n>_rep<r>/
#     instead of the p1e 2x2 nested layout.
#   - Singleton chain per (case, N) via --dependency=afterany rendered by
#     wrapper so 3 reps run serially per (case, N) — avoids intra-host
#     contention on the pinned node.
#
# Slurm 三铁律 compliance (CLAUDE.md):
#   1. sbatch is launched from /scratch (wrapper SHALL cd there first;
#      this template's #SBATCH paths are pinned to /scratch below).
#   2. --output / --error paths are under /scratch (compute node /tmp is
#      node-local and gets wiped → sacct ExitCode 127). Logs go to the
#      same per-cell directory as the artifacts.
#   3. All referenced artifacts (SHUD binary, cfg.para) live under /scratch
#      via the SHUD-OpenMP working tree at /scratch/frd_muziyao/SHUD-OpenMP.
#
# Reference style: tools/p1e_2x2_sbatch_template.sbatch (proven on cn05/cn14).

# ----------------------------------------------------------------------------
# Slurm directives
# ----------------------------------------------------------------------------

#SBATCH --job-name=p8pre_n8_profile___CASE___N__N___rep__REP__
#SBATCH --partition=CPU
#SBATCH --nodelist=__NODE__
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__N__
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/__CASE___N__N___rep__REP__/slurm.out
#SBATCH --error=/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/__CASE___N__N___rep__REP__/slurm.err

# ----------------------------------------------------------------------------
# Environment + reproducibility
# ----------------------------------------------------------------------------

set -euo pipefail

REPO=/scratch/frd_muziyao/SHUD-OpenMP
SHUD_DIR="$REPO/SHUD"
CELL_DIR="$REPO/.p8pre-runs/__CASE___N__N___rep__REP__"

mkdir -p "$CELL_DIR"
cd "$SHUD_DIR"

# Determinism guards: deterministic NUMA + thread binding so cross-N
# bitwise comparisons aren't subject to placement noise (per CLAUDE.md
# §"Slurm 三铁律" + P1e era findings).
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_NUM_THREADS=__N__
# Honored by StrictOMP RHS region (per P1e PR-H design D3).
export SHUD_RHS_THREADS=__N__

# Provenance: log what we're about to do so logs/__CASE__*.out is
# self-contained when sacct refers to it.
echo "[p8pre_n8_profile] cell=__CASE___N__N___rep__REP__"
echo "[p8pre_n8_profile] host=$(hostname) numa=$(numactl --show 2>/dev/null | head -1 || echo n/a)"
echo "[p8pre_n8_profile] outer_git=$(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "[p8pre_n8_profile] SHUD pin=$(git rev-parse --short HEAD)"
echo "[p8pre_n8_profile] OMP_NUM_THREADS=$OMP_NUM_THREADS SHUD_RHS_THREADS=$SHUD_RHS_THREADS"
echo "[p8pre_n8_profile] OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
date -u +"[p8pre_n8_profile] start_utc=%Y-%m-%dT%H:%M:%SZ"

# ----------------------------------------------------------------------------
# Cell execution: ./shud __CASE__ from SHUD/ working dir (binary is the
# Mode C build prepared in PR-A pre-flight task §1.2 by a separate one-shot
# `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`; this template
# does NOT rebuild per-cell to keep wall comparable across cells).
# ----------------------------------------------------------------------------

# Bracket with `set +e ... set -e` so the wrapper always logs run_exit_code
# (cell triage relies on this line in slurm.out).
set +e
./shud __CASE__
RC=$?
set -e

date -u +"[p8pre_n8_profile] end_utc=%Y-%m-%dT%H:%M:%SZ"
echo "[p8pre_n8_profile] run_exit_code=$RC"

# ----------------------------------------------------------------------------
# Artifact harvest into per-cell dir (Slurm 三铁律: under /scratch)
# Source layout (Mode C profile build): SHUD/Basins/<case>/output/__CASE__/
# - profile.yaml          (timer.cpp emits at SHUD_ENABLE_PROFILE=1 build)
# - cvode_stats.txt       (PrintFinalStats — SHUD canonical 15 keys)
# - rivqdown.dat          (river discharge — gate-1 of Step 2 SHA12 baseline)
# ----------------------------------------------------------------------------

# Per CLAUDE.md case-deployment-map §2.2: SHUD project name == basin folder
# name for heihe and heihe_x4 (no aliasing). Output dir convention follows
# SHUD/src/Model/shud.cpp output prefix.
OUTPUT_DIR="$SHUD_DIR/Basins/__CASE__/output/__CASE__"

for f in profile.yaml cvode_stats.txt rivqdown.dat; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        cp "$OUTPUT_DIR/$f" "$CELL_DIR/$f"
    else
        echo "[p8pre_n8_profile] WARN: $f missing in $OUTPUT_DIR"
    fi
done

# slurm.out / slurm.err already written into $CELL_DIR via #SBATCH paths above.
echo "[p8pre_n8_profile] artifacts copied to $CELL_DIR"
exit $RC
