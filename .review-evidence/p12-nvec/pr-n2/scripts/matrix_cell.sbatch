#!/bin/bash
#SBATCH --job-name=p12n2cell
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.p12-nvec-runs/matrix/slurm/%x-%j.out
#SBATCH --error=/scratch/frd_muziyao/SHUD-OpenMP/.p12-nvec-runs/matrix/slurm/%x-%j.err
#
# P12-nvec PR-N2 — ONE matrix cell = one {CONFIG x N} pair, 3 sequential runs,
# on ONE exclusive node (node-exclusivity discipline: never two timed runs on a
# node concurrently). heihe_x4 (NumEle 40046, 90-day). NON-profile wall runs;
# PROF=1 (Config E, N=16 only) is the single gate-on profiler leg.
#
# FULL PER-CELL ISOLATION (concurrent cells run on distinct nodes, zero
# cross-talk, canonical basin NEVER mutated): each cell gets its own CWD with a
# PRIVATE copy of input/heihe_x4/ (12M) + private output/. SHUD resolves
# input/<prj> and output/<prj>.out relative to CWD (CommandIn.cpp/IO.cpp).
#
# N is the SINGLE cfg-numthreads knob: cfg NUM_OPENMP=N (drives Config E's
# N_VNew_OpenMP thread count) AND OMP_NUM_THREADS=N (drives StrictOMP RHS via
# omp_get_max_threads). SHUD_RHS_THREADS UNSET everywhere.
#
# 三铁律: sbatch from /scratch; --output/--error on /scratch; bin+scripts on /scratch.
set -uo pipefail

REPO=/scratch/frd_muziyao/SHUD-OpenMP
BINDIR=$REPO/.p12-nvec-runs/matrix/bin
SRC_BASIN=$REPO/SHUD/Basins/heihe_x4
PROJECT=heihe_x4

CONFIG="${CONFIG:?set CONFIG=C|E}"
N="${N:?set N}"
PROF="${PROF:-0}"
NREPS="${NREPS:-3}"

if [ "$PROF" = "1" ]; then
  BIN=$BINDIR/shud_omp_E_prof
  CELL=${CONFIG}_n${N}_prof
else
  BIN=$BINDIR/shud_omp_${CONFIG}
  CELL=${CONFIG}_n${N}
fi
RESDIR=$REPO/.p12-nvec-runs/matrix/results/$CELL       # collected artifacts
WORK=$REPO/.p12-nvec-runs/matrix/cells/$CELL           # private CWD (input+output)
mkdir -p "$RESDIR" "$WORK"

echo "===== P12-nvec matrix cell $CELL ====="
echo "host=$(hostname)  date=$(date -u +%FT%TZ)  jobid=${SLURM_JOB_ID:-NA}"
echo "bin=$(basename "$BIN")  N=$N  PROF=$PROF  NREPS=$NREPS"
echo "shud=$(cd $REPO/SHUD && git rev-parse --short HEAD)  outer=$(cd $REPO && git rev-parse --short HEAD)"

# ---- private input copy (once) + cfg NUM_OPENMP=N ----
rm -rf "$WORK/input" "$WORK/output"
mkdir -p "$WORK/output"
cp -a "$SRC_BASIN/input" "$WORK/input"
CFG="$WORK/input/$PROJECT/$PROJECT.cfg.para"
# rewrite ONLY NUM_OPENMP value to N (tab-separated key\tvalue); 90-day window untouched
awk -v n="$N" 'BEGIN{OFS="\t"} $1=="NUM_OPENMP"{$2=n; print; next} {print}' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"
echo "cfg NUM_OPENMP -> $(grep -E '^[[:space:]]*NUM_OPENMP' "$CFG")"
echo "cfg window: $(awk '/^[[:space:]]*START/{s=$2}/^[[:space:]]*END/{e=$2}END{print "START="s" END="e" delta="e-s" (MUST be 90)"}' "$CFG")"

# ---- env: production Config C/E + N knob ----
export SHUD_LINSOL=spgmr
unset SHUD_AMG_TOL SHUD_CVODE_EPSLIN SHUD_CVODE_RELTOL SHUD_RHS_THREADS
export OMP_NUM_THREADS="$N" OMP_PROC_BIND=close OMP_PLACES=cores
if [ "$PROF" = "1" ]; then export SHUD_NVEC_PROF=1; else unset SHUD_NVEC_PROF; fi
export LIBRARY_PATH=/scratch/frd_muziyao/local/openblas/lib:/scratch/frd_muziyao/local/usr/lib/x86_64-linux-gnu/openmpi/lib:/scratch/frd_muziyao/local/usr/lib/x86_64-linux-gnu:/scratch/frd_muziyao/local/hypre-3.1.0/lib:$REPO/SHUD/InstallSundials/lib
export LD_LIBRARY_PATH="$LIBRARY_PATH"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS SHUD_RHS_THREADS=${SHUD_RHS_THREADS:-<unset>} SHUD_NVEC_PROF=${SHUD_NVEC_PROF:-<unset>} SHUD_LINSOL=$SHUD_LINSOL"

OUT="$WORK/output/$PROJECT.out"

run_one() {
  local rep="$1"
  local rout="$RESDIR/run${rep}.out" rerr="$RESDIR/run${rep}.err"
  local t0 t1 wall rc
  t0=$(date +%s.%N)
  ( cd "$WORK" && "$BIN" "$PROJECT" >"$rout" 2>"$rerr" )
  rc=$?
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
  cp -f "$OUT/cvode_stats.txt" "$RESDIR/cvode_stats.run${rep}.txt" 2>/dev/null || echo "WARN: no cvode_stats.txt rep$rep"
  local rivsha
  rivsha=$(sha256sum "$OUT/$PROJECT.rivqdown.dat" 2>/dev/null | awk '{print $1}')
  echo "$rivsha  $PROJECT.rivqdown.dat" > "$RESDIR/rivqdown.run${rep}.sha"
  if [ "$PROF" = "1" ]; then
    cp -f "$OUT/nvec_prof.csv"   "$RESDIR/nvec_prof.run${rep}.csv"   2>/dev/null || echo "WARN: no nvec_prof.csv rep$rep"
    cp -f "$OUT/profile_B0.yaml" "$RESDIR/profile_B0.run${rep}.yaml" 2>/dev/null || echo "WARN: no profile_B0.yaml rep$rep"
  fi
  local nst nfe ncfn ncfl netf
  nst=$(awk -F= '/^nst=/{print $2}'  "$RESDIR/cvode_stats.run${rep}.txt" 2>/dev/null)
  nfe=$(awk -F= '/^nfe=/{print $2}'  "$RESDIR/cvode_stats.run${rep}.txt" 2>/dev/null)
  ncfn=$(awk -F= '/^ncfn=/{print $2}' "$RESDIR/cvode_stats.run${rep}.txt" 2>/dev/null)
  ncfl=$(awk -F= '/^ncfl=/{print $2}' "$RESDIR/cvode_stats.run${rep}.txt" 2>/dev/null)
  netf=$(awk -F= '/^netf=/{print $2}' "$RESDIR/cvode_stats.run${rep}.txt" 2>/dev/null)
  echo "MARKER:P12N2 cell=$CELL rep=$rep wall_s=$wall rc=$rc nst=$nst nfe=$nfe ncfn=$ncfn ncfl=$ncfl netf=$netf rivqdown_sha=$rivsha"
}

# startup banner sanity (first rep stdout carries the config identity line)
date -u +"start_utc=%FT%TZ"
for r in $(seq 1 "$NREPS"); do
  echo "--- rep $r/$NREPS ---"
  run_one "$r"
done
# echo the config identity + thread topology from rep1 stdout
echo "--- config identity (rep1) ---"
grep -m1 -E 'openMP NVector:|NVEC_HYBRID.*installed|P1e startup' "$RESDIR/run1.out" | sed 's/^/  /' || true
date -u +"end_utc=%FT%TZ"
# tidy: drop the private input/output to reclaim scratch (artifacts already in RESDIR)
rm -rf "$WORK/input" "$WORK/output"
echo "CELL_DONE $CELL"
