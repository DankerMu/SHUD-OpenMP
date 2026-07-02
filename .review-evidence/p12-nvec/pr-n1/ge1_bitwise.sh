#!/usr/bin/env bash
# P12-nvec PR-N1 G-E1 HARD gate: Config E keliya four-leg cross-thread
# bitwise evidence (N in {1,2,4,8}). Every leg's model-output SHA manifest
# must be identical to EACH OTHER and to the Config C baseline manifest
# (reused from PR-N0 leg-a-baseline.sha; serial==Config C bitwise per P1e
# A3a chains the equivalence). No tolerance fallback.
#
# N is the SINGLE cfg-numthreads knob (NUM_OPENMP in keliya.cfg.para),
# which drives BOTH the StrictOMP RHS threads and the OpenMP-NVector
# threads (N_VNew_OpenMP(NY, MD->CS.num_threads, ...)). OMP_NUM_THREADS is
# set coherently to the same N and both are RECORDED per leg.
# SHUD_RHS_THREADS is UNSET in every leg (so it never overrides N).
#
# Run from repo root AFTER building Config E:
#   make -C SHUD shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1
set -uo pipefail

REPO="/Users/danker/Desktop/Hydro-SHUD/openMP"
SHUD="$REPO/SHUD"
BASIN="$SHUD/Basins/keliya"
OUT="$BASIN/output/keliya.out"
CFG="$BASIN/input/keliya/keliya.cfg.para"
BIN="$SHUD/shud_omp"
EVID="$REPO/.review-evidence/p12-nvec/pr-n1"
SHADIR="$EVID/sha"
BASELINE="$REPO/.review-evidence/p12-nvec/pr-n0/sha/leg-a-baseline.sha"
mkdir -p "$SHADIR"

# Manifest = sha256 of every model-output file EXCEPT the profiling /
# instrumentation artifacts (identical exclusion set to PR-N0):
#   diag_*.csv, nvec_prof.csv, profile_B0.yaml, *.time.csv
manifest() {
    local dst="$1"
    ( cd "$OUT" && \
      find . -maxdepth 1 -type f \
        ! -name 'diag_*.csv' \
        ! -name 'nvec_prof.csv' \
        ! -name 'profile_B0.yaml' \
        ! -name '*.time.csv' \
        -print | LC_ALL=C sort | xargs shasum -a256 ) > "$dst"
}

# Save original cfg NUM_OPENMP so we restore it at the end.
ORIG_NUM=$(grep -E '^NUM_OPENMP' "$CFG" | awk '{print $2}')
echo "original NUM_OPENMP=$ORIG_NUM"
restore_cfg() {
    # portable in-place sed (BSD + GNU): rewrite NUM_OPENMP line
    LC_ALL=C sed -i.bak -E "s/^NUM_OPENMP[[:space:]].*/NUM_OPENMP\t$ORIG_NUM/" "$CFG"
    rm -f "$CFG.bak"
}
trap restore_cfg EXIT

set_num() {
    LC_ALL=C sed -i.bak -E "s/^NUM_OPENMP[[:space:]].*/NUM_OPENMP\t$1/" "$CFG"
    rm -f "$CFG.bak"
}

echo "=== Config E binary identity ==="
strings "$BIN" | grep -m1 'NVEC_HYBRID.*serial reduction overrides installed' \
    && echo "marker: present" || { echo "FAIL: Config E marker absent"; exit 2; }

for N in 1 2 4 8; do
    echo "=== leg N=$N (NUM_OPENMP=$N, OMP_NUM_THREADS=$N, SHUD_RHS_THREADS unset) ==="
    set_num "$N"
    rm -f "$OUT"/nvec_prof.csv 2>/dev/null || true
    ( cd "$BASIN" && unset SHUD_RHS_THREADS; OMP_NUM_THREADS=$N "$BIN" keliya \
        >"$EVID/leg-n$N.stdout.log" 2>"$EVID/leg-n$N.stderr.log" )
    manifest "$SHADIR/leg-n$N.sha"
    # record the actually-compiled thread topology from the run log
    grep -E 'openMP NVector: ON|omp_get_max_threads|P1e startup' "$EVID/leg-n$N.stdout.log" \
        | sed "s/^/  [N=$N] /" || true
done

echo
echo "=== COMPARE (strip nothing; identical hash+name required) ==="
pass=1
ref="$SHADIR/leg-n1.sha"
for N in 1 2 4 8; do
    if diff -q <(LC_ALL=C sort "$ref") <(LC_ALL=C sort "$SHADIR/leg-n$N.sha") >/dev/null; then
        echo "leg-n$N == leg-n1 : IDENTICAL"
    else
        echo "leg-n$N == leg-n1 : MISMATCH"; pass=0
        diff <(LC_ALL=C sort "$ref") <(LC_ALL=C sort "$SHADIR/leg-n$N.sha") | head -20
    fi
done

echo
echo "=== COMPARE vs Config C baseline (PR-N0 leg-a-baseline.sha) ==="
if [ -f "$BASELINE" ]; then
    if diff -q <(LC_ALL=C sort "$BASELINE") <(LC_ALL=C sort "$SHADIR/leg-n1.sha") >/dev/null; then
        echo "leg-n1 == Config C baseline : IDENTICAL"
    else
        echo "leg-n1 == Config C baseline : MISMATCH"; pass=0
        diff <(LC_ALL=C sort "$BASELINE") <(LC_ALL=C sort "$SHADIR/leg-n1.sha") | head -30
    fi
else
    echo "WARN: baseline $BASELINE not found"
fi

echo
if [ "$pass" = "1" ]; then echo "G-E1: PASS (all four legs bitwise-identical to each other AND to Config C baseline)"; else echo "G-E1: FAIL"; exit 1; fi
