#!/usr/bin/env bash
# P12-nvec PR-N0 keliya bitwise-neutrality legs (a/b/c/d) + CSV presence.
# Run from repo root. Produces evidence under $EVID.
set -euo pipefail

REPO="/Users/danker/Desktop/Hydro-SHUD/openMP"
SHUD="$REPO/SHUD"
BASIN="$SHUD/Basins/keliya"
OUT="$BASIN/output/keliya.out"
EVID="$REPO/.review-evidence/p12-nvec/pr-n0"
SHADIR="$EVID/sha"
mkdir -p "$SHADIR" "$EVID/csv_sample" "$EVID/selfcheck"

# Manifest = sha256 of every model-output file EXCEPT:
#   diag_*.csv      (P11-osc instrumentation artifacts)
#   nvec_prof.csv   (P12 profiler artifact; only exists when gate on)
#   profile_B0.yaml (SHUD_ENABLE_PROFILE bucket dump; profiling artifact,
#                    not a model output; only exists in PROFILE builds)
#   *.time.csv      (wall-clock profiling log; varies run-to-run)
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

run_keliya() {
    ( cd "$BASIN" && "$1" keliya >/dev/null 2>"$2" )
}

echo "=== Build PRE-PATCH binary (parent commit 75afb2b, commit-based) ==="
cd "$SHUD"
PARENT=75afb2b
# Commit-based pre-patch: check out the parent version of the 3 touched
# paths (shud.cpp) and REMOVE the two new files, build, then restore HEAD.
# The profiler files do not exist at PARENT, so `git checkout PARENT -- <new>`
# would fail; we simply move them aside for the pre-patch build.
mv src/Model/MD_nvec_prof.cpp /tmp/MD_nvec_prof.cpp.hold
mv src/Model/MD_nvec_prof.hpp /tmp/MD_nvec_prof.hpp.hold
git checkout "$PARENT" -- src/Model/shud.cpp
make clean >/dev/null 2>&1
make shud >/tmp/build_prepatch.log 2>&1
cp ./shud /tmp/shud_prepatch
echo "pre-patch shud built (parent $PARENT)."

echo "=== Restore PATCHED (HEAD) tree + build patched binary ==="
git checkout HEAD -- src/Model/shud.cpp
mv /tmp/MD_nvec_prof.cpp.hold src/Model/MD_nvec_prof.cpp
mv /tmp/MD_nvec_prof.hpp.hold src/Model/MD_nvec_prof.hpp
make clean >/dev/null 2>&1
make shud >/tmp/build_patched.log 2>&1
cp ./shud /tmp/shud_patched
echo "patched shud built (HEAD)."

echo "=== leg-a: pre-patch baseline, run #1 and #2 (determinism) ==="
rm -f "$OUT"/diag_*.csv "$OUT"/nvec_prof.csv 2>/dev/null || true
run_keliya /tmp/shud_prepatch "$EVID/leg-a-run1.log"
manifest "$SHADIR/leg-a-run1.sha"
run_keliya /tmp/shud_prepatch "$EVID/leg-a-run2.log"
manifest "$SHADIR/leg-a-baseline.sha"

echo "=== leg-b: patched, SHUD_NVEC_PROF unset ==="
rm -f "$OUT"/nvec_prof.csv 2>/dev/null || true
( cd "$BASIN" && unset SHUD_NVEC_PROF; /tmp/shud_patched keliya >/dev/null 2>"$EVID/leg-b.log" )
manifest "$SHADIR/leg-b-unset.sha"
if [ -f "$OUT/nvec_prof.csv" ]; then echo "LEG-B FAIL: nvec_prof.csv created with gate UNSET" ; else echo "leg-b: no nvec_prof.csv (correct)"; fi

echo "=== leg-c: patched, SHUD_NVEC_PROF=0 ==="
rm -f "$OUT"/nvec_prof.csv 2>/dev/null || true
( cd "$BASIN" && SHUD_NVEC_PROF=0 /tmp/shud_patched keliya >/dev/null 2>"$EVID/leg-c.log" )
manifest "$SHADIR/leg-c-zero.sha"
if [ -f "$OUT/nvec_prof.csv" ]; then echo "LEG-C FAIL: nvec_prof.csv created with gate=0" ; else echo "leg-c: no nvec_prof.csv (correct, strict =1 proven)"; fi

echo "=== leg-d: patched, SHUD_NVEC_PROF=1 ==="
rm -f "$OUT"/nvec_prof.csv 2>/dev/null || true
( cd "$BASIN" && SHUD_NVEC_PROF=1 /tmp/shud_patched keliya >"$EVID/leg-d.stdout.log" 2>"$EVID/leg-d.log" )
manifest "$SHADIR/leg-d-one.sha"
if [ -f "$OUT/nvec_prof.csv" ]; then echo "leg-d: nvec_prof.csv created (correct)"; cp "$OUT/nvec_prof.csv" "$EVID/csv_sample/keliya_nvec_prof.csv"; else echo "LEG-D FAIL: nvec_prof.csv NOT created with gate=1"; fi
# capture the clone-assert + dump stdout lines
grep -i 'NVEC_PROF' "$EVID/leg-d.stdout.log" > "$EVID/selfcheck/nvec_prof_stdout.txt" || true

echo
echo "=== COMPARE manifests (strip filename col; compare hashes+names) ==="
cmp_sha() {
    # returns 0 if the two manifests are identical
    diff <(sort "$1") <(sort "$2") >/dev/null 2>&1
}
BASE="$SHADIR/leg-a-baseline.sha"
echo "leg-a run1 vs baseline (determinism): $(cmp_sha "$SHADIR/leg-a-run1.sha" "$BASE" && echo PASS || echo DIFFER)"
echo "leg-b (unset)   vs baseline: $(cmp_sha "$SHADIR/leg-b-unset.sha" "$BASE" && echo PASS || echo FAIL)"
echo "leg-c (=0)      vs baseline: $(cmp_sha "$SHADIR/leg-c-zero.sha"  "$BASE" && echo PASS || echo FAIL)"
echo "leg-d (=1)      vs baseline: $(cmp_sha "$SHADIR/leg-d-one.sha"   "$BASE" && echo PASS || echo FAIL)"
echo
echo "model-output files in manifest: $(wc -l < "$BASE")"
echo "DONE."
