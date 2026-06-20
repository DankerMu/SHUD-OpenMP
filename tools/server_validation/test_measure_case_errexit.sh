#!/bin/bash
#
# test_measure_case_errexit.sh — proves Summary block runs to completion
# under set -euo pipefail even when measure_case returns 1, per spec
# wallclock-script-result-always-emitted (#137).
#
# Strategy: emulate the script's call-loop + Summary block in a child
# bash -c process with stub measure_case, assert FAIL→exit-1 path emits
# RESULT row. Child process is used (vs in-process subshell) so the
# parent test runner's `set -euo pipefail` does not suppress errexit
# inside the subshell via the POSIX "LHS of && / || disables errexit"
# rule (which bash propagates into the subshell's execution context).

set -euo pipefail

ARCHIVE_DIR=$(mktemp -d)
trap 'rm -rf "$ARCHIVE_DIR"' EXIT

CASES_OMP_TARGET="qinyijiang heihe_x4"
CASE_HEIHE="heihe"

# ---- Test 1: stub returns 1 → || true protects → Summary runs → exit 1 ----
ARCHIVE_DIR="$ARCHIVE_DIR" \
CASES_OMP_TARGET="$CASES_OMP_TARGET" \
CASE_HEIHE="$CASE_HEIHE" \
bash -c '
set -euo pipefail
FAIL=0
measure_case() {
    FAIL=$((FAIL+1))
    return 1
}
: > "$ARCHIVE_DIR/acceptance_summary.txt"
for case in $CASES_OMP_TARGET; do
    measure_case "$case" "1.5" || true
done
measure_case "$CASE_HEIHE" "0.95" || true
# Summary block (emulated):
if [ "$FAIL" -ne 0 ]; then
    echo "RESULT: FAIL ($FAIL case(s) below target)" \
        | tee -a "$ARCHIVE_DIR/acceptance_summary.txt" >&2
    exit 1
fi
echo "RESULT: PASS" | tee -a "$ARCHIVE_DIR/acceptance_summary.txt"
exit 0
' > /dev/null 2>&1 && test1_rc=0 || test1_rc=$?
[ "$test1_rc" = "1" ] || { echo "FAIL: expected exit 1, got $test1_rc" >&2; exit 1; }
grep -q '^RESULT: FAIL' "$ARCHIVE_DIR/acceptance_summary.txt" || {
    echo "FAIL: RESULT: FAIL row missing" >&2
    cat "$ARCHIVE_DIR/acceptance_summary.txt" >&2
    exit 1
}
echo "PASS: stub fail → || true → Summary runs → exit 1 + RESULT: FAIL row"

: > "$ARCHIVE_DIR/acceptance_summary.txt"

# ---- Test 2: stub returns 0 → all pass → exit 0 + RESULT: PASS ----
ARCHIVE_DIR="$ARCHIVE_DIR" \
CASES_OMP_TARGET="$CASES_OMP_TARGET" \
CASE_HEIHE="$CASE_HEIHE" \
bash -c '
set -euo pipefail
FAIL=0
measure_case() { return 0; }
: > "$ARCHIVE_DIR/acceptance_summary.txt"
for case in $CASES_OMP_TARGET; do
    measure_case "$case" "1.5" || true
done
measure_case "$CASE_HEIHE" "0.95" || true
if [ "$FAIL" -ne 0 ]; then
    echo "RESULT: FAIL ($FAIL case(s))" | tee -a "$ARCHIVE_DIR/acceptance_summary.txt" >&2
    exit 1
fi
echo "RESULT: PASS (heihe_x4 + qinyijiang >= 1.5x, heihe >= 0.95x)" \
    | tee -a "$ARCHIVE_DIR/acceptance_summary.txt"
exit 0
' > /dev/null 2>&1 && test2_rc=0 || test2_rc=$?
[ "$test2_rc" = "0" ] || { echo "FAIL: expected exit 0, got $test2_rc" >&2; exit 1; }
grep -q '^RESULT: PASS' "$ARCHIVE_DIR/acceptance_summary.txt" || {
    echo "FAIL: RESULT: PASS row missing" >&2
    exit 1
}
echo "PASS: stub pass → exit 0 + RESULT: PASS row"

# ---- Test 3: WITHOUT || true → errexit fires → exit 1 (sanity check) ----
ARCHIVE_DIR="$ARCHIVE_DIR" \
CASES_OMP_TARGET="$CASES_OMP_TARGET" \
bash -c '
set -euo pipefail
FAIL=0
measure_case() {
    FAIL=$((FAIL+1))
    return 1
}
: > "$ARCHIVE_DIR/acceptance_summary.txt"
for case in $CASES_OMP_TARGET; do
    measure_case "$case" "1.5"   # NO || true
done
if [ "$FAIL" -ne 0 ]; then
    echo "RESULT: FAIL ($FAIL case(s))" | tee -a "$ARCHIVE_DIR/acceptance_summary.txt" >&2
    exit 1
fi
echo "RESULT: PASS" | tee -a "$ARCHIVE_DIR/acceptance_summary.txt"
exit 0
' > /dev/null 2>&1 && test3_rc=0 || test3_rc=$?
[ "$test3_rc" = "1" ] || { echo "FAIL: expected exit 1 from errexit, got $test3_rc" >&2; exit 1; }
# Without `|| true`, Summary block doesn't run → RESULT row missing
if grep -q '^RESULT:' "$ARCHIVE_DIR/acceptance_summary.txt"; then
    echo "FAIL: RESULT row should NOT exist without || true (sanity check failed)" >&2
    exit 1
fi
echo "PASS: WITHOUT || true → errexit early-exit, no RESULT row (confirms bug exists pre-fix)"

# ---- Test 4: grep that the actual sbatch has the || true guards ----
SBATCH="$(cd "$(dirname "$0")" && pwd)/run_p7_wallclock.sbatch"
[ -r "$SBATCH" ] || { echo "FAIL: sbatch not readable" >&2; exit 1; }
guard_count=$(grep -cE '^[[:space:]]*measure_case[[:space:]].*\|\|[[:space:]]*true[[:space:]]*$' "$SBATCH")
[ "$guard_count" = "2" ] || {
    echo "FAIL: expected exactly 2 'measure_case ... || true' lines, got $guard_count" >&2
    exit 1
}
echo "PASS: sbatch has exactly 2 'measure_case ... || true' guards"

echo ""
echo "ALL TESTS PASS"
exit 0
