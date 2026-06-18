#!/bin/sh
# test_cvode_stats_diff.sh — 4-scenario acceptance test for cvode_stats_diff.sh.
#
# Scenarios (per openspec/.../tasks.md task 0.2 acceptance):
#   (a) Two identical files → exit 0
#   (b) Single key value mismatch → exit non-zero + "key: golden=X, new=Y"
#   (c) Missing key → exit non-zero + "key MISSING in <file>"
#   (d) Extra (unknown) key → exit non-zero + "key UNKNOWN in <file>: <key>"
#
# Owned by: openspec change s1-rhs-core-extraction (Group 0 task 0.2).

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DIFF="$SCRIPT_DIR/cvode_stats_diff.sh"

[ -x "$DIFF" ] || { printf "FATAL: %s not executable\n" "$DIFF" >&2; exit 2; }

TMP=$(mktemp -d -t cvode_stats_diff_test.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

# Canonical 16-key reference content (matches D10 spec exactly).
make_canonical() {
    cat > "$1" <<'EOF'
nfe=102485
nfeLS=106492
nni=102484
nli=106492
nsetups=0
netf=5
nst=101188
npe=0
nps=0
ncfn=198
ncfl=66
lenrw=23294
leniw=53
lenrwLS=21474
leniwLS=42
nFCall=99999
EOF
}

PASS=0
FAIL=0

assert_exit() {
    expect="$1"; shift
    label="$1"; shift
    actual="$1"; shift
    if [ "$actual" = "$expect" ]; then
        printf "PASS  %s  (exit=%s)\n" "$label" "$actual"
        PASS=$((PASS + 1))
    else
        printf "FAIL  %s  (expect_exit=%s actual=%s)\n" "$label" "$expect" "$actual"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    label="$1"; shift
    needle="$1"; shift
    file="$1"; shift
    if grep -qF -- "$needle" "$file"; then
        printf "PASS  %s  (output contains: %s)\n" "$label" "$needle"
        PASS=$((PASS + 1))
    else
        printf "FAIL  %s  (output missing: %s)\n" "$label" "$needle"
        printf "      actual output:\n"
        sed 's/^/        /' "$file"
        FAIL=$((FAIL + 1))
    fi
}

# -----------------------------------------------------------------------------
# Scenario (a): identical files → exit 0
# -----------------------------------------------------------------------------
make_canonical "$TMP/a_golden.txt"
make_canonical "$TMP/a_new.txt"
set +e
"$DIFF" "$TMP/a_new.txt" "$TMP/a_golden.txt" > "$TMP/a.out" 2>&1
rc=$?
set -e
assert_exit 0 "scenario(a) identical → exit 0" "$rc"
if [ -s "$TMP/a.out" ]; then
    printf "INFO  scenario(a) produced output (should be empty):\n"
    sed 's/^/        /' "$TMP/a.out"
fi

# -----------------------------------------------------------------------------
# Scenario (b): single key value mismatch → exit non-zero + format-compliant
# -----------------------------------------------------------------------------
make_canonical "$TMP/b_golden.txt"
make_canonical "$TMP/b_new.txt"
# Mutate one key in the new file: nfe=102485 → nfe=99999
sed -i.bak 's/^nfe=.*/nfe=99999/' "$TMP/b_new.txt" && rm -f "$TMP/b_new.txt.bak"
set +e
"$DIFF" "$TMP/b_new.txt" "$TMP/b_golden.txt" > "$TMP/b.out" 2>&1
rc=$?
set -e
assert_exit 1 "scenario(b) single mismatch → exit non-zero" "$rc"
assert_contains "scenario(b) format-compliant output" "nfe: golden=102485, new=99999" "$TMP/b.out"

# -----------------------------------------------------------------------------
# Scenario (c): missing key → exit non-zero + MISSING report
# -----------------------------------------------------------------------------
make_canonical "$TMP/c_golden.txt"
make_canonical "$TMP/c_new.txt"
# Remove one key from the new file: drop the `npe=...` line.
grep -v '^npe=' "$TMP/c_new.txt" > "$TMP/c_new.txt.tmp"
mv "$TMP/c_new.txt.tmp" "$TMP/c_new.txt"
set +e
"$DIFF" "$TMP/c_new.txt" "$TMP/c_golden.txt" > "$TMP/c.out" 2>&1
rc=$?
set -e
assert_exit 1 "scenario(c) missing key → exit non-zero" "$rc"
assert_contains "scenario(c) MISSING report" "npe MISSING in $TMP/c_new.txt" "$TMP/c.out"

# -----------------------------------------------------------------------------
# Scenario (d): extra (unknown) key → exit non-zero + UNKNOWN report
# -----------------------------------------------------------------------------
make_canonical "$TMP/d_golden.txt"
make_canonical "$TMP/d_new.txt"
# Add a non-canonical key to the new file.
printf 'bogus_extra=42\n' >> "$TMP/d_new.txt"
set +e
"$DIFF" "$TMP/d_new.txt" "$TMP/d_golden.txt" > "$TMP/d.out" 2>&1
rc=$?
set -e
assert_exit 1 "scenario(d) unknown key → exit non-zero" "$rc"
assert_contains "scenario(d) UNKNOWN report" "key UNKNOWN in $TMP/d_new.txt: bogus_extra" "$TMP/d.out"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
printf "\nTEST SUMMARY  PASS=%d  FAIL=%d\n" "$PASS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
