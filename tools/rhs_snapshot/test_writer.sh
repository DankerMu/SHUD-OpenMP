#!/bin/sh
# test_writer.sh — 4-scenario acceptance test for compose_snapshot_filename().
#
# Scenarios (PR #54 round-1 F9 fix):
#   (a) Empty suffix → "snapshot_t<t>.bin" (legacy form, PR #53 back-compat)
#   (b) Non-empty suffix → "snapshot_t<t>_<suffix>.bin"
#   (c) `/` in suffix → rejected (compose returns empty → harness exit 1)
#   (d) `\\` in suffix → rejected (path-traversal guard)
#
# Mirrors the test pattern in tools/cvode_stats_diff/test_cvode_stats_diff.sh.
#
# Owned by: openspec change s1-rhs-core-extraction (Group 0 task 0.1b.i).

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
HARNESS_SRC="$SCRIPT_DIR/test_writer_main.cpp"
WRITER_SRC="$SCRIPT_DIR/writer.cpp"

[ -r "$HARNESS_SRC" ] || { printf "FATAL: %s missing\n" "$HARNESS_SRC" >&2; exit 2; }
[ -r "$WRITER_SRC" ]  || { printf "FATAL: %s missing\n" "$WRITER_SRC"  >&2; exit 2; }

# Pick a compiler: prefer g++, fall back to clang++ (matches the project's
# Linux-vs-macOS convention without requiring Makefile plumbing).
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    if command -v g++ >/dev/null 2>&1; then
        CXX=g++
    elif command -v clang++ >/dev/null 2>&1; then
        CXX=clang++
    else
        printf "FATAL: no g++ or clang++ in PATH\n" >&2
        exit 2
    fi
fi

TMP=$(mktemp -d -t rhs_snapshot_test_writer.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

BIN="$TMP/test_writer"
"$CXX" -std=c++17 -O0 -Wall -I"$SCRIPT_DIR" \
    -o "$BIN" "$HARNESS_SRC" "$WRITER_SRC"

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

assert_stdout() {
    label="$1"; shift
    expect="$1"; shift
    actual="$1"; shift
    if [ "$actual" = "$expect" ]; then
        printf "PASS  %s  (stdout=%s)\n" "$label" "$actual"
        PASS=$((PASS + 1))
    else
        printf "FAIL  %s  (expect_stdout=%s actual=%s)\n" "$label" "$expect" "$actual"
        FAIL=$((FAIL + 1))
    fi
}

# -----------------------------------------------------------------------------
# Scenario (a): empty suffix → legacy filename
# -----------------------------------------------------------------------------
set +e
out_a=$("$BIN" 86400 "" 2>&1); rc_a=$?
set -e
assert_exit 0 "scenario(a) empty suffix → exit 0" "$rc_a"
assert_stdout "scenario(a) empty suffix → legacy filename" "snapshot_t86400.bin" "$out_a"

# -----------------------------------------------------------------------------
# Scenario (b): non-empty suffix → snapshot_t<t>_<suffix>.bin
# -----------------------------------------------------------------------------
set +e
out_b=$("$BIN" 2592000 "before_passvalue" 2>&1); rc_b=$?
set -e
assert_exit 0 "scenario(b) non-empty suffix → exit 0" "$rc_b"
assert_stdout "scenario(b) suffixed filename" "snapshot_t2592000_before_passvalue.bin" "$out_b"

# -----------------------------------------------------------------------------
# Scenario (c): `/` in suffix → rejected
# -----------------------------------------------------------------------------
set +e
out_c=$("$BIN" 86400 "bad/path" 2>&1); rc_c=$?
set -e
assert_exit 1 "scenario(c) suffix with '/' → exit 1 (rejected)" "$rc_c"

# -----------------------------------------------------------------------------
# Scenario (d): `\\` in suffix → rejected
# -----------------------------------------------------------------------------
set +e
out_d=$("$BIN" 86400 "bad\\sep" 2>&1); rc_d=$?
set -e
assert_exit 1 "scenario(d) suffix with '\\\\' → exit 1 (rejected)" "$rc_d"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
printf "\nTEST SUMMARY  PASS=%d  FAIL=%d\n" "$PASS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
